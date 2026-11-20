# RUN: %PYTHON %s

import numpy as np
from mlir import ir
from mlir.dialects import transform
from mlir.passmanager import PassManager

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform.transform_ext.utils.subgraph_analysis import (
    SUBGRAPH_KIND_ATTR_NAME,
    tag_subgraphs,
)
from lighthouse.execution.runner import Runner
from lighthouse.schedule import schedule_boilerplate
from lighthouse.schedule.xegpu.wg_tiling_schedule import apply_subgraph_tiling

SOURCE = """
#tokens = affine_map<(b,t,h) -> (b,t)>
#output = affine_map<(b,t,h) -> (b,t,h)>
func.func @lookup(%ids: tensor<2x3xi64>, %table: tensor<5x512xbf16>,
                  %dest: memref<2x3x512xbf16>) {
  %limit = arith.constant 5 : index
  %zero = arith.constant 0 : i64
  %empty = tensor.empty() : tensor<2x3x512xbf16>
  %result = linalg.generic {indexing_maps=[#tokens,#output],
      iterator_types=["parallel","parallel","parallel"]}
    ins(%ids : tensor<2x3xi64>) outs(%empty : tensor<2x3x512xbf16>) {
  ^bb0(%id: i64, %out: bf16):
    %row = arith.index_cast %id : i64 to index
    %column = linalg.index 2 : index
    %upper = arith.cmpi slt, %row, %limit : index
    cf.assert %upper, "token id too large"
    %lower = arith.cmpi sge, %id, %zero : i64
    cf.assert %lower, "negative token id"
    %value = tensor.extract %table[%row,%column] : tensor<5x512xbf16>
    linalg.yield %value : bf16
  } -> tensor<2x3x512xbf16>
  bufferization.materialize_in_destination %result in restrict writable %dest
    : (tensor<2x3x512xbf16>, memref<2x3x512xbf16>) -> ()
  return
}
"""


def apply(source):
    module = ir.Module.parse(source)
    with schedule_boilerplate() as (schedule, sequence):
        func = lh_transform.match_op(sequence.bodyTarget, "func.func").results[0]
        apply_subgraph_tiling(func)
        transform.yield_([])
    schedule.body.operations[0].apply(module.operation)
    assert module.operation.verify()
    return module


with ir.Context(), ir.Location.unknown():
    lh_dialects.register_and_load()
    for source, expected_kind in (
        (SOURCE, "checked_embedding"),
        (SOURCE.replace("%row, %limit", "%column, %limit"), "elementwise"),
        (SOURCE.replace("x512xbf16", "x513xbf16"), "elementwise"),
    ):
        tagged = ir.Module.parse(source)
        block = tagged.body.operations[0].regions[0].blocks[0]
        assert not tag_subgraphs(block, ["checked_embedding", "elementwise"])
        (lookup,) = (
            op for op in block.operations if op.operation.name == "linalg.generic"
        )
        assert (
            ir.StringAttr(lookup.attributes[SUBGRAPH_KIND_ATTR_NAME]).value
            == expected_kind
        )

    module = apply(SOURCE)
    text = str(module)
    assert "linalg.generic" not in text
    assert "cf.assert" not in text
    assert "arith.cmpi" not in text
    assert "vector<256xbf16>" in text

    PassManager.parse(
        "builtin.module(empty-tensor-to-alloc-tensor,"
        "one-shot-bufferize{bufferize-function-boundaries "
        "function-boundary-type-conversion=identity-layout-map},"
        "scf-forall-to-for,buffer-deallocation-pipeline,"
        "convert-vector-to-scf,expand-strided-metadata,lower-affine,convert-scf-to-cf)"
    ).run(module.operation)
    Runner.make_function_callable(module, "lookup")
    PassManager.parse(
        "builtin.module(convert-vector-to-llvm,finalize-memref-to-llvm,convert-to-llvm,"
        "reconcile-unrealized-casts)"
    ).run(module.operation)
    ids = np.array([[4, 0, 3], [1, 4, 2]], dtype=np.int64)
    # Compare BF16 payloads as raw bits; the kernel only copies them.
    table = np.arange(5 * 512, dtype=np.uint16).reshape(5, 512)
    result = np.empty((2, 3, 512), dtype=np.uint16)
    Runner(module).execute("lookup", [ids, table, result])
    np.testing.assert_array_equal(result, table[ids])
