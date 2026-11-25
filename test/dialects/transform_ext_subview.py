# RUN: %PYTHON %s | FileCheck %s

"""Tests for the lighthouse transform_ext dialect operations."""

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured as transform_structured
import lighthouse.dialects as lh_dialects
from lighthouse.dialects.transform import transform_ext


def run(payload_fn):
    def decorator(f):
        print("Test: ", f.__name__, flush=True)
        with ir.Context(), ir.Location.unknown():
            lh_dialects.register_and_load()

            mod = ir.Module.create()
            mod.operation.attributes["transform.with_named_sequence"] = (
                ir.UnitAttr.get()
            )
            with ir.InsertionPoint(mod.body):
                named_seq = transform.named_sequence(
                    "main", [transform.AnyOpType.get()], []
                )
                any_op = transform.AnyOpType.get()
                with ir.InsertionPoint(named_seq.body):
                    func_funcs = transform_structured.structured_match(
                        any_op, named_seq.bodyTarget, ops={"func.func"}
                    )
                    payload_handle = transform.get_parent_op(any_op, func_funcs)

                    f(payload_handle)
                    transform.yield_()

                named_seq.verify()

                try:
                    payload_mod = payload_fn()
                    named_seq.apply(payload_mod)
                except Exception as e:
                    print(f"Caught exception: {e}", flush=True)
        return f

    return decorator


def payload_read():
    mod = ir.Module.parse(
        r"""
func.func @move_read_offsets(%arg0: memref<20x4096x4096xbf16>) -> vector<32x32xbf16> {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = vector.transfer_read %arg0[%c16, %c8, %c16], %cst
        {in_bounds = [true, true],
        permutation_map = affine_map<(d0, d1, d2) -> (d1, d2)>}
        : memref<20x4096x4096xbf16>, vector<32x32xbf16>
    return %0 : vector<32x32xbf16>
}
    """
    )

    return mod


# CHECK-LABEL: Test: test_move_read_offsets
@run(payload_read)
def test_move_read_offsets(payload_handle):
    any_op = transform.AnyOpType.get()
    transfers = transform_structured.structured_match(
        any_op, payload_handle, ops={"vector.transfer_read"}
    )
    transform_ext.move_offsets_to_subview(transfers)

    # CHECK-LABEL: @move_read_offsets(
    # CHECK-SAME: %[[ARG0:.+]]: memref<20x4096x4096xbf16>) -> vector<32x32xbf16> {
    # CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
    # CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
    # CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
    # CHECK: %[[SUBVIEW:.+]] = memref.subview %[[ARG0]][%[[C0]], %[[C0]], %[[C0]]] [20, 4096, 4096] [1, 1, 1]
    # CHECK: vector.transfer_read %[[SUBVIEW]][%[[C16]], %[[C8]], %[[C16]]]
    transform.print_()


def payload_write():
    mod = ir.Module.parse(
        r"""
func.func @move_write_offsets(%arg0: memref<32x4096x4096xbf16>, %arg1: vector<32x32xbf16>) {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    vector.transfer_write %arg1, %arg0[%c16, %c8, %c16]
        {in_bounds = [true, true],
        permutation_map = affine_map<(d0, d1, d2) -> (d1, d2)>}
        : vector<32x32xbf16>, memref<32x4096x4096xbf16>
    return
}
    """
    )

    return mod


# CHECK-LABEL: Test: test_move_write_offsets
@run(payload_write)
def test_move_write_offsets(payload_handle):
    any_op = transform.AnyOpType.get()
    transfers = transform_structured.structured_match(
        any_op, payload_handle, ops={"vector.transfer_write"}
    )
    transform_ext.move_offsets_to_subview(transfers)

    # CHECK-LABEL: @move_write_offsets(
    # CHECK-SAME: %[[ARG0:.+]]: memref<32x4096x4096xbf16>, %[[ARG1:.+]]: vector<32x32xbf16>) {
    # CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
    # CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
    # CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
    # CHECK: %[[SUBVIEW:.+]] = memref.subview %[[ARG0]][%[[C0]], %[[C0]], %[[C0]]] [32, 4096, 4096] [1, 1, 1]
    # CHECK: vector.transfer_write %[[ARG1]], %[[SUBVIEW]][%[[C16]], %[[C8]], %[[C16]]]
    transform.print_()


def payload_read_permutation():
    mod = ir.Module.parse(
        r"""
func.func @negative_non_minor_identity(%arg0: memref<20x4096x4096xbf16>) -> vector<16x16xbf16> {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = vector.transfer_read %arg0[%c0, %c8, %c16], %cst
        {in_bounds = [true, true],
        permutation_map = affine_map<(d0, d1, d2) -> (d2, d0)>}
        : memref<20x4096x4096xbf16>, vector<16x16xbf16>
    return %0 : vector<16x16xbf16>
}
    """
    )

    return mod


# CHECK-LABEL: Test: test_negative_non_minor_identity
@run(payload_read_permutation)
def test_negative_non_minor_identity(payload_handle):
    any_op = transform.AnyOpType.get()
    transfers = transform_structured.structured_match(
        any_op, payload_handle, ops={"vector.transfer_read"}
    )
    transform_ext.move_offsets_to_subview(transfers)

    # CHECK-LABEL: @negative_non_minor_identity(
    # CHECK-NOT: memref.subview
    # CHECK: vector.transfer_read
    transform.print_()


def payload_read_dynamic_shape():
    mod = ir.Module.parse(
        r"""
func.func @negative_dynamic_shape(%arg0: memref<?x4096x4096xbf16>) -> vector<32x32xbf16> {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = vector.transfer_read %arg0[%c16, %c8, %c16], %cst
        {in_bounds = [true, true],
        permutation_map = affine_map<(d0, d1, d2) -> (d1, d2)>}
        : memref<?x4096x4096xbf16>, vector<32x32xbf16>
    return %0 : vector<32x32xbf16>
}
    """
    )

    return mod


# CHECK-LABEL: Test: test_negative_dynamic_shape
@run(payload_read_dynamic_shape)
def test_negative_dynamic_shape(payload_handle):
    any_op = transform.AnyOpType.get()
    transfers = transform_structured.structured_match(
        any_op, payload_handle, ops={"vector.transfer_read"}
    )
    transform_ext.move_offsets_to_subview(transfers)

    # CHECK-LABEL: @negative_dynamic_shape(
    # CHECK-NOT: memref.subview
    # CHECK: vector.transfer_read
    transform.print_()


def payload_write_dynamic_strides():
    mod = ir.Module.parse(
        r"""
func.func @negative_dynamic_strides(
    %arg0: memref<32x4096x4096xbf16, strided<[?, ?, 1]>>,
    %arg1: vector<32x32xbf16>) {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    vector.transfer_write %arg1, %arg0[%c16, %c8, %c16]
        {in_bounds = [true, true],
        permutation_map = affine_map<(d0, d1, d2) -> (d1, d2)>}
        : vector<32x32xbf16>, memref<32x4096x4096xbf16, strided<[?, ?, 1]>>
    return
}
    """
    )

    return mod


# CHECK-LABEL: Test: test_negative_dynamic_strides
@run(payload_write_dynamic_strides)
def test_negative_dynamic_strides(payload_handle):
    any_op = transform.AnyOpType.get()
    transfers = transform_structured.structured_match(
        any_op, payload_handle, ops={"vector.transfer_write"}
    )
    transform_ext.move_offsets_to_subview(transfers)

    # CHECK-LABEL: @negative_dynamic_strides(
    # CHECK-NOT: memref.subview
    # CHECK: vector.transfer_write
    transform.print_()


def payload_rank_reduce_read():
    return ir.Module.parse(
        r"""
func.func @rank_reduce_read(%arg0: memref<8x256x128xbf16>, %k: index) -> vector<8x32xbf16> {
    %c0 = arith.constant 0 : index
    %c255 = arith.constant 255 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = vector.transfer_read %arg0[%c0, %c255, %k], %cst
        {in_bounds = [true, true],
        permutation_map = affine_map<(d0, d1, d2) -> (d0, d2)>}
        : memref<8x256x128xbf16>, vector<8x32xbf16>
    return %0 : vector<8x32xbf16>
}
    """
    )


# CHECK-LABEL: Test: test_rank_reduce_read
@run(payload_rank_reduce_read)
def test_rank_reduce_read(payload_handle):
    any_op = transform.AnyOpType.get()
    transfers = transform_structured.structured_match(
        any_op, payload_handle, ops={"vector.transfer_read"}
    )
    transform_ext.move_offsets_to_subview(transfers, rank_reduce=True)

    # CHECK-LABEL: @rank_reduce_read(
    # CHECK-SAME: %[[ARG0:.+]]: memref<8x256x128xbf16>, %[[K:.+]]: index)
    # CHECK: %[[VIEW:.+]] = memref.subview %[[ARG0]][0, 255, 0] [8, 1, 128] [1, 1, 1]
    # CHECK-SAME: to memref<8x128xbf16, strided<[32768, 1], offset: 32640>>
    # CHECK: vector.transfer_read %[[VIEW]][%{{.+}}, %[[K]]], %{{.+}} {in_bounds = [true, true]}
    # CHECK-SAME: memref<8x128xbf16, strided<[32768, 1], offset: 32640>>, vector<8x32xbf16>
    transform.print_()


def payload_rank_reduce_dynamic_write():
    return ir.Module.parse(
        r"""
func.func @rank_reduce_dynamic_write(%arg0: memref<8x256x128xbf16>, %s: index, %v: vector<8x32xbf16>) {
    %c0 = arith.constant 0 : index
    vector.transfer_write %v, %arg0[%c0, %s, %c0]
        {in_bounds = [true, true],
        permutation_map = affine_map<(d0, d1, d2) -> (d0, d2)>}
        : vector<8x32xbf16>, memref<8x256x128xbf16>
    return
}
    """
    )


# CHECK-LABEL: Test: test_rank_reduce_dynamic_write
@run(payload_rank_reduce_dynamic_write)
def test_rank_reduce_dynamic_write(payload_handle):
    any_op = transform.AnyOpType.get()
    transfers = transform_structured.structured_match(
        any_op, payload_handle, ops={"vector.transfer_write"}
    )
    transform_ext.move_offsets_to_subview(transfers, rank_reduce=True)

    # CHECK-LABEL: @rank_reduce_dynamic_write(
    # CHECK-SAME: %[[ARG0:.+]]: memref<8x256x128xbf16>, %[[S:.+]]: index,
    # CHECK: %[[VIEW:.+]] = memref.subview %[[ARG0]][0, %[[S]], 0] [8, 1, 128] [1, 1, 1]
    # CHECK-SAME: to memref<8x128xbf16, strided<[32768, 1], offset: ?>>
    # CHECK: vector.transfer_write %{{.+}}, %[[VIEW]][%{{.+}}, %{{.+}}] {in_bounds = [true, true]}
    # CHECK-SAME: vector<8x32xbf16>, memref<8x128xbf16, strided<[32768, 1], offset: ?>>
    transform.print_()


def payload_negative_rank_reduce():
    return ir.Module.parse(
        r"""
func.func @negative_rank_reduce(%arg0: memref<8x256x128xbf16>) -> (vector<32x8xbf16>, vector<8x32xbf16>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst
        {in_bounds = [true, true],
        permutation_map = affine_map<(d0, d1, d2) -> (d2, d0)>}
        : memref<8x256x128xbf16>, vector<32x8xbf16>
    %1 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst
        {in_bounds = [true, true],
        permutation_map = affine_map<(d0, d1, d2) -> (d1, d2)>}
        : memref<8x256x128xbf16>, vector<8x32xbf16>
    return %0, %1 : vector<32x8xbf16>, vector<8x32xbf16>
}
    """
    )


# Transposing transfers are not rank-reduced; with rank_reduce, minor-identity
# transfers are left to the default mode.
# CHECK-LABEL: Test: test_negative_rank_reduce
@run(payload_negative_rank_reduce)
def test_negative_rank_reduce(payload_handle):
    any_op = transform.AnyOpType.get()
    transfers = transform_structured.structured_match(
        any_op, payload_handle, ops={"vector.transfer_read"}
    )
    transform_ext.move_offsets_to_subview(transfers, rank_reduce=True)

    # CHECK-LABEL: @negative_rank_reduce(
    # CHECK-NOT: memref.subview
    # CHECK: vector.transfer_read %{{.+}} permutation_map
    # CHECK-NOT: memref.subview
    # CHECK: vector.transfer_read %{{.+}} : memref<8x256x128xbf16>, vector<8x32xbf16>
    transform.print_()


def payload_negative_rank_reduce_masked():
    return ir.Module.parse(
        r"""
func.func @negative_rank_reduce_masked(%arg0: memref<8x256x128xbf16>, %m: vector<8x32xi1>) -> vector<8x32xbf16> {
    %c0 = arith.constant 0 : index
    %c255 = arith.constant 255 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = vector.transfer_read %arg0[%c0, %c255, %c0], %cst, %m
        {in_bounds = [true, true],
        permutation_map = affine_map<(d0, d1, d2) -> (d0, d2)>}
        : memref<8x256x128xbf16>, vector<8x32xbf16>
    return %0 : vector<8x32xbf16>
}
    """
    )


# Masked transfers keep their original form.
# CHECK-LABEL: Test: test_negative_rank_reduce_masked
@run(payload_negative_rank_reduce_masked)
def test_negative_rank_reduce_masked(payload_handle):
    any_op = transform.AnyOpType.get()
    transfers = transform_structured.structured_match(
        any_op, payload_handle, ops={"vector.transfer_read"}
    )
    transform_ext.move_offsets_to_subview(transfers, rank_reduce=True)

    # CHECK-LABEL: @negative_rank_reduce_masked(
    # CHECK-NOT: memref.subview
    # CHECK: vector.transfer_read %{{.+}}, %{{.+}}, %{{.+}} {{.*}}permutation_map
    # CHECK-SAME: memref<8x256x128xbf16>, vector<8x32xbf16>
    transform.print_()
