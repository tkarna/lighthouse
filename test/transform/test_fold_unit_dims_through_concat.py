# RUN: %PYTHON %s | FileCheck %s

"""Check rank reduction of concats and preservation of other users."""

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule import schedule_boilerplate
from lighthouse.transform import cleanup


SOURCE = """
func.func @transpose_concat(%arg: tensor<3x5xf32>) -> tensor<5x6xf32> {
  %a = tensor.expand_shape %arg [[0, 1], [2]] output_shape [1, 3, 5]
    : tensor<3x5xf32> into tensor<1x3x5xf32>
  %e = tensor.empty() : tensor<1x5x3xf32>
  %b = linalg.transpose ins(%a : tensor<1x3x5xf32>)
    outs(%e : tensor<1x5x3xf32>) permutation = [0, 2, 1]
  %c = tensor.concat dim(2) %b, %b
    : (tensor<1x5x3xf32>, tensor<1x5x3xf32>) -> tensor<1x5x6xf32>
  %d = tensor.collapse_shape %c [[0, 1], [2]]
    : tensor<1x5x6xf32> into tensor<5x6xf32>
  return %d : tensor<5x6xf32>
}

func.func @shared_concat(%a: tensor<2x1x3xf32>, %b: tensor<2x1x4xf32>)
    -> (tensor<2x7xf32>, tensor<2x1x7xf32>) {
  %c = tensor.concat dim(2) %a, %b
    : (tensor<2x1x3xf32>, tensor<2x1x4xf32>) -> tensor<2x1x7xf32>
  %d = tensor.collapse_shape %c [[0], [1, 2]]
    : tensor<2x1x7xf32> into tensor<2x7xf32>
  return %d, %c : tensor<2x7xf32>, tensor<2x1x7xf32>
}

func.func @merge_nonunit(%a: tensor<2x3xf32>, %b: tensor<2x4xf32>)
    -> tensor<14xf32> {
  %c = tensor.concat dim(1) %a, %b
    : (tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x7xf32>
  %d = tensor.collapse_shape %c [[0, 1]]
    : tensor<2x7xf32> into tensor<14xf32>
  return %d : tensor<14xf32>
}
"""

with ir.Context(), ir.Location.unknown():
    lh_dialects.register_and_load()
    module = ir.Module.parse(SOURCE)
    with schedule_boilerplate() as (schedule, sequence):
        target = transform_ext.fold_unit_dims_through_concat(sequence.bodyTarget)
        cleanup(target)
        transform.yield_([])
    schedule.body.operations[0].apply(module.operation)
    assert module.operation.verify()
    print(module)

# CHECK-LABEL: func.func @transpose_concat
# CHECK-NOT: tensor.expand_shape
# CHECK: %[[T:.*]] = linalg.transpose ins({{.*}} : tensor<3x5xf32>)
# CHECK-SAME: permutation = [1, 0]
# CHECK: %[[C:.*]] = tensor.concat dim(1) %[[T]], %[[T]]
# CHECK-NOT: tensor.collapse_shape
# CHECK: return %[[C]] : tensor<5x6xf32>

# CHECK-LABEL: func.func @shared_concat
# CHECK: %[[ORIG:.*]] = tensor.concat dim(2)
# CHECK: %[[LOWER:.*]] = tensor.concat dim(1)
# CHECK: return %[[LOWER]], %[[ORIG]] : tensor<2x7xf32>, tensor<2x1x7xf32>

# CHECK-LABEL: func.func @merge_nonunit
# CHECK: %[[MERGE:.*]] = tensor.concat dim(1)
# CHECK: tensor.collapse_shape %[[MERGE]]
