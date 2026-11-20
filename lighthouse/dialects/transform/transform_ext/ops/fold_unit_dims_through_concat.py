"""Fold singleton-only tensor collapses through concatenation."""

from math import prod

from mlir import ir
from mlir.dialects import ext, linalg, tensor, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.utils.mlir import defining_op


def _static_shape(value):
    ty = value.type
    if not isinstance(ty, ir.RankedTensorType) or ty.encoding is not None:
        return None
    shape = list(ty.shape)
    return shape if all(size > 0 for size in shape) else None


def _collapse_operand(value, groups, kept_axes, shape):
    ty = ir.RankedTensorType.get(
        [prod(shape[d] for d in g) for g in groups], value.type.element_type
    )
    producer = defining_op(value)
    if producer is not None and isinstance(producer.opview, linalg.TransposeOp):
        transpose = producer.opview
        source = defining_op(transpose.input)
        if source is not None and isinstance(source.opview, tensor.ExpandShapeOp):
            expand = source.opview
            input_groups = [list(map(int, group)) for group in expand.reassociation]
            input_shape = _static_shape(expand.src)
            expanded_shape = _static_shape(expand.result)
            if input_shape is not None and expanded_shape is not None:
                kept_inputs = [transpose.permutation[d] for d in kept_axes]
                permutation = [
                    i
                    for axis in kept_inputs
                    for i, group in enumerate(input_groups)
                    if axis in group
                ]
                if (
                    sorted(permutation) == list(range(len(input_shape)))
                    and [input_shape[i] for i in permutation] == list(ty.shape)
                    and all(
                        size == 1
                        for d, size in enumerate(expanded_shape)
                        if d not in kept_inputs
                    )
                ):
                    dest = tensor.EmptyOp(list(ty.shape), ty.element_type).result
                    return linalg.transpose(
                        expand.src, outs=[dest], permutation=permutation
                    )
    return tensor.CollapseShapeOp(ty, value, groups).result


def _fold_collapse(collapse, rewriter):
    owner = defining_op(collapse.src)
    if owner is None or not isinstance(owner.opview, tensor.ConcatOp):
        return
    concat = owner.opview
    shape = _static_shape(collapse.src)
    operand_shapes = [_static_shape(value) for value in concat.inputs]
    if shape is None or any(s is None for s in operand_shapes):
        return

    concat_axis = int(concat.dim)
    groups = [list(map(int, group)) for group in collapse.reassociation]
    if not groups:
        return
    kept_axes = []
    for group in groups:
        axis = (
            concat_axis
            if concat_axis in group
            else next((d for d in group if shape[d] != 1), group[0])
        )
        if any(shape[d] != 1 for d in group if d != axis):
            return
        kept_axes.append(axis)

    with collapse.location, ir.InsertionPoint(collapse):
        reduced = {}
        for value, operand_shape in zip(concat.inputs, operand_shapes):
            if value not in reduced:
                reduced[value] = _collapse_operand(
                    value, groups, kept_axes, operand_shape
                )
        replacement = tensor.ConcatOp(
            collapse.result.type,
            kept_axes.index(concat_axis),
            [reduced[value] for value in concat.inputs],
        )
    rewriter.replace_op(collapse, [replacement.result])


class FoldUnitDimsThroughConcatOp(
    TransformExtensionDialect.Operation, name="fold_unit_dims_through_concat"
):
    """Rebuild a collapsed concat at its result rank.

    Folds collapse_shape users of static, unencoded concats when the collapse
    removes only singleton axes other than the concat axis. Other users of the
    original concat are unchanged. Bypasses matching singleton expansions
    across named transposes by building a lower-rank transpose.

    Consumes the container handle and returns a fresh handle to the container.
    """

    target: ext.Operand[transform.AnyOpType]
    result: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(op, rewriter, results, state) -> DiagnosedSilenceableFailure:
            targets = [target.operation for target in state.get_payload_ops(op.target)]
            if len(targets) != 1 or targets[0].name not in (
                "func.func",
                "builtin.module",
            ):
                op.location.emit_error(
                    "fold_unit_dims_through_concat expects one func.func or builtin.module"
                )
                return DiagnosedSilenceableFailure.SilenceableFailure

            collapses = []

            def visit(candidate):
                if isinstance(candidate.opview, tensor.CollapseShapeOp):
                    collapses.append(candidate.opview)
                return ir.WalkResult.ADVANCE

            targets[0].walk(visit)
            for collapse in collapses:
                _fold_collapse(collapse, rewriter)
            results.set_ops(op.result, targets)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op) -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op):
            return (
                transform.consumes_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def fold_unit_dims_through_concat(target):
    """Fold singleton-only collapses into lower-rank concats."""
    return FoldUnitDimsThroughConcatOp(target=target).result
