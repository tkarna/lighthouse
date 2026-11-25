from mlir import ir
from mlir.dialects import ext, transform, vector, memref, arith
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


def get_base(op: ir.OpView) -> ir.Value:
    assert isinstance(op, (vector.TransferReadOp, vector.TransferWriteOp)), (
        "Expected vector transfer op"
    )
    return op.base


def get_vector_type(op: ir.OpView) -> ir.VectorType:
    if isinstance(op, vector.TransferReadOp):
        return op.vector.type
    elif isinstance(op, vector.TransferWriteOp):
        return op.valueToStore.type
    else:
        raise NotImplementedError("Unsupported op")


def get_indices(op: ir.OpView) -> ir.OpOperandList:
    assert isinstance(op, (vector.TransferReadOp, vector.TransferWriteOp)), (
        "Expected vector transfer op"
    )
    return op.indices


def get_permutation_map(op: ir.OpView) -> ir.AffineMapAttr:
    assert isinstance(op, (vector.TransferReadOp, vector.TransferWriteOp)), (
        "Expected vector transfer op"
    )
    return op.permutation_map


def get_in_bounds(op: ir.OpView) -> ir.ArrayAttr:
    assert isinstance(op, (vector.TransferReadOp, vector.TransferWriteOp)), (
        "Expected vector transfer op"
    )
    return op.in_bounds


class MoveOffsetsToSubviewOp(
    TransformExtensionDialect.Operation, name="move_offsets_to_subview"
):
    """
    Outlines indices from vector reads and writes to subviews.

    By default, non-constant indices of minor-identity transfers move to a
    subview, a support rewrite to simplify offsets analysis.

    With `rank_reduce`, a transfer whose permutation map is not a minor
    identity but drops source dimensions while keeping the others in order
    instead becomes a minor-identity transfer of a rank-reduced subview. The subview fixes the
    dropped dimensions at the transfer's indices; the remaining indices stay
    on the transfer. Other transfers are left unchanged. For example,

        vector.transfer_read %src[%c0, %c255, %k]
            {permutation_map = (d0, d1, d2) -> (d0, d2)}
            : memref<8x256x128xbf16>, vector<8x32xbf16>

    reads through a zero-copy strided view:

        %view = memref.subview %src[0, 255, 0] [8, 1, 128] [1, 1, 1]
            : memref<8x256x128xbf16>
            to memref<8x128xbf16, strided<[32768, 1], offset: 32640>>
        vector.transfer_read %view[%c0, %k] : ..., vector<8x32xbf16>

    Args:
        target: Handle to target op
        rank_reduce: Rewrite dimension-dropping transfers instead.
    Returns:
        Updated ops with offsets moved to subviews
    """

    target: ext.Operand[transform.AnyOpType]
    rank_reduce: ir.IntegerAttr = ext.attribute(
        default_factory=lambda: ir.IntegerAttr.get(ir.IntegerType.get_signless(1), 0)
    )
    updated_op: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    @staticmethod
    def create_subview(target: ir.OpView) -> ir.OpView | None:
        assert isinstance(target, (vector.TransferReadOp, vector.TransferWriteOp)), (
            "Expected vector transfer op"
        )

        base_val = get_base(target)
        assert isinstance(base_val.type, ir.MemRefType), (
            "Expected memref type for base value"
        )

        memref_type: ir.MemRefType = base_val.type

        # Bail on dynamic values.
        if not memref_type.has_static_shape:
            return None
        strides, _ = memref_type.get_strides_and_offset()
        dynamic_val = ir.ShapedType.get_dynamic_stride_or_offset()
        if dynamic_val in strides:
            return None

        vec_type: ir.VectorType = get_vector_type(target)

        map_attr = get_permutation_map(target)
        map: ir.AffineMap = map_attr.value
        if map != ir.AffineMap.get_minor_identity(map.n_dims, len(map.results)):
            return None

        sizes = memref_type.shape
        strides = [1] * memref_type.rank
        indices = get_indices(target)

        subview_offsets = []
        zero_offset = arith.ConstantOp(ir.IndexType.get(), 0)
        for offset in indices:
            if not isinstance(offset.owner, arith.ConstantOp):
                subview_offsets.append(offset)
            else:
                subview_offsets.append(zero_offset)

        base_subview = memref.subview(
            base_val,
            subview_offsets,
            sizes,
            strides,
        )

        transfer_indices = []
        for offset in indices:
            if isinstance(offset.owner, arith.ConstantOp):
                transfer_indices.append(offset)
            else:
                transfer_indices.append(zero_offset)
        transfer_op = (
            vector.transfer_read(
                vec_type,
                base_subview,
                transfer_indices,
                map_attr,
                target.padding,
                get_in_bounds(target),
            ).owner
            if isinstance(target, vector.TransferReadOp)
            else vector.transfer_write(
                None,
                target.valueToStore,
                base_subview,
                transfer_indices,
                map_attr,
                get_in_bounds(target),
            )
        )

        return transfer_op

    @staticmethod
    def create_rank_reduced_subview(target: ir.OpView) -> ir.OpView | None:
        # The rebuilt transfer carries no mask; leave masked transfers alone.
        if target.mask is not None:
            return None
        base_val = get_base(target)
        memref_type: ir.MemRefType = base_val.type
        if not memref_type.has_static_shape:
            return None
        strides, offset = memref_type.get_strides_and_offset()
        dynamic_val = ir.ShapedType.get_dynamic_stride_or_offset()
        if dynamic_val in strides:
            return None

        map_attr = get_permutation_map(target)
        map: ir.AffineMap = map_attr.value
        kept = []
        for expr in map.results:
            if not isinstance(expr, ir.AffineDimExpr):
                return None
            kept.append(expr.position)
        minor_identity = ir.AffineMap.get_minor_identity(map.n_dims, len(map.results))
        if map == minor_identity or kept != sorted(set(kept)):
            return None

        indices = list(get_indices(target))
        subview_offsets = []
        sizes = []
        for dim, (size, index) in enumerate(zip(memref_type.shape, indices)):
            if dim in kept:
                subview_offsets.append(0)
                sizes.append(size)
                continue
            sizes.append(1)
            if isinstance(index.owner, arith.ConstantOp):
                const_index = ir.IntegerAttr(index.owner.value).value
                subview_offsets.append(const_index)
                if offset != dynamic_val:
                    offset += const_index * strides[dim]
            else:
                subview_offsets.append(index)
                offset = dynamic_val

        layout = ir.StridedLayoutAttr.get(offset, [strides[dim] for dim in kept])
        result_type = ir.MemRefType.get(
            [memref_type.shape[dim] for dim in kept],
            memref_type.element_type,
            layout,
            memref_type.memory_space,
        )
        base_subview = memref.subview(
            base_val,
            subview_offsets,
            sizes,
            [1] * memref_type.rank,
            result_type=result_type,
        )

        identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(len(kept)))
        transfer_indices = [indices[dim] for dim in kept]
        if isinstance(target, vector.TransferReadOp):
            return vector.transfer_read(
                get_vector_type(target),
                base_subview,
                transfer_indices,
                identity,
                target.padding,
                get_in_bounds(target),
            ).owner
        return vector.transfer_write(
            None,
            target.valueToStore,
            base_subview,
            transfer_indices,
            identity,
            get_in_bounds(target),
        )

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "MoveOffsetsToSubviewOp",
            rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            targets = state.get_payload_ops(op.target)
            updated_ops = []
            create = (
                MoveOffsetsToSubviewOp.create_rank_reduced_subview
                if op.rank_reduce.value
                else MoveOffsetsToSubviewOp.create_subview
            )

            for target in targets:
                if not isinstance(
                    target, (vector.TransferReadOp, vector.TransferWriteOp)
                ):
                    return DiagnosedSilenceableFailure.SilenceableFailure

                base_val = get_base(target)
                if not isinstance(base_val.type, ir.MemRefType):
                    return DiagnosedSilenceableFailure.SilenceableFailure

                with ir.InsertionPoint(target), target.location:
                    updated_op = create(target)
                    if updated_op is None:
                        updated_op = target
                    else:
                        rewriter.replace_op(target, updated_op)
                updated_ops.append(updated_op)

            results.set_ops(op.updated_op, updated_ops)

            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "MoveOffsetsToSubviewOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: "MoveOffsetsToSubviewOp"):
            return (
                transform.consumes_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def move_offsets_to_subview(
    target: ir.Value[transform.AnyOpType],
    rank_reduce: bool = False,
) -> ir.Value[transform.AnyOpType]:
    """snake_case wrapper to create a MoveOffsetsToSubviewOp."""
    op = MoveOffsetsToSubviewOp(
        target=target,
        rank_reduce=ir.IntegerAttr.get(ir.IntegerType.get_signless(1), rank_reduce),
    )
    return op.updated_op
