"""Rewrite checked embedding tiles as contiguous vector copies."""

from mlir import ir
from mlir.dialects import affine, arith, cf, ext, linalg, tensor, transform, vector
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.utils.mlir import indexing_maps, iterator_types


def match_checked_embedding(op: ir.Operation, *, tiled: bool = False):
    """Match a checked row lookup before or after conventional tiling."""
    op = op.opview
    if not isinstance(op, linalg.GenericOp):
        return None
    if len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1:
        return
    ids, out = op.inputs[0], op.outputs[0]
    if not all(
        isinstance(v.type, ir.RankedTensorType)
        and v.type.has_static_shape
        and v.type.encoding is None
        for v in (ids, out)
    ):
        return
    rank = out.type.rank
    if rank not in (2, 3) or out.type.shape[:-1] != ids.type.shape:
        return
    if not all(size > 0 for size in out.type.shape):
        return
    if iterator_types(op) != ["parallel"] * rank or indexing_maps(op) != [
        ir.AffineMap.get(rank, 0, [ir.AffineDimExpr.get(i) for i in range(rank - 1)]),
        ir.AffineMap.get_identity(rank),
    ]:
        return
    body = op.regions[0].blocks[0]
    if len(body.operations) < 2:
        return
    *prefix, load, yield_op = list(body.operations)
    if (
        not isinstance(load, tensor.ExtractOp)
        or not isinstance(yield_op, linalg.YieldOp)
        or list(yield_op.operands) != [load.result]
        or len(load.indices) != 2
    ):
        return
    table = load.tensor
    width = out.type.shape[-1]
    if (
        not isinstance(table.type, ir.RankedTensorType)
        or table.type.encoding is not None
        or not table.type.has_static_shape
        or table.type.rank != 2
        or table.type.shape[1] < width
        or not isinstance(out.type.element_type, ir.BF16Type)
    ):
        return
    row, col = load.indices
    column_ops = []
    start = None
    if tiled:
        if any(size != 1 for size in ids.type.shape):
            return
        offset = col.owner
        if isinstance(offset, affine.AffineApplyOp) and offset in prefix:
            d, s = ir.AffineDimExpr.get(0), ir.AffineSymbolExpr.get(0)
            if offset.map.value != ir.AffineMap.get(1, 1, [d + s]):
                return
            start, col = offset.operands
            column_ops.append(offset)
        elif table.type.shape[1] != width:
            return
    elif table.type.shape[1] != width:
        return
    row_op, col_op = row.owner, col.owner
    if (
        not isinstance(row_op, arith.IndexCastOp)
        or row_op not in prefix
        or row_op.operands[0] != body.arguments[0]
        or not isinstance(col_op, linalg.IndexOp)
        or col_op not in prefix
        or int(col_op.dim) != rank - 1
        or sum(isinstance(p, cf.AssertOp) for p in prefix) != 2
    ):
        return
    local_values = set(body.arguments) | {v for p in prefix for v in p.results}
    available = {body.arguments[0], row}
    for p in prefix:
        if p in (row_op, col_op) or p in column_ops:
            continue
        if not isinstance(p, (arith.CmpIOp, cf.AssertOp)) or any(
            v in local_values and v not in available for v in p.operands
        ):
            return
        available.update(p.results)
    if start in local_values:
        return
    return load, start, [p for p in prefix if p != col_op and p not in column_ops]


def _rewrite_lookup(
    op: linalg.GenericOp, lookup, rewriter: transform.TransformRewriter
):
    load, start, prefix = lookup
    ids, out = op.inputs[0], op.outputs[0]
    rank = out.type.rank
    body = op.regions[0].blocks[0]
    with op.location, ir.InsertionPoint(op):
        zero = arith.constant(ir.IndexType.get(), 0)
        values = {body.arguments[0]: tensor.extract(ids, [zero] * (rank - 1))}
        for p in prefix:
            cloned = p.operation.clone()
            for i, operand in enumerate(p.operands):
                if operand in values:
                    cloned.operands[i] = values[operand]
            values.update(zip(p.results, cloned.results))
        elem = out.type.element_type
        in_bounds = ir.ArrayAttr.get([ir.BoolAttr.get(True)])
        tile = vector.transfer_read(
            ir.VectorType.get([out.type.shape[-1]], elem),
            load.tensor,
            [values[load.indices[0]], zero if start is None else start],
            ir.AffineMapAttr.get(ir.AffineMap.get_minor_identity(2, 1)),
            arith.constant(elem, ir.FloatAttr.get(elem, 0.0)),
            in_bounds,
        )
        write = vector.TransferWriteOp(
            out.type,
            tile,
            out,
            [zero] * rank,
            ir.AffineMapAttr.get(ir.AffineMap.get_minor_identity(rank, 1)),
            in_bounds,
        )
    rewriter.replace_op(op, write)
    return write


class RewriteCheckedEmbeddingOp(
    TransformExtensionDialect.Operation, name="rewrite_checked_embedding"
):
    """Replace single-token embedding tiles with checked vector row copies.

    Use the tile's column offset and destination. Run both token-ID checks
    before the vector load. The enclosing workgroup loop and slice insertion
    are unchanged.
    """

    target: ext.Operand[transform.AnyOpType]
    result: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(op, rewriter, results, state) -> DiagnosedSilenceableFailure:
            lookups = []
            for target in state.get_payload_ops(op.target):
                generic = target.opview
                lookup = match_checked_embedding(target, tiled=True)
                if not isinstance(generic, linalg.GenericOp) or lookup is None:
                    target.location.emit_error(
                        "expected a statically shaped single-token checked embedding tile"
                    )
                    return DiagnosedSilenceableFailure.DefiniteFailure
                lookups.append((generic, lookup))
            writes = [
                _rewrite_lookup(generic, lookup, rewriter)
                for generic, lookup in lookups
            ]
            results.set_ops(op.result, writes)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op):
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op):
            return (
                transform.consumes_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def rewrite_checked_embedding(target: ir.Value[transform.AnyOpType]) -> ir.Value:
    return RewriteCheckedEmbeddingOp(target=target).result
