import ctypes
from contextlib import contextmanager
from dataclasses import dataclass, field
import abc
import numpy as np
from typing import Any

from lighthouse.ingress.mlir_gen.gpu_utils import emit_gpu_util_funcs
from lighthouse.utils.memref import to_ctype as memref_to_ctype
from lighthouse.utils.numpy import numpy_to_mlir_type

from mlir import ir
from mlir.execution_engine import ExecutionEngine
from mlir.runtime.np_to_memref import (
    make_nd_memref_descriptor,
    as_ctype,
)


def mlir_to_type_str(mlir_type):
    if isinstance(mlir_type, ir.F32Type):
        return "f32"
    if isinstance(mlir_type, ir.F16Type):
        return "f16"
    raise ValueError(f"Unsupported MLIR type: {mlir_type}")


class MemoryManager(abc.ABC):
    """Abstract base class for memory management."""

    @abc.abstractmethod
    def copy_to_device(self, host_buffer: Any, device_buffer: Any):
        """Copy data from a host buffer to a device buffer."""
        pass

    @abc.abstractmethod
    def allocate_buffers(self, shapes_and_types: list[tuple[tuple[int, ...], type]]):
        """Context manager for allocating and freeing device buffers."""
        pass

    @staticmethod
    def emit_memory_management_funcs(
        payload_module: ir.Module, host_inputs: list[np.ndarray]
    ):
        """Emit utility functions required by this memory manager into the payload module."""
        pass


@dataclass
class GPUMemoryManager(MemoryManager):
    """GPU memory manager that uses MLIR gpu.alloc/dealloc/memcpy ops."""

    execution_engine: ExecutionEngine
    allocated_buffers: list[tuple[str, ctypes.Structure]] = field(default_factory=list)

    def _alloc(self, shape: tuple[int, ...], elem_type: type) -> ctypes.Structure:
        type_str = mlir_to_type_str(elem_type)
        np_dtype = {
            "f16": np.float16,
            "f32": np.float32,
        }[type_str]
        mref = make_nd_memref_descriptor(len(shape), as_ctype(np_dtype))()
        ptr_mref = memref_to_ctype(mref)
        ptr_dims = [ctypes.pointer(ctypes.c_int32(d)) for d in shape]
        rank = len(shape)
        assert rank in (1, 2), "Only 1d or 2d arrays are supported."
        suffix = f"{rank}d_{type_str}"
        self.execution_engine.invoke("gpu_alloc_" + suffix, ptr_mref, *ptr_dims)

        # NOTE need to track datatype as MemRefDescriptor does not include element type
        self.allocated_buffers.append((type_str, mref))
        return mref

    def _free_all(self):
        for type_str, mref in self.allocated_buffers:
            ptr_mref = memref_to_ctype(mref)
            rank = len(mref.shape)
            suffix = f"{rank}d_{type_str}"
            self.execution_engine.invoke("gpu_dealloc_" + suffix, ptr_mref)
        self.allocated_buffers.clear()

    def copy_to_device(self, host_mref: ctypes.Structure, gpu_mref: ctypes.Structure):
        rank = len(gpu_mref.shape)
        type_str = None
        for t_str, mref in self.allocated_buffers:
            if mref == gpu_mref:
                type_str = t_str
                break
        assert type_str is not None, (
            "GPU memory reference not found in allocated buffers."
        )
        copy_func_name = f"gpu_copy_{rank}d_{type_str}"
        self.execution_engine.invoke(
            copy_func_name, memref_to_ctype(host_mref), memref_to_ctype(gpu_mref)
        )

    @contextmanager
    def allocate_buffers(self, host_inputs: list[np.ndarray]):
        buffers = []
        try:
            for arr in host_inputs:
                buf = self._alloc(arr.shape, numpy_to_mlir_type(arr.dtype))
                buffers.append(buf)
            yield buffers
        finally:
            self._free_all()

    @staticmethod
    def emit_memory_management_funcs(
        payload_module: ir.Module, host_inputs: list[np.ndarray]
    ):
        """Emit utility functions required by this class into the payload module."""
        arg_kinds = set(
            (numpy_to_mlir_type(arr.dtype), arr.ndim) for arr in host_inputs
        )
        with ir.InsertionPoint(payload_module.body):
            for elem_type, rank in arg_kinds:
                emit_gpu_util_funcs(elem_type, rank)
