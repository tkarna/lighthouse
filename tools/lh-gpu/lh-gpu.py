#! /usr/bin/env python

import argparse

from lighthouse import dialects as lh_dialects
from lighthouse.pipeline.helper import import_mlir_module
from lighthouse.schedule.xegpu.mlp_schedule import get_schedule_module
from lighthouse.workload.runner import get_engine
from lighthouse.utils.memref import to_packed_args

import numpy as np
from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from mlir.dialects import func, linalg

from mlir.runtime.np_to_memref import get_ranked_memref_descriptor


from lighthouse.dialects import transform_ext
from lighthouse.schedule import schedule_boilerplate
from memory_manager import MemoryManager, GPUMemoryManager


def mlir_to_numpy_type(mlir_type):
    if isinstance(mlir_type, ir.F32Type):
        return np.float32
    if isinstance(mlir_type, ir.F16Type):
        return np.float16
    raise ValueError(f"Unsupported MLIR type: {mlir_type}")


def get_bench_wrapper_schedule(
    payload_function_name: str, benchmark_function_name: str
):
    with schedule_boilerplate() as (schedule, named_seq):
        named_func = structured.structured_match(
            transform.AnyOpType.get(),
            target=named_seq.bodyTarget,
            ops={"func.func"},
            op_attrs={"sym_name": ir.StringAttr.get(payload_function_name)},
        )
        bench_func = transform_ext.wrap_in_benching_func(
            named_func, bench_name=benchmark_function_name
        )
        transform.yield_([bench_func])

    schedule.body.operations[0].verify()
    return schedule


def inspect_payload(payload_module: ir.Module) -> dict:
    """Inspect the payload module and extract metadata about the functions/ops it contains."""

    functions = {}

    def match_funcs(op: ir.Operation) -> ir.WalkResult:
        op = op.opview
        match op:
            case func.FuncOp():
                matmuls = []

                def match_linalg(op: ir.Operation) -> ir.WalkResult:
                    op = op.opview
                    match op:
                        case linalg.MatmulOp():
                            inputs = op.inputs
                            outputs = op.outputs
                            assert len(inputs) == 2 and len(outputs) == 1
                            m, k = inputs[0].type.shape
                            _, n = inputs[1].type.shape
                            matmuls.append((m, n, k))
                    return ir.WalkResult.ADVANCE

                op.walk(match_linalg, ir.WalkOrder.PRE_ORDER)
                functions[op.sym_name.value] = {
                    "inputs": op.type.inputs,
                    "results": op.type.results,
                    "matmuls": matmuls,
                }
        return ir.WalkResult.ADVANCE

    op = payload_module.body.operations[0]
    op.walk(match_funcs, ir.WalkOrder.PRE_ORDER)
    return functions


def execute(
    payload_module: ir.Module,
    payload_function_name: str,
    schedule_modules: list[ir.Module],
    host_np_inputs: list[np.ndarray],
    MemManager: type[MemoryManager],
    arg_metadata: list[tuple[tuple[int, ...], type]],
    shared_libs: list[str],
    benchmark: bool = False,
):
    # Emit utility functions for memory manager (if any)
    MemManager.emit_memory_management_funcs(payload_module, arg_metadata)

    if benchmark:
        bench_func_name = payload_func_name + "_benchmark"
        schedule_modules = [
            get_bench_wrapper_schedule(
                payload_function_name=payload_function_name,
                benchmark_function_name=bench_func_name,
            )
        ] + schedule_modules

    # Lower payload
    for schedule_module in schedule_modules:
        schedule_module.body.operations[0].apply(payload_module)

    # Create execution engine
    execution_engine = get_engine(payload_module, shared_libs=shared_libs)

    # Allocate device arrays
    mem_manager = MemManager(execution_engine)
    with mem_manager.allocate_buffers(arg_metadata) as memrefs:
        # Copy host arrays to device
        for host_np_arr, mref in zip(host_np_inputs, memrefs):
            host_mref = get_ranked_memref_descriptor(host_np_arr)
            mem_manager.copy_to_device(host_mref, mref)

        if benchmark:
            # Allocate buffer for timings
            time_array = np.zeros((nruns,), dtype=np.float64)
            time_memref = get_ranked_memref_descriptor(time_array)

            # Call benchmark function
            all_args = memrefs + [time_memref, nruns, nwarmup]
            packed_args_with_time = to_packed_args(all_args)
            benchmark_func = execution_engine.lookup(bench_func_name)
            benchmark_func(packed_args_with_time)

            # Calculate timings
            time_array *= 1e6  # convert to microseconds
            mean = np.mean(time_array)
            min = np.min(time_array)
            max = np.max(time_array)
            std = np.std(time_array)
            print(f"Running benchmark for function '{payload_function_name}'...")
            print(
                f"Timings (us): mean = {mean:.2f} +/-{std:.2f} min={min:.2f} max={max:.2f}"
            )
        else:
            # Call payload function once
            print(f"Running payload function '{payload_function_name}'...")
            packed_args = to_packed_args(memrefs)
            payload_func = execution_engine.lookup(payload_function_name)
            payload_func(packed_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Lighthouse Intel GPU compiler and execution engine."""
    )
    parser.add_argument("payload_module", type=str, help="Path to payload MLIR file.")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Whether to time the kernel. If not set, will just run the payload once.",
    )
    args = parser.parse_args()

    # Other cli args
    nruns = 500
    nwarmup = 200

    # Constants
    benchmark_func_name = "benchmark"

    with ir.Context() as ctx, ir.Location.unknown():
        lh_dialects.register_and_load()

        payload_module = import_mlir_module(args.payload_module, ctx)

        # Extract function argument shapes and types and matmul shapes
        # TODO: Inspect the payload and determine if it is suitable for known
        # lowering pipelines, e.g. MLP-like func, no allocs, no return values
        function_metadata = inspect_payload(payload_module)
        assert len(function_metadata) == 1, (
            "Expected exactly one function in the payload module."
        )
        payload_func_name = function_metadata.keys().__iter__().__next__()
        function_metadata = function_metadata[payload_func_name]
        assert len(function_metadata["results"]) == 0, (
            "Expected payload function to have no return values."
        )

        has_bias = False
        has_relu = False
        has_convert_c = False
        arg_metadata = [(i.shape, i.element_type) for i in function_metadata["inputs"]]

        # TODO figure out what shared libs are needed
        shared_libs = [
            "libmlir_levelzero_runtime.so",  # for xegpu target
            "libmlir_c_runner_utils.so",  # in case we are benchmarking
        ]

        # Allocate and initialize host arrays with numpy
        # TODO support different initialization schemes

        def gen_random(shape: tuple[int, ...], dtype: type):
            # generate values in range [-3, 3]
            a = np.random.randint(-3, 4, shape)
            return a.astype(dtype)

        host_np_inputs = [
            gen_random(shape, mlir_to_numpy_type(dtype))
            for shape, dtype in arg_metadata
        ]

        # TODO hook up with parameter selector, it should live in lighthouse
        # TODO parameter selector should have sane default case for non-optimized shapes (?)
        matmuls = function_metadata["matmuls"]
        assert len(matmuls) == 1, (
            "Expected exactly one matmul op in the payload function."
        )
        M, N, K = matmuls[0]
        matmul_parameters = {
            "m": M,
            "n": N,
            "k": K,
            "wg_m": 256,
            "wg_n": 256,
            "sg_m": 32,
            "sg_n": 32,
            "k_tile": 64,
            "load_a_m": 32,
            "load_a_k": 16,
            "load_b_k": 32,
            "load_b_n": 16,
            "prefetch_a_m": 8,
            "prefetch_a_k": 32,
            "prefetch_b_k": 8,
            "prefetch_b_n": 32,
            "prefetch_nb": 1,
        }

        # Get schedules
        # TODO get schedules from CLI?
        # NOTE at the moment we only have 1 schedule, and it cannot be broken apart.
        schedule_modules = [
            get_schedule_module(
                has_bias=has_bias,
                has_relu=has_relu,
                has_convert_c=has_convert_c,
                params=[matmul_parameters],
            ),
        ]

        # Run benchmark
        execute(
            payload_module,
            payload_func_name,
            schedule_modules,
            host_np_inputs,
            GPUMemoryManager,
            arg_metadata,
            shared_libs,
            benchmark=args.benchmark,
        )
