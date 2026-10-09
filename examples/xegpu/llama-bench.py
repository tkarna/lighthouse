"""Lower and execute llama3 inspired kernels.

Imports the GQA + causal SDPA case (llama-bench_5_attention.py) to bf16
linalg-on-tensors IR and applies the parameter-free XeGPU lowering pipeline --
the same stages as the test driver -- stopping at --dump-kernel.

    python llama-bench.py --dump-kernel initial
"""

import argparse
from pathlib import Path

import torch
import torch._dynamo as dynamo
from mlir import ir

from lighthouse import dialects as lh_dialects
from lighthouse.ingress.torch import gpu_backend, import_model, TargetDialect
from lighthouse.pipeline.helper import PipelineInterrupt
from lighthouse.pipeline.driver import TransformDriver
from lighthouse.schedule import xegpu
from lighthouse.execution.runner import Runner

ATTENTION_CASE = Path(__file__).with_name("llama-bench_5_attention.py")

STAGES = [
    "initial",
    "cleanup",
    "tiled",
    "vectorized",
    "bufferized",
    "gpu-outlining",
    "xegpu-initial",
    "xegpu-wg",
    "final",
]


def build_pipeline(stop_at_stage: str) -> list[ir.Module]:
    """Parameter-free XeGPU lowering pipeline, truncated at `stop_at_stage`."""
    schedules = []
    if stop_at_stage == "initial":
        return schedules
    schedules.append(xegpu.cleanup_schedule())
    if stop_at_stage == "cleanup":
        return schedules
    schedules.append(xegpu.wg_tiling_schedule())
    if stop_at_stage == "tiled":
        return schedules
    schedules.append(xegpu.vectorize_schedule())
    if stop_at_stage == "vectorized":
        return schedules
    schedules.append(xegpu.bufferize_schedule())
    if stop_at_stage == "bufferized":
        return schedules
    schedules.append(xegpu.outline_gpu_func_schedule())
    if stop_at_stage == "gpu-outlining":
        return schedules
    schedules.append(xegpu.vector_to_xegpu_schedule())
    if stop_at_stage == "xegpu-initial":
        return schedules
    schedules.append(xegpu.annotate_layouts_schedule())
    if stop_at_stage == "xegpu-wg":
        return schedules
    schedules.append(xegpu.xegpu_to_binary())
    return schedules


def is_caused_by_pipeline_interrupt(exc: BaseException) -> bool:
    pending = [exc]
    visited = set()
    while pending:
        current = pending.pop()
        if current is None or current in visited:
            continue
        visited.add(current)
        if isinstance(current, PipelineInterrupt):
            return True
        pending.extend(
            [getattr(current, "__cause__", None), getattr(current, "__context__", None)]
        )
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump-kernel",
        choices=STAGES,
        default="initial",
        help="Stop the pipeline at this stage and print the IR.",
    )
    args = parser.parse_args()

    benchmark = True
    payload_func_name = "main"

    with ir.Context() as ctx, ir.Location.unknown():
        lh_dialects.register_and_load()

        model, inputs, _kwargs = import_model(
            ATTENTION_CASE, model_datatype=torch.bfloat16
        )
        inputs = [t.to(torch.bfloat16) for t in inputs]

        def compile_model(mod: ir.Module) -> ir.Module:
            Runner.make_function_callable(mod, payload_func_name)
            schedules = build_pipeline(args.dump_kernel)
            if benchmark:
                wrapper = Runner.get_bench_wrapper_schedule(payload_func_name)
                schedules = [wrapper] + schedules

            lowered_mod = (
                TransformDriver(schedules=schedules).apply(mod) if schedules else mod
            )
            print(lowered_mod)
            raise PipelineInterrupt()

        backend = gpu_backend(
            compile_model,
            device=torch.device("xpu"),
            dialect=TargetDialect.LINALG_ON_TENSORS,
            ir_context=ctx,
        )
        model.compile(dynamic=False, backend=backend)
        try:
            with torch.no_grad():
                model(*inputs)
        except dynamo.exc.BackendCompilerFailed as exc:
            if not is_caused_by_pipeline_interrupt(exc):
                raise


if __name__ == "__main__":
    main()
