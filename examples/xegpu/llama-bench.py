"""Lower and execute llama3 inspired kernels.

Imports the GQA + causal SDPA case (llama-bench_5_attention.py) to bf16
linalg-on-tensors IR and applies the parameter-free XeGPU lowering pipeline --
the same stages as the test driver -- stopping at --dump-kernel.

    python llama-bench.py --dump-kernel initial
"""

import argparse
from pathlib import Path

import torch
from mlir import ir

from lighthouse import dialects as lh_dialects
from lighthouse.ingress.torch import import_model, import_from_model
from lighthouse.pipeline.driver import TransformDriver
from lighthouse.schedule import xegpu
from lighthouse.schedule.func import convert_function_results

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
    # Move the tensor result into an output arg, matching the kernel-bench
    # "initial" form (the ingress backend does this via move_results_to_args).
    schedules = [convert_function_results()]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump-kernel",
        choices=STAGES,
        default="initial",
        help="Stop the pipeline at this stage and print the IR.",
    )
    args = parser.parse_args()

    with ir.Context() as ctx, ir.Location.unknown():
        lh_dialects.register_and_load()

        model, inputs, kwargs = import_model(
            ATTENTION_CASE, model_datatype=torch.bfloat16
        )
        inputs = [t.to(torch.bfloat16) for t in inputs]
        mod = import_from_model(model, inputs, kwargs, ir_context=ctx)

        TransformDriver(schedules=build_pipeline(args.dump_kernel)).apply(mod)
        print(mod)


if __name__ == "__main__":
    main()
