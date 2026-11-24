"""Import or execute a local Hugging Face causal LM prefill graph.

Examples:
    python examples/xegpu/hf_bench.py /path/to/Llama-3.2-1B \
        --output /tmp/llama-hf.mlir --batch-size 1 --sequence-length 16
    python examples/xegpu/hf_bench.py /path/to/Llama-3.2-1B \
        --dump-kernel cleanup --output /tmp/llama-cleanup.mlir
    python examples/xegpu/hf_bench.py /path/to/Llama-3.2-1B --execute
    python examples/xegpu/hf_bench.py /path/to/Llama-3.2-1B \
        --execute --benchmark --nwarmup 10 --nruns 30

The model and its pretrained weights stay on CPU for capture and import.
--dump-kernel stops after the selected stage and dumps MLIR without running
the model. The dump goes to stdout unless --output is provided. Using
--output alone dumps the imported linalg-on-tensors MLIR as before.
--execute attempts XeGPU lowering and transfers the captured tensors to XPU
in Dynamo order. --verify checks against eager CPU execution without an
explicit mask. --benchmark times resident-input prefill calls, including
output allocation, host launch work, and GPU completion, but not compilation
or CPU/XPU transfers. It does not perform generation, sampling, or KV caching.
The graph has fixed input shapes, no padding, and no KV cache. By default,
all positions are projected to logits, producing [batch, sequence, vocabulary].
Use --logits-to-keep N to project only the last N positions.
By default, capture receives a prebuilt 4D additive causal mask
shared across batches and heads. Use --no-attention-mask to capture the
Hugging Face model's default masking path instead.
"""

import argparse
from pathlib import Path
import sys
from time import perf_counter

import torch
from mlir import ir
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

from lighthouse.execution.runner import Runner
from lighthouse.ingress.torch import import_from_model
from lighthouse.ingress.torch.compile import MLIRBackend
from lighthouse.pipeline.driver import TransformDriver
from lighthouse.schedule.xegpu import get_xegpu_pipeline


class ImportComplete(Exception):
    pass


def time_prefill(jit_fn, device_args, *, nwarmup=10, nruns=30):
    """Time synchronous JIT calls with inputs already resident on XPU."""
    if nwarmup < 0 or nruns <= 0:
        raise ValueError("nwarmup must be nonnegative and nruns positive")
    with torch.no_grad():
        for _ in range(nwarmup):
            jit_fn(*device_args)
        torch.xpu.synchronize()
        times = []
        for _ in range(nruns):
            start = perf_counter()
            result = jit_fn(*device_args)
            torch.xpu.synchronize()
            times.append(perf_counter() - start)
            del result
    samples = torch.tensor(times, dtype=torch.float64) * 1000
    print(
        f"Resident-input prefill latency ({nwarmup} warmup, {nruns} runs): "
        f"mean {samples.mean().item():.3f} ms, "
        f"p50 {samples.quantile(0.5).item():.3f} ms, "
        f"p90 {samples.quantile(0.9).item():.3f} ms, "
        f"p99 {samples.quantile(0.99).item():.3f} ms",
        flush=True,
    )
    return times


class LogitDrift:
    """Accumulate RMS error and top-1 comparisons in bounded row chunks."""

    def __init__(self):
        self.squared_error = 0.0
        self.finite_elements = 0
        self.rows = 0
        self.finite_rows = 0
        self.flip_margins = []

    def update(self, actual, reference):
        finite = (torch.isfinite(actual) & torch.isfinite(reference)).all(dim=-1)
        self.rows += finite.numel()
        self.finite_rows += finite.sum().item()
        actual, reference = actual[finite], reference[finite]
        self.finite_elements += actual.numel()
        self.squared_error += (
            (actual.float() - reference.float())
            .square()
            .sum(dtype=torch.float64)
            .item()
        )
        flipped = actual.argmax(dim=-1) != reference.argmax(dim=-1)
        top2 = reference[flipped].topk(2, dim=-1).values.float()
        self.flip_margins.append(top2[:, 0] - top2[:, 1])

    def report(self):
        margins = torch.cat(self.flip_margins)
        if self.finite_rows:
            rms = (self.squared_error / self.finite_elements) ** 0.5
            print(f"Logit RMS error (finite rows): {rms:.6g}")
            print(
                f"Top-1 agreement: {self.finite_rows - margins.numel()}/"
                f"{self.finite_rows} finite rows"
            )
        else:
            print("Logit RMS error and top-1 agreement unavailable: no finite rows")
        if self.finite_rows != self.rows:
            print(
                f"Excluded {self.rows - self.finite_rows}/{self.rows} rows with non-finite logits"
            )
        if margins.numel():
            print(
                f"Eager top-two margin on {margins.numel()} flips: "
                f"{(margins == 0).sum().item()} exact ties; "
                f"min/median/max {margins.min().item():.6g}/"
                f"{margins.quantile(0.5).item():.6g}/{margins.max().item():.6g}"
            )


def load_prompt_ids(model_path, path, batch_size, sequence_length):
    """Tokenize one prompt per line, truncating long prompts without padding."""
    prompts = path.read_text(encoding="utf-8").splitlines()
    if len(prompts) != batch_size or any(not prompt.strip() for prompt in prompts):
        raise ValueError(
            f"{path}: expected exactly {batch_size} non-empty prompt lines"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    rows = tokenizer(prompts, truncation=True, max_length=sequence_length)["input_ids"]
    for i, row in enumerate(rows, start=1):
        if len(row) != sequence_length:
            raise ValueError(
                f"{path}:{i}: got {len(row)} tokens; need at least {sequence_length} "
                "before truncation (padding is not supported)"
            )
    return torch.tensor(rows, dtype=torch.long)


def lower_for_xpu(module: ir.Module, stop_at_stage: str | None = None) -> ir.Module:
    with module.context, ir.Location.unknown():
        Runner.make_function_callable(module, "main")
        schedules = get_xegpu_pipeline(device="B70", stop_at_stage=stop_at_stage)
        if not schedules:
            return module
        return TransformDriver(schedules=schedules).apply(module)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Local Hugging Face model directory")
    parser.add_argument("--output", type=Path, help="Write the MLIR dump to this file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dump-kernel",
        choices=[
            "imported",
            "initial",
            "cleanup",
            "tiled",
            "vectorized",
            "bufferized",
            "gpu-outlining",
            "xegpu-initial",
            "xegpu-wg",
            "final",
        ],
        help="Stop at this stage and dump MLIR (stdout unless --output is supplied)",
    )
    mode.add_argument("--execute", action="store_true", help="Lower and execute on XPU")
    parser.add_argument("--verify", action="store_true", help="Check against eager CPU")
    parser.add_argument(
        "--benchmark", action="store_true", help="Time repeated XPU prefills"
    )
    parser.add_argument("--nwarmup", type=int, default=10)
    parser.add_argument("--nruns", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument(
        "--logits-to-keep",
        type=int,
        default=0,
        help="Project only the last N positions to logits, use 0 for all positions (default: 0)",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="One prompt per batch row, each at least --sequence-length tokens",
    )
    parser.add_argument(
        "--report-logits",
        action="store_true",
        help="Compare logit RMS error and top-1 stability against eager CPU",
    )
    parser.add_argument(
        "--attention-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pass a prebuilt 4D additive causal mask (default: enabled)",
    )
    args = parser.parse_args()
    if not (args.output or args.dump_kernel or args.execute):
        parser.error("specify --output, --dump-kernel, or --execute")
    if args.execute and args.output:
        parser.error("--output cannot be used with --execute")
    if args.report_logits and not args.execute:
        parser.error("--report-logits requires --execute")
    if (args.verify or args.benchmark) and not args.execute:
        parser.error("--verify and --benchmark require --execute")
    if args.nwarmup < 0 or args.nruns <= 0:
        parser.error("--nwarmup must be nonnegative and --nruns positive")
    if args.batch_size <= 0 or args.sequence_length <= 0:
        parser.error("batch size and sequence length must be positive")
    if not 0 <= args.logits_to_keep <= args.sequence_length:
        parser.error("--logits-to-keep must be between 0 and --sequence-length")
    if args.execute and not torch.xpu.is_available():
        parser.error("--execute requires a PyTorch XPU device")

    print("Loading Hugging Face model", file=sys.stderr, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.cpu().eval()
    model.config.use_cache = False
    if args.prompt_file is None:
        inputs = torch.ones((args.batch_size, args.sequence_length), dtype=torch.long)
    else:
        inputs = load_prompt_ids(
            args.model, args.prompt_file, args.batch_size, args.sequence_length
        )
        if (
            inputs.min().item() < 0
            or inputs.max().item() >= model.get_input_embeddings().weight.shape[0]
        ):
            raise ValueError("Prompt contains token IDs outside the embedding table")
    print(
        f"Prefill: {len(model.model.layers)} layers, {args.batch_size} x {args.sequence_length} tokens",
        file=sys.stderr,
        flush=True,
    )
    attention_mask = None
    if args.attention_mask:
        attention_mask = torch.full(
            (1, 1, args.sequence_length, args.sequence_length),
            float("-inf"),
            dtype=torch.bfloat16,
        ).triu(diagonal=1)
    model_inputs = dict(
        input_ids=inputs,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=False,
        logits_to_keep=args.logits_to_keep,
    )

    def import_backend(graph, example_inputs):
        stage = args.dump_kernel or "imported"
        print("Importing full graph to linalg-on-tensors", file=sys.stderr, flush=True)
        if stage == "imported":
            mlir = import_from_model(graph, sample_args=example_inputs)
        else:
            backend = MLIRBackend(torch.device("cpu"), lower_for_xpu)
            module = backend.get_mlir(graph, example_inputs)
            entry = backend.get_entry_func(module)
            if entry is None:
                raise ValueError(f"Failed to find MLIR entry: {backend.entry_func}")
            backend.preprocess_func(entry)
            mlir = lower_for_xpu(module, stop_at_stage=stage)
        if args.output is None:
            sys.stdout.write(str(mlir))
        else:
            with args.output.open("x") as output:
                output.write(str(mlir))
            print(f"Wrote {args.output}", file=sys.stderr, flush=True)

        def stop_after_import(*_args):
            raise ImportComplete()

        return stop_after_import

    if args.execute:
        device = torch.device("xpu")
        xpu_backend = MLIRBackend(
            device,
            lower_for_xpu,
            shared_libs=["libmlir_levelzero_runtime.so"],
        )

        captured_device_args = None

        def execute_backend(graph, example_inputs):
            xpu_backend(graph, example_inputs)
            jit_fn = xpu_backend.jit_function

            def execute_on_xpu(*captured_args):
                nonlocal captured_device_args
                # The runtime consumes device pointers; Dynamo supplies CPU tensors.
                device_args = [tensor.to(device) for tensor in captured_args]
                captured_device_args = device_args
                return [result.cpu() for result in jit_fn(*device_args)]

            return execute_on_xpu

        compiled = torch.compile(
            model, backend=execute_backend, fullgraph=True, dynamic=False
        )
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            start = perf_counter()
            result = compiled(**model_inputs)
            print(
                f"Compiled first call: {perf_counter() - start:.1f} s",
                file=sys.stderr,
                flush=True,
            )
            if args.verify or args.report_logits:
                start = perf_counter()
                reference = model(**(model_inputs | {"attention_mask": None}))
                print(
                    f"Eager CPU reference: {perf_counter() - start:.1f} s",
                    file=sys.stderr,
                    flush=True,
                )
        if args.benchmark:
            if captured_device_args is None:
                raise RuntimeError("No captured XPU arguments available for timing")
            time_prefill(
                xpu_backend.jit_function,
                captured_device_args,
                nwarmup=args.nwarmup,
                nruns=args.nruns,
            )
        if args.report_logits:
            actual_logits, reference_logits = result[0], reference[0]
            if actual_logits.shape != reference_logits.shape:
                raise ValueError("Compiled and eager logits have different shapes")
            drift = LogitDrift()
            for actual, expected in zip(
                actual_logits.flatten(0, 1).split(64),
                reference_logits.flatten(0, 1).split(64),
                strict=True,
            ):
                drift.update(actual, expected)
            drift.report()
        if args.verify:
            torch.testing.assert_close(result, reference, rtol=2e-2, atol=2e-2)
            print("Verification PASSED (eager CPU reference)", flush=True)
        return

    compiled = torch.compile(
        model, backend=import_backend, fullgraph=True, dynamic=False
    )
    try:
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            compiled(**model_inputs)
    except ImportComplete:
        return
    raise RuntimeError("The compiled model returned without importing its graph")


if __name__ == "__main__":
    main()
