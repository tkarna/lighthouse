# RUN: %PYTHON %s
# REQUIRES: torch, torch_mlir, transformers

import sys
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "xegpu"))

import hf_bench as driver


model = (
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    )
    .to(dtype=torch.bfloat16)
    .eval()
)
inputs = torch.ones((2, 4), dtype=torch.long)
with torch.no_grad():
    all_logits = model(inputs, use_cache=False).logits


for options, positions in (
    ([], 4),
    (["--logits-to-keep", "1"], 1),
    (["--logits-to-keep", "0"], 4),
):
    projected_shapes = []
    hook = model.lm_head.register_forward_pre_hook(
        lambda module, args: projected_shapes.append(tuple(args[0].shape))
    )

    def compiled(**kwargs):
        assert kwargs["logits_to_keep"] == (0 if positions == 4 else 1)
        result = model(**kwargs)
        assert result[0].shape == (2, positions, 32)
        torch.testing.assert_close(result[0], all_logits[:, -positions:, :])
        return result

    with (
        patch.object(
            sys,
            "argv",
            [
                "driver",
                "/unused",
                "--execute",
                "--verify",
                "--batch-size",
                "2",
                "--sequence-length",
                "4",
                *options,
            ],
        ),
        patch.object(
            driver.AutoModelForCausalLM, "from_pretrained", return_value=model
        ),
        patch.object(driver.torch.xpu, "is_available", return_value=True),
        patch.object(driver, "MLIRBackend"),
        patch.object(driver.torch, "compile", return_value=compiled),
    ):
        driver.main()
    hook.remove()
    assert projected_shapes == [(2, positions, 16)] * 2
