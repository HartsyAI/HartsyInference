#!/usr/bin/env python3
"""Small random Muse Spark checkpoint for GGUF parity against llama.cpp.

Writes a MuseSparkForConditionalGeneration with a quant-friendly text config (every projection row a multiple of 256,
so each GGUF quant type applies to every weight instead of falling back) and the real 202,048-token Muse tokenizer,
then the float32 transformers logits for a fixed prompt. Convert it with llama.cpp's own converter and quantizer
(PR ggml-org/llama.cpp#30616 adds the muse-spark architecture):

    python -I tests/python-reference/make_muse_spark_small_gguf.py --tokenizer-dir <dir with tokenizer.json> --out <dir>
    python convert_hf_to_gguf.py <out>/hf --outtype f32 --outfile <out>/muse-spark-small-F32.gguf
    llama-quantize --imatrix <imatrix> <out>/muse-spark-small-F32.gguf <out>/muse-spark-small-Q4_K_M.gguf Q4_K_M
    llama-debug -m <gguf> -p "<prompt.json prompt>" --save-logits --logits-output-dir <out>/llamacpp -fa off -ctk f32 -ctv f32

llama.cpp's default flash attention keeps K/V in F16; the F32 cache makes its F32 run match transformers to 1e-4. Its quantized
matmuls still round activations to Q8, which flips near-tie expert choices of a random router, so --stable-routing builds the
copy used for the quantized comparison.

The tokenizer files come from any Muse Spark repo (trl-internal-testing/tiny-MuseSparkForConditionalGeneration is
ungated). MuseSparkGgufParityTests reads <out> through MUSE_SPARK_SMALL_DIR.
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file
from transformers import (AutoTokenizer, MuseSparkAudioConfig, MuseSparkConfig, MuseSparkForConditionalGeneration,
                          MuseSparkTextConfig, MuseSparkVisionConfig)

PROMPT = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")
OFFSET_NORMS = ("input_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
                "pre_expert_norm.weight", "post_expert_norm.weight")


def build_model(seed: int, stable_routing: bool) -> MuseSparkForConditionalGeneration:
    text = MuseSparkTextConfig(
        hidden_size=256, intermediate_size=512, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, sliding_window=16, num_local_experts=16, num_experts_per_tok=4, moe_hidden_size=256,
        moe_intermediate_size=256, pad_token_id=200018)
    vision = MuseSparkVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2)
    audio = MuseSparkAudioConfig(hidden_sizes=[8, 8, 8, 8], depths=[1, 1, 1, 1], out_hidden_size=text.hidden_size)
    config = MuseSparkConfig(text_config=text.to_dict(), vision_config=vision.to_dict(), audio_config=audio.to_dict(),
                             out_hidden_size=32, projector_hidden_size=16)
    torch.manual_seed(seed)
    model = MuseSparkForConditionalGeneration(config).float().eval()
    gen = torch.Generator().manual_seed(seed + 1)

    def draw(shape, std, mean=0.0):
        return mean + std * torch.randn(shape, generator=gen)

    with torch.no_grad():
        for name, p in model.named_parameters():
            if not (name.startswith("model.language_model.") or name == "lm_head.weight"):
                continue
            if name.endswith(OFFSET_NORMS):
                p.copy_(draw(p.shape, 0.1))
            elif name.endswith("language_model.norm.weight"):
                p.copy_(draw(p.shape, 0.1, 1.0))
            elif name.endswith("_residual_gate.gate"):
                p.copy_(draw(p.shape, 0.8))
            elif name.endswith("mlp.gate.weight"):
                p.copy_(draw(p.shape, 0.15))
            elif name.endswith("embed_tokens.weight"):
                p.copy_(draw(p.shape, 1.0))
            elif name == "lm_head.weight":
                p.copy_(draw(p.shape, 0.7))
            elif ".experts." in name:
                p.copy_(draw(p.shape, 0.06))
            else:
                p.copy_(draw(p.shape, 0.05))
        for name, b in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                if stable_routing:
                    # Ranks two apart outweigh any sigmoid score, so the selection cannot flip under activation rounding.
                    b.copy_(2.0 * torch.randperm(b.numel(), generator=gen).to(b.dtype))
                else:
                    b.copy_(draw(b.shape, 0.05))
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stable-routing", action="store_true",
                        help="fix each layer's expert selection with a large selection-only bias, so comparing against an engine that "
                             "rounds activations (llama.cpp's quantized matmuls) measures the numerics, not a flipped near-tie")
    args = parser.parse_args()

    hf_dir = os.path.join(args.out, "hf")
    os.makedirs(hf_dir, exist_ok=True)
    model = build_model(args.seed, args.stable_routing)
    model.save_pretrained(hf_dir, safe_serialization=True)
    for name in TOKENIZER_FILES:
        src = os.path.join(args.tokenizer_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(hf_dir, name))

    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    ids = tokenizer(PROMPT, return_tensors="pt", add_special_tokens=True).input_ids
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[0].float().contiguous()
    save_file({"input_ids": ids[0].to(torch.int32).contiguous(), "logits": logits},
              os.path.join(args.out, "transformers_reference.safetensors"))
    with open(os.path.join(args.out, "prompt.json"), "w", encoding="utf-8") as f:
        json.dump({"prompt": PROMPT, "input_ids": ids[0].tolist()}, f)
    print(f"{len(ids[0])} prompt tokens, logits {tuple(logits.shape)}, top-1 {int(logits[-1].argmax())}")


if __name__ == "__main__":
    main()
