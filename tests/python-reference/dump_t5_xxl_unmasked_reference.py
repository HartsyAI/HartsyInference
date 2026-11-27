"""Reference T5-XXL hidden states for the FLUX.1 prompt path, computed the way diffusers runs it.

diffusers' FluxPipeline calls ``text_encoder_2(text_input_ids)`` with the prompt padded to 512 tokens and no attention
mask, so every pad row is attended. This script does the same: it loads a standalone T5-v1.1-XXL encoder file (the
ComfyUI-style single file, F8_E4M3 or F16/BF16/F32), runs it in F32 on the CPU, and writes the last hidden state.
The engine's T5TextEncoder is checked against these dumps by tests/HartsyInference.Diffusion.Tests/Parity/
T5StandaloneParityTests.cs.

Output layout (the same per-prompt files as dump_t5_xxl_hidden_states.py):
  meta.json                  {"T5Repo", "MaxLength", "Prompts": [{"Index", "Prompt", "SeqLen", "HiddenDim"}]}
  prompt_NN_tokens.bin       I32 [MaxLength]   token ids, padded with 0
  prompt_NN_attention_mask.bin I32 [MaxLength] all ones: the unmasked (diffusers) semantics
  prompt_NN_hidden.bin       F32 [MaxLength * HiddenDim]

Usage: dump_t5_xxl_unmasked_reference.py --safetensors <t5xxl_enconly.safetensors> \
         --tokenizer-json <tokenizer.json> --output <dir> --prompt "remove" --prompt "person, people, ..."
"""
import argparse
import json
import os

import numpy as np
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers import T5Config, T5EncoderModel

# T5-v1.1-XXL, as FLUX.1 uses it (gated-GELU feed-forward, 64 heads of 64).
T5_XXL = dict(
    vocab_size=32128,
    d_model=4096,
    d_ff=10240,
    d_kv=64,
    num_heads=64,
    num_layers=24,
    relative_attention_num_buckets=32,
    relative_attention_max_distance=128,
    feed_forward_proj="gated-gelu",
    layer_norm_epsilon=1e-6,
    is_encoder_decoder=False,
    tie_word_embeddings=False,
)


def load_encoder(safetensors_path: str) -> T5EncoderModel:
    state = load_file(safetensors_path)
    state = {k: v.float() for k, v in state.items()}
    config = T5Config(**T5_XXL)
    model = T5EncoderModel(config).float().eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k != "encoder.embed_tokens.weight"]  # tied to shared.weight
    if missing or unexpected:
        raise SystemExit(f"weight mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safetensors", required=True)
    parser.add_argument("--tokenizer-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    tokenizer = Tokenizer.from_file(args.tokenizer_json)
    model = load_encoder(args.safetensors)

    prompts_meta = []
    for idx, prompt in enumerate(args.prompt):
        ids = tokenizer.encode(prompt).ids[: args.max_length]  # the tokenizer appends </s>
        ids = ids + [0] * (args.max_length - len(ids))
        input_ids = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            hidden = model(input_ids=input_ids).last_hidden_state.float()  # no attention mask: diffusers semantics
        hidden_np = hidden.squeeze(0).contiguous().numpy().astype(np.float32)
        np.asarray(ids, dtype=np.int32).tofile(os.path.join(args.output, f"prompt_{idx:02d}_tokens.bin"))
        np.ones(args.max_length, dtype=np.int32).tofile(os.path.join(args.output, f"prompt_{idx:02d}_attention_mask.bin"))
        hidden_np.tofile(os.path.join(args.output, f"prompt_{idx:02d}_hidden.bin"))
        prompts_meta.append({"Index": idx, "Prompt": prompt, "SeqLen": args.max_length, "HiddenDim": hidden_np.shape[-1]})
        print(f"prompt {idx}: {len(ids)} real tokens, hidden {tuple(hidden_np.shape)}", flush=True)

    with open(os.path.join(args.output, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"T5Repo": args.safetensors, "MaxLength": args.max_length, "Prompts": prompts_meta}, f, indent=2)
    print(f"wrote {len(prompts_meta)} prompts to {args.output}", flush=True)


if __name__ == "__main__":
    main()
