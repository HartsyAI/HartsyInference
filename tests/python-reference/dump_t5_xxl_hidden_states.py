"""
Encodes a fixed set of prompts through HuggingFace T5-XXL and dumps the encoder
output hidden states as F32 binaries. The C# `T5EncoderDiffTests` load these and
compare against the C# T5TextEncoder output.

Output: tests/python-reference/t5_reference/
  meta.json                        # prompt list + tokenizer config + checkpoint hash
  prompt_<idx>_tokens.bin          # I32 [seqLen]
  prompt_<idx>_attention_mask.bin  # I32 [seqLen] (1 = valid, 0 = pad)
  prompt_<idx>_hidden.bin          # F32 [1, seqLen, 4096]

The expected layout (token IDs + attention mask + hidden state) is what
`FluxPipeline` / `Sd3Pipeline` consume — i.e. matches the production T5 path.

Usage:
    python dump_t5_xxl_hidden_states.py --t5-repo google/t5-v1_1-xxl --output ./t5_reference
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import numpy as np


PROMPTS = [
    "A photograph of an astronaut riding a horse",
    "Detailed close-up of a hummingbird mid-flight, hovering near a red flower",
    "Empty",
    "A " * 64,
    "Rendered in soft watercolor: a winding mountain path leading to a small wooden cabin at dusk",
]

MAX_LENGTH = 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5-repo", default="google/t5-v1_1-xxl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import T5EncoderModel, T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.t5_repo)
    model = T5EncoderModel.from_pretrained(args.t5_repo, torch_dtype=torch.float32)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    meta = {
        "t5_repo": args.t5_repo,
        "max_length": args.max_length,
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "vocab_size": tokenizer.vocab_size,
        "prompts": [],
    }

    for idx, prompt in enumerate(PROMPTS):
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=args.max_length,
            truncation=True,
        )
        ids = enc.input_ids.to(model.device)
        mask = enc.attention_mask.to(model.device)

        with torch.no_grad():
            out_h = model(input_ids=ids, attention_mask=mask).last_hidden_state.float().cpu()

        ids.cpu().numpy().astype(np.int32).tofile(out / f"prompt_{idx:02d}_tokens.bin")
        mask.cpu().numpy().astype(np.int32).tofile(out / f"prompt_{idx:02d}_attention_mask.bin")
        out_h.numpy().astype(np.float32).tofile(out / f"prompt_{idx:02d}_hidden.bin")

        meta["prompts"].append({
            "index": idx,
            "prompt": prompt,
            "seq_len": int(ids.shape[1]),
            "hidden_dim": int(out_h.shape[-1]),
        })
        print(f"[{idx + 1}/{len(PROMPTS)}] '{prompt[:50]}...' → hidden shape {tuple(out_h.shape)}")

    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nT5 reference written to {out}")


if __name__ == "__main__":
    main()
