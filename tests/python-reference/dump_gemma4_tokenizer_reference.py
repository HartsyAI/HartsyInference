#!/usr/bin/env python3
"""Dumps expected token ids for the REAL Gemma 4 tokenizer so Gemma4Tokenizer can be checked against the
HuggingFace `tokenizers` library rather than only against a hand-built miniature.

The tokenizer.json lives inside the LTX-2.5 text-encoder safetensors as the U8 `tokenizer_json` tensor; extract
it (or point at any copy) and pass it here. The blob is ~32 MB and is not committed, so the C# test that reads
this dump is Integration-tier and skips when the directory is absent.

  "/home/hartsy/Desktop/Swarm/SwarmUI.not too old/dlbackend/ComfyUI/venv/bin/python3" \
      tests/python-reference/dump_gemma4_tokenizer_reference.py --tokenizer-json <path> --output <dir>
"""
import argparse
import json
import os
import shutil

from tokenizers import Tokenizer

PROMPTS = [
    "a cat",
    "A cinematic shot of a red sports car drifting through a rainy neon-lit city street at night.",
    "  leading and trailing spaces  ",
    "unicode: café, 日本語, emoji 🎬, math ∑∫",
    "punctuation!?,.;:'\"()[]{}<>/\\|@#$%^&*-_=+~`",
    "digits 0123456789 and 1,234,567.89",
    "newlines\nand\ttabs",
    "",
    "x",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-json", required=True)
    parser.add_argument("--output", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "gemma4_tokenizer_reference"))
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer_json)
    os.makedirs(args.output, exist_ok=True)
    shutil.copyfile(args.tokenizer_json, os.path.join(args.output, "tokenizer.json"))

    cases = []
    for prompt in PROMPTS:
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        cases.append({"text": prompt, "ids": ids})
        print(f"{len(ids):4d} ids  {prompt!r}")

    with open(os.path.join(args.output, "expected.json"), "w") as handle:
        json.dump({"cases": cases}, handle, indent=2)
    print(f"wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
