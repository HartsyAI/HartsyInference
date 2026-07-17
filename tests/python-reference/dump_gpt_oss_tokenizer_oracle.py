"""Regenerate gpt_oss_tokenizer_oracle.json — expected o200k_harmony token ids for GptOssTokenizerTests.

Extracts the tokenizer.json embedded in the ComfyUI Lens GPT-OSS encoder checkpoint (the
``tokenizer_json`` byte tensor) and tokenizes a small corpus + the full Lens Harmony chat template
with the upstream HuggingFace ``tokenizers`` library.

Usage:
  python dump_gpt_oss_tokenizer_oracle.py \
      --te-checkpoint "<Models>/clip/gpt_oss_20b_nvfp4.safetensors" \
      --out gpt_oss_tokenizer_oracle.json
"""

import argparse
import json
import struct

_LENS_CHAT_SYSTEM = (
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background."
)
_LENS_CHAT_ASSISTANT_THINKING = "Need to generate one image according to the description."
_LENS_CHAT_DATE = "2026-05-23"


def render(prompt: str) -> str:
    """Must stay byte-identical to GptOssTokenizer.RenderChatTemplate."""
    return (
        f"<|start|>system<|message|>"
        f"You are ChatGPT, a large language model trained by OpenAI.\n"
        f"Knowledge cutoff: 2024-06\n"
        f"Current date: {_LENS_CHAT_DATE}\n\n"
        f"Reasoning: medium\n\n"
        f"# Valid channels: analysis, commentary, final. "
        f"Channel must be included for every message.<|end|>"
        f"<|start|>developer<|message|># Instructions\n\n"
        f"{_LENS_CHAT_SYSTEM}\n\n<|end|>"
        f"<|start|>user<|message|>{prompt}<|end|>"
        f"<|start|>assistant<|channel|>analysis<|message|>"
        f"{_LENS_CHAT_ASSISTANT_THINKING}<|end|>"
        f"<|start|>assistant<|channel|>final<|message|>"
    )


def extract_tokenizer_json(te_checkpoint: str) -> bytes:
    with open(te_checkpoint, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        off = header["tokenizer_json"]["data_offsets"]
        f.seek(8 + n + off[0])
        return f.read(off[1] - off[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--te-checkpoint", required=True)
    ap.add_argument("--out", default="gpt_oss_tokenizer_oracle.json")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_str(extract_tokenizer_json(args.te_checkpoint).decode("utf-8"))

    cases = {}
    corpus = [
        "a photo of a corgi wearing a top hat",
        "Hello, world! 12345 test's  \n\n multi  space",
        "Ünïcödé — em-dash… 汉字テスト 🎨🖌️",
        "  leading spaces and trailing   ",
        "CamelCase UPPER lower 3.14159 2026-05-23 a/b/c http://x.y/z?q=1",
        "quote \"double\" 'single' `tick` [bracket] {brace}",
    ]
    for text in corpus:
        cases[text] = tok.encode(text, add_special_tokens=False).ids
    cases["__template__"] = tok.encode(render("a photo of a corgi wearing a top hat"), add_special_tokens=False).ids
    cases["__template_empty__"] = tok.encode(render(""), add_special_tokens=False).ids

    with open(args.out, "w") as f:
        json.dump(cases, f, ensure_ascii=False)
    prefix = render("")
    marker = "<|start|>user<|message|>"
    wrapper = tok.encode(prefix[: prefix.index(marker) + len(marker)], add_special_tokens=False).ids
    print(f"wrote {args.out}; wrapper token count = {len(wrapper)} (Lens requires 97)")


if __name__ == "__main__":
    main()
