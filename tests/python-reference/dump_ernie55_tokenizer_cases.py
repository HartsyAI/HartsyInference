"""Offline reference dump for the C# ERNIE-5.5 tokenizer and completion prompt (never imported by engine code).

Encodes a fixed set of inputs with the Hugging Face ``tokenizers`` library on the checkpoint's ``tokenizer.json``, with the
default ``add_special_tokens=True`` that a ``/v1/completions`` endpoint uses, and writes ``[{"text": ..., "ids": [...]}]``.
The cases cover the right-to-left digit grouping (``\\p{Nd}{3}+``), CJK, astral characters, code and whitespace.

    python -I dump_ernie55_tokenizer_cases.py --dir <ERNIE-5.5-Flash-Base dir> --out cases.json

Then run ``Ernie5RealCheckpointTests.Real_CompletionPrompt_MatchesTheHfTokenizer`` with ``ERNIE_REAL_DIR`` set to the same
directory and ``ERNIE_TOKENIZER_CASES`` set to ``cases.json``.
"""

import argparse
import json

from tokenizers import Tokenizer

CASES = [
    "Large language models are",
    "1234567",
    "12",
    "0.5 and 1000000 or 31415926535",
    "百度文心大模型是什么？",
    "日本語のテキストとカタカナ",
    "def f(x):\n    return x**2  # square\n\n\n",
    "Emoji 🚀🔥 test, 𝔘𝔫𝔦𝔠𝔬𝔡𝔢!",
    "  leading spaces\tand\ttabs\r\n",
    "e.g. U.S.A.-based (1999) [ok] {x}",
    "a" * 600 + "9" * 1100,
    "Ünïcödé café naïve – “quotes” — dash…",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(f"{args.dir}/tokenizer.json")
    rows = [{"text": text, "ids": tokenizer.encode(text).ids} for text in CASES]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"wrote {len(rows)} cases to {args.out}")


if __name__ == "__main__":
    main()
