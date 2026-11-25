"""Writes the token ids the release's own tokenizer gives for a set of probe strings, for the C# tokenizer parity test.

Run in the CPU reference image (never the Windows venv), with the release's tokenizer folder mounted read-only:

    run.sh -v "<worktree>:/wt" -v "<models>/LLM/maxwell-v1-flash:/mx:ro" -- python -I /wt/tools/maxwell-reference/dump_tokenizer.py \
        --tokenizer-dir /mx --out /wt/tests/HartsyInference.LLM.Tests/Fixtures/Maxwell/tokenizer_cases.json

The probes cover ASCII, Han, Thai, Lao, Myanmar, Khmer, digits, punctuation runs, a newline run, an emoji and a rendered chat prompt. Ids are
written without special tokens, which is what the engine's EncodeOrdinary produces; the chat prompt is rendered through the release's
chat_template.jinja with thinking on.
"""

import argparse
import json

from transformers import AutoTokenizer

PROBES = [
    "Why is it harder to keep a room cool when the door is open on a hot day?",
    "The quick brown fox jumps over the lazy dog.",
    "我们今天去公园散步。",
    "สวัสดีครับ วันนี้อากาศดีมาก",
    "ສະບາຍດີ ເຈົ້າສະບາຍດີບໍ",
    "မင်္ဂလာပါ ဒီနေ့ ရာသီဥတု ကောင်းပါတယ်",
    "សួស្តី ថ្ងៃនេះអាកាសធាតុល្អណាស់",
    "Price: 1234567890 dollars, 3.14159.",
    "line one\n\n\nline two\n",
    "emoji 😀🙂 and symbols ©®™ — “quotes”",
    "mixed 中文 English ไทย 123 ok",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=False)
    cases = []
    for text in PROBES:
        cases.append({"text": text, "ids": tok.encode(text, add_special_tokens=False)})

    messages = [{"role": "user", "content": PROBES[0]}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    cases.append({"question": PROBES[0], "chat_prompt": prompt, "ids": tok.encode(prompt, add_special_tokens=False)})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"bos_id": tok.bos_token_id, "eos_id": tok.eos_token_id, "cases": cases}, f, ensure_ascii=False, indent=1)
    print("wrote", len(cases), "cases; bos", tok.bos_token_id, "eos", tok.eos_token_id)


if __name__ == "__main__":
    main()
