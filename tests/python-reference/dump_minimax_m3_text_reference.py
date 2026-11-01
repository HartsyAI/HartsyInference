#!/usr/bin/env python3
"""Reference token ids and chat-template ids for MiniMax-M3.1-Flash, from transformers.

Reads the release's tokenizer.json, tokenizer_config.json and chat_template.jinja from a directory and writes
tests/HartsyInference.LLM.Tests/Fixtures/MiniMaxM3/minimax_m3_text_reference.json: ids of plain texts, and ids of two rendered conversations.

Run: python -I tests/python-reference/dump_minimax_m3_text_reference.py <release-dir>
"""
import json
import os
import sys

from transformers import AutoTokenizer

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "HartsyInference.LLM.Tests", "Fixtures", "MiniMaxM3", "minimax_m3_text_reference.json")
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
texts = ["Hello, world!", "  indented\n\ncode: x = 3.14159 // 你好 world's", "naïve café 12345 67", "<mm:think>\nreason</mm:think>"]
conversations = [
    [{"role": "user", "content": "Hello!"}],
    [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."},
     {"role": "user", "content": "Bye"}],
]
rendered = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conversations]
reference = {
    "texts": texts,
    "textIds": [tokenizer.encode(t, add_special_tokens=False) for t in texts],
    "conversations": conversations,
    "conversationIds": [tokenizer.encode(r, add_special_tokens=False) for r in rendered],
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(reference, f, ensure_ascii=False)
print("wrote", OUT, [len(i) for i in reference["conversationIds"]])
