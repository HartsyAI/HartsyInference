#!/usr/bin/env python3
"""Reference ids for tests/HartsyInference.LLM.Tests/NanbeigeGgufHeaderTests.cs.

Encodes the sample texts and the default-system chat prompt with transformers' AutoTokenizer built from the checkpoint's own
tokenizer.json/tokenizer_config.json (no remote code) and prints them as C# array bodies.

Run: python -I tests/python-reference/dump_nanbeige_tokenizer_reference.py <dir with tokenizer.json and tokenizer_config.json of Nanbeige/Nanbeige4.3-33B>
"""
import sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
texts = ["The capital of France is Paris. The capital of Germany is", "Hello, world! 12345 + 678 = 813.", "def f(x):\n    return x * 2\n",
         "中文测试：你好，世界。  Multiple   spaces\tand tabs.", "naïve café — “quotes” 😀"]
for text in texts:
    print(repr(text), "->", ", ".join(map(str, tok.encode(text, add_special_tokens=False))))
prompt = tok.apply_chat_template([{"role": "user", "content": "What is the capital of France? Answer with one word."}],
                                 add_generation_prompt=True, tokenize=False)
print(repr(prompt))
print("ChatIds ->", ", ".join(map(str, tok.encode(prompt, add_special_tokens=False))))
