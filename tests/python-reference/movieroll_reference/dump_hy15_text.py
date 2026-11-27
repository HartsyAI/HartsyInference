"""Reference strings for the HunyuanVideo-1.5 / MovieRoll text side: the video chat system prompt and the glyph-text extraction and formatting.

  bash .claude/run-outputs/refenv/run.sh -v "<worktree>:/wt" -v "<MovieRoll checkout at 0b5f73e>:/src:ro" -- \
      python -I /wt/tests/python-reference/movieroll_reference/dump_hy15_text.py --out /wt/tests/HartsyInference.Diffusion.Tests/Fixtures/hy15_text.json

The system prompt is read with ast.literal_eval from the reference source, so backslash line continuations resolve exactly as Python resolves them.
extract_glyph_texts and MultilingualPromptFormat.format_prompt are the reference functions themselves (format_prompt needs no font or colour tables
when no style is given, so the instance is built without its JSON files).
"""
import argparse
import ast
import importlib.util
import json
import sys

sys.path.insert(0, "/src")

CASES = [
    'a cat holds a sign that says "HELLO WORLD" at noon',
    'two signs: "OPEN" and “CLOSED” on a door',
    'curly first “B” then straight "A"',
    'repeats "X" and "X" and "Y"',
    'a single repeat "Z"',
    'single quotes ‘not read’ and "read"',
    'no quotes at all',
    '',
    'empty quotes "" here',
    'multi\nline "one" and\n"two"',
    '中文 "你好" and “世界”',
]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def system_prompt(source_path):
    tree = ast.parse(open(source_path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "PROMPT_TEMPLATE_ENCODE_VIDEO_JSON" for t in node.targets):
            template = ast.literal_eval(node.value)
            assert [m["role"] for m in template] == ["system", "user"] and template[1]["content"] == "{}"
            return template[0]["content"]
    raise SystemExit("PROMPT_TEMPLATE_ENCODE_VIDEO_JSON not found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    glyph = load_module("glyph_text", "/src/movieroll/models/text_encoders/glyph_text.py")
    fmt = load_module("format_prompt", "/src/movieroll/models/text_encoders/byT5/format_prompt.py")
    formatter = object.__new__(fmt.MultilingualPromptFormat)
    cases = []
    for prompt in CASES:
        texts = glyph.extract_glyph_texts(prompt)
        formatted = formatter.format_prompt(texts, [{"color": None, "font-family": None} for _ in texts]) if texts else None
        cases.append({"prompt": prompt, "texts": texts, "formatted": formatted})
    out = {
        "source": "Tencent-Hunyuan/MovieRoll@0b5f73e2d7cad86826dcf8fc420991d789d88646",
        "system_prompt": system_prompt("/src/movieroll/models/text_encoders/__init__.py"),
        "glyph_cases": cases,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote", args.out, len(cases), "glyph cases")


if __name__ == "__main__":
    main()
