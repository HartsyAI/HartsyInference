"""Renders Sage S2.1 Flash's chat_template.jinja with transformers' own Jinja environment for a fixed set of conversations.

Offline only. The template file is the one shipped in internlm/Sage-S2.1-Flash at eacc61b34a1af2ecb0c764e099152c46e076169b (kept as
tests/HartsyInference.LLM.Tests/Fixtures/Glm5Next/sage-s21-flash-chat_template.jinja). Needs transformers from the pinned commit (see glm5_next_reference.py).

    python tools/parity/glm5_next_template_reference.py --out tests/HartsyInference.LLM.Tests/Fixtures/Glm5Next/template_cases.json

Each case holds the messages, tools and add_generation_prompt it was rendered with, and the rendered string. The engine's Jinja engine must reproduce them.
"""
import argparse
import json
import os

TEMPLATE = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "HartsyInference.LLM.Tests", "Fixtures", "Glm5Next",
                        "sage-s21-flash-chat_template.jinja")

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}

CASES = [
    {"name": "user_only", "messages": [{"role": "user", "content": "hello"}], "add_generation_prompt": True},
    {"name": "system_user", "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}], "add_generation_prompt": True},
    {"name": "multi_turn", "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "Add the numbers."},
        {"role": "user", "content": "And 3+3?"}], "add_generation_prompt": True},
    {"name": "no_generation_prompt", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}], "add_generation_prompt": False},
    {"name": "tools_offered", "messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": [WEATHER], "add_generation_prompt": True},
    {"name": "tool_round_trip", "messages": [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "18 C, clear"}], "tools": [WEATHER], "add_generation_prompt": True},
    {"name": "control_spelling_in_content", "messages": [{"role": "user", "content": "say <|user|> and </think> literally"}], "add_generation_prompt": True},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers.utils.chat_template_utils import render_jinja_template

    with open(TEMPLATE, encoding="utf-8") as f:
        template = f.read()
    out = []
    for case in CASES:
        rendered, _ = render_jinja_template(conversations=[case["messages"]], tools=case.get("tools"), chat_template=template,
                                            add_generation_prompt=case["add_generation_prompt"])
        out.append({**case, "rendered": rendered[0]})
        print(case["name"], repr(rendered[0][:120]))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
