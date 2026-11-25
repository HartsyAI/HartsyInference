"""Chat renders of the Hy4 Flash template, rendered with jinja2 the way transformers' apply_chat_template renders it (ImmutableSandboxedEnvironment,
trim_blocks and lstrip_blocks on, raise_exception and strftime_now defined). The C# JinjaChatTemplate must reproduce each text.

Run in hartsy-ref:cpu:  python dump_chat_renders.py <chat_template.jinja> <output.json>
"""

import datetime
import json
import sys

from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

CASES = [
    {"name": "plain", "messages": [{"role": "user", "content": "hello"}], "add_generation_prompt": True},
    {"name": "no_think", "messages": [{"role": "user", "content": "hello"}], "add_generation_prompt": True, "reasoning_effort": "no_think"},
    {"name": "high", "messages": [{"role": "user", "content": "hello"}], "add_generation_prompt": True, "reasoning_effort": "high"},
    {"name": "system_and_turns", "messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "and 3+3?"},
    ], "add_generation_prompt": True},
    {"name": "no_generation_prompt", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}], "add_generation_prompt": False},
]


def raise_exception(message):
    raise TemplateError(message)


def main(template_path, out_path):
    source = open(template_path, encoding="utf-8").read()
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
    template = env.from_string(source)
    out = []
    for case in CASES:
        kwargs = {
            "messages": case["messages"],
            "add_generation_prompt": case["add_generation_prompt"],
            "bos_token": "<｜hy_start:opensource｜>",
            "eos_token": "<｜hy_end:opensource｜>",
            "tools": None,
            "documents": None,
        }
        if "reasoning_effort" in case:
            kwargs["reasoning_effort"] = case["reasoning_effort"]
        out.append({"name": case["name"], "messages": case["messages"], "add_generation_prompt": case["add_generation_prompt"],
                    "reasoning_effort": case.get("reasoning_effort"), "text": template.render(**kwargs)})
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=1)
    print(f"wrote {len(out)} renders to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
