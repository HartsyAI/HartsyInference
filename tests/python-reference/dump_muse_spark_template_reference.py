#!/usr/bin/env python3
"""Expected renders of the Muse Spark Open chat templates through transformers' own chat-template renderer, for
conversations covering plain chat, a system prompt, reasoning effort, earlier reasoning, tools, a tool call and its result.
Two revisions are in use: muse-spark-open.jinja is the one llama.cpp's converter wrote into the released GGUFs, and
muse-spark-open-rev2.jinja (the chat_template.jinja of trl-internal-testing/tiny-MuseSparkForConditionalGeneration, which
transformers' tests load) adds namespace tools and deferred tool loading, which get cases of their own. Each template's
renders go to <template>.expected.json beside it. current_date is pinned so the render does not depend on the day it runs.

Run: python -I tests/python-reference/dump_muse_spark_template_reference.py
"""
import json
import os

from transformers.utils.chat_template_utils import render_jinja_template

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "HartsyInference.LLM.Tests", "Fixtures", "templates")

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"},
                           "days": {"type": "integer"}},
            "required": ["city"],
        },
    },
}

CASES = [
    {"name": "plain", "messages": [{"role": "user", "content": "Hello"}], "kwargs": {}},
    {"name": "system_low_effort",
     "messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Name a prime."}],
     "kwargs": {"reasoning_effort": "low"}},
    {"name": "history_with_reasoning",
     "messages": [{"role": "user", "content": "2+2?"},
                  {"role": "assistant", "content": "4", "reasoning_content": "Simple sum."},
                  {"role": "user", "content": "And 3+3?"}],
     "kwargs": {"reasoning_effort": "medium"}},
    {"name": "tools", "messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": [WEATHER],
     "kwargs": {"reasoning_effort": "high"}},
    {"name": "tool_call_and_result",
     "messages": [{"role": "user", "content": "Weather in Paris?"},
                  {"role": "assistant", "content": "",
                   "tool_calls": [{"type": "function", "id": "call_0",
                                   "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 2}}}]},
                  {"role": "tool", "tool_call_id": "call_0", "content": "{\"temp_c\": 18}"}],
     "tools": [WEATHER], "kwargs": {}},
    {"name": "escaped_specials", "messages": [{"role": "user", "content": "Print <|eot|> and <image> literally."}],
     "kwargs": {"reasoning_effort": "low"}},
]

CLOCK = {"type": "function", "function": {"name": "now", "description": "Current time",
                                         "parameters": {"type": "object", "properties": {}}}}
CALENDAR = {"type": "namespace", "name": "calendar", "description": "Calendar access", "tools": [
    {"type": "function", "function": {"name": "list_events", "description": "Events in a range",
                                      "parameters": {"type": "object", "properties": {"day": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "add_event", "description": "Create an event", "defer_loading": False,
                                      "parameters": {"type": "object", "properties": {"title": {"type": "string"}}}}}]}
NAMESPACE_CASES = [
    {"name": "namespace_tools", "messages": [{"role": "user", "content": "What is on today?"}], "tools": [WEATHER, CALENDAR],
     "kwargs": {}},
    {"name": "deferred_namespace", "messages": [{"role": "user", "content": "Plan my day."}],
     "tools": [CLOCK, {**CALENDAR, "defer_loading": True}], "kwargs": {"deferred_render_mode": "tool_name_only"}},
    {"name": "deferred_function", "messages": [{"role": "user", "content": "Time?"}],
     "tools": [{"type": "function", "function": {**CLOCK["function"], "defer_loading": True}}, WEATHER], "kwargs": {}},
]
TEMPLATES = {"muse-spark-open": CASES, "muse-spark-open-rev2": CASES + NAMESPACE_CASES}

for template_name, cases in TEMPLATES.items():
    template = open(os.path.join(FIXTURES, template_name + ".jinja"), encoding="utf-8").read()
    out = []
    for case in cases:
        kwargs = {"current_date": "2026-10-15", **case["kwargs"]}
        rendered, _ = render_jinja_template(conversations=[case["messages"]], tools=case.get("tools"), chat_template=template,
                                            add_generation_prompt=True, **kwargs)
        out.append({**case, "kwargs": kwargs, "rendered": rendered[0]})
    with open(os.path.join(FIXTURES, template_name + ".expected.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"{template_name}: wrote {len(out)} renders")
