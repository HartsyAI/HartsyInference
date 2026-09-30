"""Dumps the DeepSeek-V4.1 completion-parser reference used by the C# streaming-parser tests.

Runs the upstream encoding.py `parse_message_from_completion_text` on (a) the assistant turns of the five upstream golden
fixtures, (b) canonical turns rendered by encoding.py itself, and (c) hand-written well-formed and malformed completions, and
writes parser_reference/parser_reference.json. Deterministic: re-running produces identical output.
Usage: ~/dsv41-ref/.venv/bin/python dump_parser_reference.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("DSV41_REF_ROOT", Path.home() / "dsv41-ref"))
sys.path.insert(0, str(ROOT / "upstream" / "encoding"))

import encoding as enc  # noqa: E402

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "encoder_reference" / "encoding"
OUT = HERE / "parser_reference"
ASSISTANT = "<｜Assistant｜>"


def parse(text, mode):
    """Returns (ok, result-or-error) from the upstream parser without raising."""
    try:
        message = enc.parse_message_from_completion_text(text, mode)
    except (AssertionError, ValueError) as error:
        return False, str(error)
    calls = []
    for entry in message["tool_calls"]:
        calls.append({
            "name": entry["function"]["name"],
            "namespace": entry.get("namespace"),
            "arguments": entry["function"]["arguments"],
        })
    return True, {"reasoning": message["reasoning_content"], "content": message["content"], "tool_calls": calls}


def record(name, source, text, mode, prompt=None):
    ok, result = parse(text, mode)
    entry = {"name": name, "source": source, "mode": mode, "text": text, "ok": ok}
    if ok:
        entry.update(result)
    else:
        entry["error"] = result
    if prompt is not None:
        entry["prompt"] = prompt
    return entry


def golden_turns():
    """Assistant completions found in the golden prompts, cut at their EOS."""
    turns = []
    for index in range(1, 6):
        prompt = (GOLDEN / f"test_output_{index}.txt").read_text(encoding="utf-8")
        for turn, segment in enumerate(prompt.split(ASSISTANT)[1:]):
            end = segment.find(enc.eos_token)
            if end < 0:
                continue
            body = segment[: end + len(enc.eos_token)]
            if body.startswith(enc.thinking_start_token):
                turns.append(record(f"golden_{index}_turn{turn}", "golden", body[len(enc.thinking_start_token):], "thinking"))
            elif body.startswith(enc.thinking_end_token):
                turns.append(record(f"golden_{index}_turn{turn}", "golden", body[len(enc.thinking_end_token):], "chat"))
    return turns


def canonical(name, mode, content, reasoning=None, calls=None):
    """Renders one assistant turn with encoding.py and records the completion part of that prompt."""
    assistant = {"role": "assistant", "content": content}
    if reasoning is not None:
        assistant["reasoning_content"] = reasoning
    if calls:
        assistant["tool_calls"] = [
            {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}}
            for i, (n, a) in enumerate(calls)
        ]
    prompt = enc.encode_messages([{"role": "user", "content": "q"}, assistant], mode, drop_thinking=False)
    tail = prompt.split(ASSISTANT)[1]
    prefix = enc.thinking_start_token if mode == "thinking" else enc.thinking_end_token
    assert tail.startswith(prefix), tail[:20]
    return record(name, "canonical", tail[len(prefix):], mode, prompt=prompt)


def canonical_cases():
    weird = 'He said "hi"\\path\nline2\ttab\u0001ctl < > & \u65e5\u672c\u8a9e \U0001f600 caf\u00e9'
    return [
        canonical("chat_plain", "chat", "Hello there."),
        canonical("chat_multiline", "chat", "Line one\n\nLine two\n"),
        canonical("chat_lookalike_text", "chat", "a\n\n<b>c</b> and <｜not a tag"),
        canonical("chat_unicode", "chat", "\u65e5\u672c\u8a9e \U0001f600 caf\u00e9 \u0928\u092e\u0938\u094d\u0924\u0947"),
        canonical("chat_empty", "chat", ""),
        canonical("thinking_plain", "thinking", "Answer.", reasoning="Let me think.\nStep 2"),
        canonical("thinking_empty_reasoning", "thinking", "Hi", reasoning=""),
        canonical("thinking_lookalike_reasoning", "thinking", "ok", reasoning="a < b \n\n<｜ maybe </think"),
        canonical("thinking_single_call", "thinking", "I will look it up.", reasoning="Need data.",
                  calls=[("lookup", {"query": "cats", "limit": 3})]),
        canonical("thinking_call_no_content", "thinking", "", reasoning="r", calls=[("lookup", {"query": "x"})]),
        canonical("chat_two_calls", "chat", "Checking two things.", calls=[
            ("weather::get_forecast", {"city": "Paris", "days": 3, "metric": True}),
            ("search", {"q": weird, "filters": {"a": [1, 2, {"b": None}]}, "empty": "", "ratio": 0.5}),
        ]),
        canonical("chat_zero_param_call", "chat", "", calls=[("ping", {})]),
        canonical("chat_call_value_lookalikes", "chat", "x", calls=[("t", {"v": "</｜DSML｜ nothing> < / >"})]),
        canonical("thinking_three_calls", "thinking", "Doing three.", reasoning="plan", calls=[
            ("a", {"x": 1}), ("ns::b", {"y": "two"}), ("c", {}),
        ]),
    ]


def handwritten_cases():
    eos = enc.eos_token
    calls_open = "\n\n<｜DSML｜ calls>\n"
    invoke = '<｜DSML｜ invoke name="{n}">\n{p}</｜DSML｜ invoke>\n'
    param = '<｜DSML｜ parameter name="{k}" string="{s}">{v}</｜DSML｜ parameter>\n'
    calls_end = "</｜DSML｜ calls>"
    one = invoke.format(n="f", p=param.format(k="a", s="true", v="x"))
    cases = []

    def add(name, text, mode):
        cases.append(record(name, "handwritten", text, mode))

    add("calls_without_eos", "ok" + calls_open + one + calls_end, "chat")
    add("calls_only_no_content", calls_open + one + calls_end + eos, "chat")
    add("name_extra_space", "x" + calls_open + '<｜DSML｜ invoke  name="f">\n</｜DSML｜ invoke>\n' + calls_end + eos, "chat")
    add("thinking_reasoning_then_eos", "r</think>done" + eos, "thinking")
    add("err_thinking_missing_end", "just reasoning" + eos, "thinking")
    add("err_thinking_calls_in_reasoning", "r" + calls_open + one + calls_end + eos, "thinking")
    add("err_think_start_in_content", "a<think>b" + eos, "chat")
    add("err_think_end_in_chat_content", "a</think>b" + eos, "chat")
    add("err_dsml_in_content", "a ｜DSML｜ b" + eos, "chat")
    add("err_bos_in_content", "a <｜begin▁of▁sentence｜> b" + eos, "chat")
    add("err_dsml_in_reasoning", "a ｜DSML｜ b</think>c" + eos, "thinking")
    add("err_duplicate_param", "x" + calls_open + invoke.format(
        n="f", p=param.format(k="a", s="true", v="1") + param.format(k="a", s="true", v="2")) + calls_end + eos, "chat")
    add("err_bad_gap_after_calls_open", "x\n\n<｜DSML｜ calls>junk\n" + one + calls_end + eos, "chat")
    add("err_garbage_after_calls_end", "x" + calls_open + one + calls_end + "tail" + eos, "chat")
    add("err_param_missing_string_attr", "x" + calls_open + invoke.format(
        n="f", p='<｜DSML｜ parameter name="a">1</｜DSML｜ parameter>\n') + calls_end + eos, "chat")
    add("err_nested_namespace", "x" + calls_open + invoke.format(n="a::b::c", p="") + calls_end + eos, "chat")
    add("err_call_truncated", "x" + calls_open + '<｜DSML｜ invoke name="f">\n<｜DSML｜ parameter name="a" string="true">par', "chat")
    add("err_call_truncated_after_invoke", "x" + calls_open + one, "chat")
    add("err_bad_name_attr", "x" + calls_open + '<｜DSML｜ invoke nom="f">\n</｜DSML｜ invoke>\n' + calls_end + eos, "chat")
    return cases


def main():
    cases = golden_turns() + canonical_cases() + handwritten_cases()
    OUT.mkdir(exist_ok=True)
    payload = {"special": {"bos": enc.bos_token, "eos": enc.eos_token, "think": enc.thinking_start_token,
                           "think_end": enc.thinking_end_token, "dsml": enc.dsml_token}, "cases": cases}
    (OUT / "parser_reference.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    ok = sum(1 for c in cases if c["ok"])
    print(f"{len(cases)} cases ({ok} ok, {len(cases) - ok} rejected by encoding.py)")


if __name__ == "__main__":
    main()
