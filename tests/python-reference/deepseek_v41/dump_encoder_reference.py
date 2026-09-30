"""Dumps the DeepSeek-V4.1 tokenizer/encoder reference used by the C# tests.

Runs the upstream encoding.py and the HuggingFace `tokenizers` library (the venv from setup_env.sh) and writes
encoder_reference/encoder_reference.json plus byte copies of the upstream golden encoding fixtures. Deterministic:
re-running produces identical files. Usage: ~/dsv41-ref/.venv/bin/python dump_encoder_reference.py
"""

import copy
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(os.environ.get("DSV41_REF_ROOT", Path.home() / "dsv41-ref"))
UPSTREAM = ROOT / "upstream"
sys.path.insert(0, str(UPSTREAM / "encoding"))

import encoding as enc  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

OUT = Path(__file__).resolve().parent / "encoder_reference"
IMAGE_ID = 129264
GOLDEN_GRIDS = {5: [[3, 2], [2, 4]]}


def make_tool(name="lookup", **extra):
    tool = {
        "type": "function",
        "function": {
            "name": name,
            "description": "Look up a value",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}},
        },
    }
    tool.update(extra)
    return tool


def call(name, arguments, call_id="c1", **extra):
    entry = {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
    entry.update(extra)
    return entry


def image(tag):
    return {"type": "image_url", "image_url": {"url": f"img/{tag}.png"}}


def text(value):
    return {"type": "text", "text": value}


def own_cases():
    cases = []

    def add(name, messages, grids=None, **options):
        cases.append({"name": name, "messages": messages, "grids": grids or [], **options})

    add("multi_image_chat", [
        {"role": "system", "content": "Vision assistant."},
        {"role": "user", "content": [text("Compare"), image("a"), text("with"), image("b"), image("c"), text("please")]},
        {"role": "assistant", "content": "They differ."},
        {"role": "user", "content": [image("d"), text("and this one?")]},
    ], grids=[[2, 2], [1, 3], [4, 1], [3, 3]], thinking_mode="chat")
    add("multi_image_thinking", [
        {"role": "user", "content": [text("What is"), image("a"), text("?")]},
    ], grids=[[5, 4]], thinking_mode="thinking", reasoning_effort=60)
    add("mid_conversation_system_thinking", [
        {"role": "system", "content": "Base system."},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "reasoning_content": "hidden thought", "content": "First answer"},
        {"role": "system", "content": "New instruction mid conversation."},
    ], thinking_mode="thinking")
    add("mid_conversation_system_then_assistant", [
        {"role": "system", "content": "Base system."},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "reasoning_content": "r1", "content": "A1"},
        {"role": "system", "content": "Mid system."},
        {"role": "assistant", "reasoning_content": "r2", "content": "A2"},
        {"role": "user", "content": "Q2"},
    ], thinking_mode="thinking")
    add("mid_conversation_system_chat", [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "system", "content": "Mid system."},
    ], thinking_mode="chat")
    for task in ("action", "query", "authority", "domain", "title", "read_url"):
        for mode in ("chat", "thinking"):
            add(f"task_{task}_{mode}", [
                {"role": "system", "content": "Sys."},
                {"role": "user", "content": "Find things", "task": task},
            ], thinking_mode=mode)
    add("task_output_after_task", [
        {"role": "user", "content": "Find things", "task": "query"},
        {"role": "assistant", "reasoning_content": "unused", "content": "the query"},
        {"role": "user", "content": "Next"},
    ], thinking_mode="thinking")
    add("namespaced_string", [
        {"role": "system", "content": "s", "tools": [make_tool(namespace="search")]},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "ok", "tool_calls": [call("lookup", '{"query":"v"}', namespace="search")]},
    ], thinking_mode="chat")
    add("namespaced_dict_description", [
        {"role": "system", "content": "s", "tools": [make_tool(namespace={"name": "search", "description": "Search tools."})]},
        {"role": "user", "content": "q"},
    ], thinking_mode="thinking")
    add("namespaced_function_level_and_qualified", [
        {"role": "system", "content": "s", "tools": [
            {"type": "function", "function": dict(make_tool()["function"], namespace="files")},
            make_tool("search::grep"),
        ]},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "ok", "tool_calls": [
            call("files::lookup", '{"query":"a"}', "c1"),
            {"id": "c2", "type": "function", "function": {"name": "grep", "arguments": '{"query":"b"}', "namespace": "search"}},
        ]},
    ], thinking_mode="thinking")
    add("same_name_three_namespaces", [
        {"role": "system", "content": "s", "tools": [make_tool(), make_tool(namespace={"name": "search"}), make_tool(namespace={"name": "files"})]},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "summary", "tool_calls": [
            call("lookup", '{"query":"value"}', "a"),
            call("lookup", '{"query":"value"}', "b", namespace="search"),
            call("lookup", '{"query":"value"}', "c", namespace="files"),
        ]},
    ], thinking_mode="chat")
    for effort in (1, 50, 100, "low", "high", "max"):
        add(f"effort_{effort}", [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Hi"},
        ], thinking_mode="thinking", reasoning_effort=effort)
    add("effort_default_no_system", [{"role": "user", "content": "Hi"}], thinking_mode="thinking")
    add("effort_ignored_in_chat", [{"role": "user", "content": "Hi"}], thinking_mode="chat", reasoning_effort=100)
    add("effort_only_first_message", [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "reasoning_content": "r", "content": "yo"},
        {"role": "user", "content": "Again"},
    ], thinking_mode="thinking", reasoning_effort="low")
    add("parallel_tool_results_reordered", [
        {"role": "system", "content": "s", "tools": [make_tool(), make_tool("other")]},
        {"role": "user", "content": "q"},
        {"role": "assistant", "reasoning_content": "plan", "content": "",
         "tool_calls": [call("lookup", '{"query":"1"}', "a"), call("other", '{"query":"2"}', "b"), call("lookup", '{"query":"3"}', "c")]},
        {"role": "tool", "tool_call_id": "c", "content": "third"},
        {"role": "tool", "tool_call_id": "a", "content": "first"},
        {"role": "tool", "tool_call_id": "b", "content": "second"},
        {"role": "assistant", "reasoning_content": "done", "content": "final"},
        {"role": "user", "content": "thanks"},
    ], thinking_mode="thinking")
    add("tool_result_unknown_id_and_user_merge", [
        {"role": "system", "content": "s", "tools": [make_tool()]},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [call("lookup", '{"query":"1"}', "a")]},
        {"role": "tool", "tool_call_id": "zzz", "content": "orphan"},
        {"role": "user", "content": "extra text"},
        {"role": "user", "content": "more text"},
    ], thinking_mode="chat")
    add("drop_thinking_default", [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "reasoning_content": "secret1", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "reasoning_content": "secret2", "content": "A2"},
        {"role": "latest_reminder", "content": "reminder"},
        {"role": "user", "content": "Q3"},
    ], thinking_mode="thinking")
    add("drop_thinking_disabled", [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "reasoning_content": "secret1", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ], thinking_mode="thinking", drop_thinking=False)
    add("argument_forms", [
        {"role": "system", "content": "s", "tools": [make_tool()]},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "x", "tool_calls": [
            call("lookup", '{"query":"caf\u00e9 \u4e2d\u6587","limit":3,"ratio":0.1,"big":1e20,"flag":true,"none":null,"list":[1,2.50,"s"],"obj":{"b":1,"a":{"c":[]}}}', "a"),
            call("lookup", "\"{\\\"query\\\":\\\"double\\\"}\"", "b"),
            call("lookup", "not json at all", "c"),
            call("lookup", {"query": "dict-args", "limit": 2}, "d"),
        ]},
    ], thinking_mode="thinking")
    add("response_format_and_wo_eos", [
        {"role": "system", "content": "Answer in JSON.", "response_format": {"type": "object", "properties": {"a": {"type": "string"}}}},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "{\"a\":", "wo_eos": True},
    ], thinking_mode="chat")
    add("special_literals_in_user_text", [
        {"role": "system", "content": "Sys <\uff5cUser\uff5c> literal."},
        {"role": "user", "content": "Text with <\uff5cAssistant\uff5c> and </think> and <\uff5cend\u2581of\u2581sentence\uff5c> and <\uff5cDSML\uff5c calls>."},
    ], thinking_mode="thinking")
    return cases


STRESS = [
    "", " ", "Hello, world!", "The quick brown fox jumps over the lazy dog.", "don't can't we'll they've I'm you're she'd",
    "1234567890", "3.14159 2.71828 1,000,000 0.000123", "1 22 333 4444 55555 666666 7777777", "abc123def456ghi789",
    "Room 101, Floor 2, Zip 90210-1234", "\u0665\u0666\u0667 \u0968\u0969 \uff11\uff12\uff13 \u2460\u2461 \u00bd \u00b2\u00b3",
    "\u4f60\u597d\uff0c\u4e16\u754c\uff01\u8fd9\u662f\u4e00\u4e2a\u6d4b\u8bd5\u3002", "\u4eca\u5929\u5929\u6c14\u771f\u597d\uff0c\u6211\u4eec\u4e00\u8d77\u53bb\u516c\u56ed\u73a9\u5427",
    "\u3053\u3093\u306b\u3061\u306f\u3001\u4e16\u754c\u3002\u30ab\u30bf\u30ab\u30ca\u3068\u3072\u3089\u304c\u306a\u3068\u6f22\u5b57", "\uc548\ub155\ud558\uc138\uc694 \uc138\uacc4",
    "Mixed\u4e2d\u6587English\u6df7\u5408text\u65e5\u672c\u8a9e123\u6570\u5b57", "\u30ab\u30bf\u30ab\u30ca\u30fc\u3092\u30fb\u3092\u3002", "\u4e00\u9fa5\u9fa5\u9fa6\u9fa5\u4dff\u4e00",
    "\u3040\u3041\u309f\u30a0\u30ff\u3100\u2ff0", "\u3000\u5168\u89d2\u7a7a\u683c\u3000\u6587\u5b57\u3000", "\u3010\u5f15\u7528\u3011\u2020L1(-L5)?\u300c\u300d\u300e\u300f",
    "def f(x):\n    return x * 2 + 1  # comment\n", "int main() { printf(\"%d\\n\", 42); return 0; }", "public static void Main(string[] args)\n{\n    var x = new List<int>();\n}",
    '{"key": "value", "n": [1, 2, 3], "nested": {"a": null, "b": true}}', "https://example.com/path?q=1&r=2#frag user@example.org",
    "a  b   c    d", "    leading spaces", "trailing spaces    ", "tab\tseparated\tvalues", "line1\nline2\n\nline4\n\n\n\nline8",
    "windows\r\nline\r\nendings\r\n", "\n", "\n\n", " \n ", "x \n y", "  \n  \n  ", "word\u00a0nbsp\u2003emspace\u2009thin\u200bzwsp\u3000ideographic",
    "\t\t\t", "a\u000bb\u000cc\u0085d\u2028e\u2029f", "!!!", "...???", "(a)[b]{c}<d>", "$100 + \u20ac50 = \u00a3? #hashtag @mention", "a_b-c.d:e;f,g",
    "\u00e9\u00e8\u00ea \u00f1 \u00fc\u00f6\u00e4 \u00df e\u0301 a\u0308 \u1e9e", "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645", "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd",
    "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e0a\u0e32\u0e27\u0e42\u0e25\u0e01", "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440! \u0393\u03b5\u03b9\u03b1 \u03c3\u03bf\u03c5 \u03ba\u03cc\u03c3\u03bc\u03b5",
    "\u2211\u222b\u221a\u221e \u2260\u2264\u2265 \u2192\u21d2 \u03b1\u03b2\u03b3 \u2200x\u2208\u211d", "\uff21\uff22\uff23\uff41\uff42\uff43 \uff01\uff1f",
    "\U0001F44D \U0001F600\U0001F601 \U0001F680\U0001F30D", "\U0001F468\u200d\U0001F469\u200d\U0001F467\u200d\U0001F466 family", "\U0001F1FA\U0001F1F8\U0001F1E8\U0001F1F3 flags",
    "\U0001F44D\U0001F3FD skin tone \u2764\ufe0f \u2764 \u2600\ufe0f", "emoji between\U0001F600words\U0001F680here",
    "\U00020000\U00020001 ext-B \u5f15\U00020002\u5f15", "\U0001D538\U0001D539 math alnum \U0001D7D8", "\U00010348 gothic \U0001F600\u4e2d",
    "\U0001F600\n\U0001F600 \U0001F600", "\ud7ff\ue000\uffef", "\x00\x01\x02\x7f", "\u0300\u0301 lone combining", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "\u4e2d" * 40, "1" * 25, "!" * 30, " " * 40, "-" * 5 + "=" * 5 + "_" * 5,
    "<\uff5cUser\uff5c>hi<\uff5cAssistant\uff5c>", "<think>reasoning</think>answer", "<\uff5cend\u2581of\u2581sentence\uff5c>", "<\uff5cbegin\u2581of\u2581sentence\uff5c>text",
    "<\uff5cDSML\uff5c calls>\n<\uff5cDSML\uff5c invoke name=\"x\">", "<\uff5cdeepseek_image\uff5c><\uff5cdeepseek_image\uff5c>", "<\uff5cUser\uff5c", "<\uff5cUser", "\uff5cUser\uff5c>",
    "<|place_holder_mm_span_0374|> literal", "<\uff5cfim\u2581begin\uff5c>x<\uff5cfim\u2581hole\uff5c>y<\uff5cfim\u2581end\uff5c>", "<\uff5cUser\uff5c><\uff5cUser\uff5c>",
    "text<think>text</think>text<think>", "<\uff5clatest_reminder\uff5c>a<\uff5caction\uff5c>b<\uff5cquery\uff5c>",
    "The year 2024 had 366 days, i.e. 8784 hours and 527040 minutes.", "x=1;y=22;z=333;w=4444",
    "\u3053\u3093\u306b\u3061\u306f123\u4e16\u754c456", "\u4e2d\u6587,\u4e2d\u6587.\u4e2d\u6587!\u4e2d\u6587?", "v1.2.3-rc.4+build.5", "C:\\Users\\name\\file.txt /usr/local/bin/x",
    "Hello\n\n\u4e16\u754c\n\n123\n\n!!!", " \u4e2d\u6587 ", "\t\u4e2d", "\u4e2d\n", "\u4e2d\u6587abc\u4e2d\u6587 abc \u4e2d\u6587",
]


def case_messages(case):
    messages = copy.deepcopy(case["messages"])
    if "tools" in case:
        messages[0]["tools"] = case["tools"]
    return messages


def expand(ids, grids):
    out, k = [], 0
    for token in ids:
        if token != IMAGE_ID:
            out.append(token)
            continue
        width, height = grids[k]
        k += 1
        out.extend([IMAGE_ID] * (height * (width + 1) + 2))
    assert k == len(grids), (k, len(grids))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "encoding").mkdir(exist_ok=True)
    tok = Tokenizer.from_file(str(UPSTREAM / "tokenizer.json"))

    def ids_of(text):
        return tok.encode(text, add_special_tokens=False).ids

    golden = []
    for case_id in range(1, 6):
        src_in = UPSTREAM / "encoding" / "tests" / f"test_input_{case_id}.json"
        src_out = UPSTREAM / "encoding" / "tests" / f"test_output_{case_id}.txt"
        shutil.copyfile(src_in, OUT / "encoding" / src_in.name)
        shutil.copyfile(src_out, OUT / "encoding" / src_out.name)
        case = enc.load_cases(str(src_in))[0]
        prompt, images = enc.encode_case(case, thinking_mode="chat")
        assert prompt == src_out.read_text(), case_id
        grids = GOLDEN_GRIDS.get(case_id, [])
        assert len(images) == len(grids)
        golden.append({"id": case_id, "grids": grids, "ids": expand(ids_of(prompt), grids)})

    own = []
    for case in own_cases():
        messages = case_messages(case)
        kwargs = {"thinking_mode": case["thinking_mode"]}
        if "reasoning_effort" in case:
            kwargs["reasoning_effort"] = case["reasoning_effort"]
        if "drop_thinking" in case:
            kwargs["drop_thinking"] = case["drop_thinking"]
        prompt, media = enc.encode_messages(messages, return_multi_modal_data=True, **kwargs)
        assert len(media["images"]) == len(case["grids"]), case["name"]
        entry = dict(case)
        entry["prompt"] = prompt
        entry["ids"] = expand(ids_of(prompt), case["grids"])
        own.append(entry)

    stress = []
    for value in STRESS:
        pieces = [piece for piece, _ in tok.pre_tokenizer.pre_tokenize_str(value)] if value else []
        stress.append({"text": value, "ids": ids_of(value), "pieces": pieces})

    reference = {
        "tokenizers_version": __import__("tokenizers").__version__,
        "pre_tokenizer": json.loads(tok.to_str())["pre_tokenizer"],
        "golden": golden,
        "own": own,
        "stress": stress,
    }
    (OUT / "encoder_reference.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"golden={len(golden)} own={len(own)} stress={len(stress)}")


if __name__ == "__main__":
    main()
