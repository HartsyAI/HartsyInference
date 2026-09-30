"""Cross-check parser_reference.json against DeepSeek deepseek-recipe (pip install deepseek-recipe) at ~40 split points per case.

Known intentional differences vs encoding.py (which the C# parser follows): chat_call_value_lookalikes and name_extra_space.
Run from the repo root: python tests/python-reference/deepseek_v41/cross_check_recipe.py
"""
import json, sys
from deepseek_recipe import *
d = json.load(open('tests/python-reference/deepseek_v41/parser_reference/parser_reference.json'))
EOS = d['special']['eos']

def run(case, cut):
    thinking = case['mode'] == 'thinking'
    body = {"model": "m", "messages": [{"role": "user", "content": "q"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
            "thinking": {"type": "enabled" if thinking else "disabled"}}
    conv = ChatCompletionRequest(body).convert(ConversionOptions())
    gen = ChatCompletionRequest.chunk_generator(conv, "id", "m")
    proc = StreamProcessor(gen, conv.parsing_options)
    text = case['text']
    if text.endswith(EOS): text = text[:-len(EOS)]
    parts = [text[:cut], text[cut:]] if cut is not None else [text]
    chunks = []
    chunks += proc.push(InferenceChunk.ready(system_fingerprint="f", prompt_usage=PromptUsage(prompt_tokens=1)))
    for p in parts:
        if p: chunks += proc.push(InferenceChunk.text(p, content_tokens=1))
    chunks += proc.push(InferenceChunk.finish(InferenceFinishReason.Stop))
    chunks += proc.finish()
    reasoning = content = ""; calls = {}
    for c in chunks:
        j = json.loads(c.to_json())
        for ch in j.get('choices', []):
            dl = ch.get('delta', {})
            reasoning += dl.get('reasoning_content') or ''
            content += dl.get('content') or ''
            for tc in dl.get('tool_calls') or []:
                e = calls.setdefault(tc['index'], {'name': '', 'args': ''})
                fn = tc.get('function', {})
                e['name'] += fn.get('name') or ''
                e['args'] += fn.get('arguments') or ''
    return reasoning, content, [calls[k] for k in sorted(calls)]

bad = 0
for c in d['cases']:
    if not c['ok']: continue
    exp_calls = [((t['namespace'] + '::' if t['namespace'] else '') + t['name'], t['arguments']) for t in c['tool_calls']]
    for cut in [None] + list(range(0, len(c['text']) + 1, max(1, len(c['text']) // 40))):
        try:
            r, ct, calls = run(c, cut)
        except Exception as ex:
            print('EXC', c['name'], cut, repr(ex)[:100]); bad += 1; break
        got = [(x['name'], x['args']) for x in calls]
        ok = (r == c['reasoning'] and ct == c['content'] and got == exp_calls)
        if not ok:
            print('DIFF', c['name'], cut, '| reasoning', repr(r)[:50], repr(c['reasoning'])[:50], '| content', repr(ct)[:60], repr(c['content'])[:60], '| calls', got[:1], exp_calls[:1]); bad += 1; break
print('mismatching cases:', bad)
