"""Dump CosyVoice 2's reference text frontend for parity tests.

Run with a venv holding the reference pins (inflect==7.3.1, wetext==0.0.4, transformers) and the upstream repo:
    python tools/parity/cosyvoice_frontend_dump.py <CosyVoice repo> <CosyVoice-BlankEN tokenizer dir> <out.json>
Writes number_to_words cases, the frontend's segments with and without wetext, and the Qwen token ids per segment.
"""
import json
import sys

COSY, TOK, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, COSY)

# wetext 0.0.4 calls ModelScope's snapshot_download on every Normalizer() even when the FSTs are cached, and the
# ModelScope API rate-limits; point it at the local snapshot when one exists so runs are reproducible offline.
import os as _os
import modelscope as _modelscope
_WETEXT_CACHE = _os.path.expanduser("~/.cache/modelscope/models/pengzhendong--wetext/snapshots/master")
if _os.path.isdir(_WETEXT_CACHE):
    _modelscope.snapshot_download = lambda *a, **k: _WETEXT_CACHE
import inflect  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from wetext import Normalizer  # noqa: E402
from cosyvoice.utils.frontend_utils import (  # noqa: E402
    contains_chinese, is_only_punctuation, remove_bracket, replace_blank, replace_corner_mark, spell_out_number,
    split_paragraph)

inf = inflect.engine()
# CosyVoice2Tokenizer (cosyvoice/tokenizer/tokenizer.py) without importing it (it pulls torch/whisper): the same
# AutoTokenizer plus the same added special tokens, in the same order.
SPECIALS = ['<|im_start|>', '<|im_end|>', '<|endofprompt|>', '[breath]', '<strong>', '</strong>', '[noise]',
            '[laughter]', '[cough]', '[clucking]', '[accent]', '[quick_breath]', "<laughter>", "</laughter>",
            "[hissing]", "[sigh]", "[vocalized-noise]", "[lipsmack]", "[mn]"]
tok = AutoTokenizer.from_pretrained(TOK)
tok.add_special_tokens({'eos_token': '<|endoftext|>', 'pad_token': '<|endoftext|>', 'additional_special_tokens': SPECIALS})
try:
    en_tn = Normalizer(lang="en", operator="tn") if "lang" in Normalizer.__init__.__code__.co_varnames else Normalizer()
    zh_tn = Normalizer(remove_erhua=False)
except Exception as e:  # FSTs unavailable (wetext 0.0.4 fetches them from ModelScope): dump the plain path only
    print(f"wetext unavailable, tn fields will be null: {e}"[:200])
    en_tn = zh_tn = None
encode = lambda s: tok([s])["input_ids"][0]  # CosyVoice2Tokenizer.encode


TN_RAW = {}  # input passed to the FST normalizer -> its output, so tests can inject the normalizer as a lookup


def normalize(text, use_tn):
    """cosyvoice/cli/frontend.py text_normalize (non-ttsfrd), returning (normalized, segments)."""
    if ('<|' in text and '|>' in text) or text == '':
        return text, [text]
    text = text.strip()
    if contains_chinese(text):
        if use_tn:
            TN_RAW[text] = zh_tn.normalize(text)
            text = TN_RAW[text]
        text = text.replace("\n", "")
        text = replace_blank(text)
        text = replace_corner_mark(text)
        text = text.replace(".", "。").replace(" - ", "，")
        text = remove_bracket(text)
        import re
        text = re.sub(r'[，,、]+$', '。', text)
        segs = list(split_paragraph(text, encode, "zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False))
    else:
        if use_tn:
            TN_RAW[text] = en_tn.normalize(text)
            text = TN_RAW[text]
        text = spell_out_number(text, inf)
        segs = list(split_paragraph(text, encode, "en", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False))
    return text, [s for s in segs if not is_only_punctuation(s)]


numbers = [str(n) for n in [0, 1, 2, 7, 9, 10, 11, 13, 19, 20, 21, 30, 45, 99, 100, 101, 110, 115, 120, 199, 200, 250,
                            999, 1000, 1001, 1010, 1100, 1999, 2000, 2001, 2026, 2100, 9999, 10000, 10001, 12345, 99999,
                            100000, 100001, 123456, 999999, 1000000, 1000001, 1234567, 10000000, 123456789, 1000000000,
                            9876543210]] + ["007", "00", "0001", "000123"]
texts = [
    "Yes.", "Okay, see you.", "The weather is nice today, so we might take a walk.",
    "After the meeting ended, the team gathered in the kitchen to talk about the new product launch, the budget, and the schedule for the coming quarter.",
    "Our water bill arrived late this month, so I called the office to ask why. The clerk said a meter reading was missed in March. She promised to fix the record and send a corrected statement within two weeks. I thanked her, hung up, and wrote the date on my calendar so I would remember to check. Last time this happened it took three phone calls, so I am hoping this one goes faster.",
    "On March 3rd, 2026 at 4:30 PM, 1,250 people paid $99.95 each.",
    "Wait... what?! No -- really; (yes), 'quoted' & \"double\" -- ok?",
    "No no no no no no no no no no no no.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Dr. Smith lives at 221B Baker St. and earns 12% more in 1999.",
    "No final punctuation here", "  Leading and trailing spaces.  ", "He said \"stop.\" Then he left.",
    "Room 101 has 3 chairs, 12 desks and 1000 pens.", "Call 555-0199 before 9 or after 21.",
    "First sentence. Second one! Third one? Fourth; fifth: sixth.",
    "It is 50/50 whether the 2nd or 3rd option wins on 12/25.",
    "今天是2026年3月3日，天气很好。", "我有3个苹果和12个橘子，一共15个水果。",
    "这是一个很长的句子，用来测试分段逻辑是否和参考实现一致。第二句也在这里。第三句话结束了！",
    "Mixed 中文 and English 文本 123.",
    "<|endofprompt|>SSML-like text 42 stays untouched.",
    "Breathe [breath] then laugh [laughter] and <strong>stress</strong> this.",
]
cases = []
for t in texts:
    plain_norm, plain_segs = normalize(t, False)
    tn_norm, tn_segs = normalize(t, True) if en_tn is not None else (None, None)
    cases.append({"text": t, "plain_normalized": plain_norm, "plain_segments": plain_segs,
                  "tn_normalized": tn_norm, "tn_segments": tn_segs,
                  "tn_segment_ids": [encode(s) for s in tn_segs] if tn_segs is not None else None, "plain_segment_ids": [encode(s) for s in plain_segs]})
json.dump({"special_tokens": {t: tok.convert_tokens_to_ids(t) for t in SPECIALS},
           "tn_raw": TN_RAW,
           "numbers": [{"in": n, "out": inf.number_to_words(n)} for n in numbers], "cases": cases},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"wrote {len(numbers)} numbers, {len(cases)} cases -> {OUT}")
