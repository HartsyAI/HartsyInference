"""Regenerates the reference the Kokoro Mandarin parity tests diff against.

Kokoro-82M v1.0 reads Mandarin through misaki's legacy ZHG2P: cn2an 0.5.24, jieba 0.42.1 and pypinyin 0.55.0. This
rewrites kokoro_zh_parity.json from the sentences in its `corpus` list:

- `ref`: ZHG2P()(text)[0] per sentence;
- `jieba`: the jieba.lcut words of each sentence's Han runs (after cn2an and punctuation mapping), space-joined;
- `pinyin`: lazy_pinyin(word, TONE3, neutral_tone_with_five=True) over those words, space-joined;
- `syllables`: every first reading in pypinyin's dictionaries (CJK basic block) -> [TONE3, ZHG2P.py2ipa].

    pip install "misaki[zh]==0.9.4"
    python tools/kokoro/mandarin_parity_reference.py tests/HartsyInference.Audio.Tests/Fixtures/kokoro_zh_parity.json
"""
import json
import re
import sys

import cn2an
import jieba
from misaki.zh import ZHG2P
from pypinyin import Style, lazy_pinyin
from pypinyin.constants import PHRASES_DICT, PINYIN_DICT
from pypinyin.converter import UltimateConverter


def han_words(text: str) -> list:
    text = ZHG2P.map_punctuation(cn2an.transform(text, "an2cn"))
    return [w for run in re.findall("[一-鿿]+", text) for w in jieba.lcut(run, cut_all=False)]


def syllables() -> dict:
    raw = {v.split(",")[0] for k, v in PINYIN_DICT.items() if 0x4E00 <= k <= 0x9FFF}
    raw |= {item[0] for v in PHRASES_DICT.values() for item in v}
    conv = UltimateConverter(neutral_tone_with_five=True)
    out = {}
    for r in sorted(raw):
        tone3 = conv.convert_style("", r, Style.TONE3, True)
        out[r] = [tone3, ZHG2P.py2ipa(tone3)]
    return out


def main(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    g2p = ZHG2P()
    corpus = data["corpus"]
    data["ref"] = [g2p(text)[0] for text in corpus]
    words = [han_words(text) for text in corpus]
    data["jieba"] = [" ".join(ws) for ws in words]
    data["pinyin"] = [" ".join(" ".join(lazy_pinyin(w, style=Style.TONE3, neutral_tone_with_five=True)) for w in ws)
                      for ws in words]
    data["syllables"] = syllables()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=0)


if __name__ == "__main__":
    main(sys.argv[1])
