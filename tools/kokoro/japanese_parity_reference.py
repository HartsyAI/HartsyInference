"""Regenerates the reference the Kokoro Japanese G2P parity test diffs against.

Kokoro reads Japanese through misaki's JAG2P in its default cutlet mode: fugashi (MeCab) over full UniDic 3.1.0,
misaki's ja_words grouping, num2kana and cutlet's kana table. This rewrites the `ipa` and `tokens` of every entry in
kokoro_japanese_parity.json from its `text`: the phoneme string JAG2P returns and, per MeCab token, the surface,
pron and kana features, char_type and unknown flag fugashi reports. Texts misaki itself fails on are dropped.
With a second argument, the corpus is replaced by that file's non-empty lines first.

    pip install "misaki[ja]" && python -m unidic download
    python tools/kokoro/japanese_parity_reference.py tests/HartsyInference.Audio.Tests/Fixtures [corpus.txt]
"""
import json
import os
import sys

from misaki import ja


def main(fixtures: str, corpus_path: str | None = None) -> None:
    path = os.path.join(fixtures, "kokoro_japanese_parity.json")
    if corpus_path:
        with open(corpus_path, encoding="utf-8") as f:
            texts = [line.rstrip("\n") for line in f if line.strip()]
    else:
        with open(path, encoding="utf-8") as f:
            texts = [entry["text"] for entry in json.load(f)["entries"]]
    g2p = ja.JAG2P()
    tagger = g2p.cutlet.tagger
    entries = []
    for text in texts:
        try:
            ipa = g2p(text)[0]
        except Exception as e:  # noqa: BLE001 - misaki asserts/raises on some inputs; those are not references
            print(f"skipped {text!r}: {type(e).__name__} {e}", file=sys.stderr)
            continue
        tokens = ["\t".join([w.surface, w.feature.pron or "", w.feature.kana or "", str(w.char_type),
                             "1" if w.is_unk else "0"]) for w in tagger(g2p.cutlet._normalize_text(text))]
        entries.append({"text": text, "ipa": ipa, "tokens": tokens})
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"entries": entries}, f, ensure_ascii=False, indent=0)
    print(f"wrote {len(entries)} entries to {path}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
