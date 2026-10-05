"""Regenerates the reference phonemes the espeak parity tests diff against.

Kokoro reads Spanish, French, Hindi, Italian and Brazilian Portuguese through misaki's EspeakG2P, which runs
espeak-ng 1.52 (the data in the espeakng_loader 0.2.4 wheel) through phonemizer. This rewrites the `ref` lists of
kokoro_espeak_parity.json from the sentences in its `corpus` lists, and the reference column of
en_sentence_parity.tsv and en_ipa_parity.tsv (espeak-ng en-us IPA, no ties) from their first columns.

    pip install "misaki[en]" espeakng-loader==0.2.4 phonemizer
    python tools/kokoro/espeak_parity_reference.py tests/HartsyInference.Audio.Phonemizer.Tests/Fixtures
"""
import ctypes
import json
import os
import sys

import espeakng_loader
from misaki.espeak import EspeakG2P


def espeak_ipa(lib, text: str) -> str:
    """espeak-ng's own IPA for text, clause by clause, as espeak_TextToPhonemes returns it."""
    source = ctypes.c_char_p(text.encode())
    pointer = ctypes.pointer(source)
    out = []
    while source.value:
        out.append(lib.espeak_TextToPhonemes(pointer, 1, 0x02).decode())
    return " ".join(out).strip()


def main(fixtures: str) -> None:
    path = os.path.join(fixtures, "kokoro_espeak_parity.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for lang, corpus in data["corpus"].items():
        g2p = EspeakG2P(language=lang)
        data["ref"][lang] = [g2p(text)[0] for text in corpus]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=0)

    lib = ctypes.cdll.LoadLibrary(espeakng_loader.get_library_path())
    lib.espeak_Initialize(0x02, 0, espeakng_loader.get_data_path().encode(), 0)
    lib.espeak_SetVoiceByName(b"en-us")
    lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
    for name in ("en_sentence_parity.tsv", "en_ipa_parity.tsv"):
        tsv = os.path.join(fixtures, name)
        with open(tsv, encoding="utf-8") as f:
            texts = [line.split("\t")[0] for line in f if line.strip()]
        with open(tsv, "w", encoding="utf-8") as f:
            f.write("\n".join(f"{t}\t{espeak_ipa(lib, t)}" for t in texts) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
