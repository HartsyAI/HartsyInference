"""Writes the reference the Kokoro prosody parity test diffs against.

Runs the official hexgrad `kokoro` KModel (CPU) with the `af_heart` voice on a few sentences phonemized by the
official misaki G2P, and records, per sentence: the phoneme string, the predicted per-phoneme durations and the
F0/energy curves the predictor hands the decoder.

    pip install torch kokoro "misaki[en]"
    python tools/kokoro/prosody_reference.py kokoro_prosody_ref.json
    REF_KOKORO_PROSODY=kokoro_prosody_ref.json dotnet test --filter FullyQualifiedName~KokoroProsodyParity
"""
import json
import sys

import torch
from kokoro import KModel, KPipeline

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello! I'm Kokoro, a small text to speech model, and today I'm running inside HartsyInference.",
    "In 1990, the U.S. had about 1,000 GPUs costing $5.50 each.",
]


def main(out_path: str) -> None:
    model = KModel(repo_id="hexgrad/Kokoro-82M").eval()
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False)
    pack = pipeline.load_voice("af_heart")
    captured = {}
    f0ntrain = model.predictor.F0Ntrain

    def capture(x, s):
        f0, n = f0ntrain(x, s)
        captured["f0"], captured["n"] = f0.numpy().ravel().tolist(), n.numpy().ravel().tolist()
        return f0, n

    model.predictor.F0Ntrain = capture
    rows = []
    for text in SENTENCES:
        _, tokens = pipeline.g2p(text)
        ps = "".join(t.phonemes + (" " if t.whitespace else "") for t in tokens).strip()
        with torch.no_grad():
            out = model(ps, pack[len(ps) - 1], 1.0, return_output=True)
        rows.append({"text": text, "phonemes": ps, "durations": out.pred_dur.tolist(),
                     "f0": captured["f0"], "n": captured["n"]})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "kokoro_prosody_ref.json")
