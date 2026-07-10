#!/usr/bin/env python3
"""Ground-truth reference generation for NeuTTS Air (neuphonic/neutts-air + neuphonic/neucodec).

Run ON GPU by a human (loads ~750M backbone + neucodec into torch). DO NOT run under `dotnet test`
or alongside other torch models — host-RAM is the gate (see feedback_audio_inference_ram_oom).

Purpose: produce the exact upstream waveform for the sentence our C# port targets, so the C# NeuTTS
pipeline output can be diffed against a known-good reference. Voice cloning is MANDATORY upstream —
`infer()` always conditions on a reference voice; there is no "default voice" mode. This script uses the
official `samples/dave.wav` (+ `samples/dave.txt`) shipped in the neuphonic/neutts-air repo.

It also DUMPS the exact special-token ids the tokenizer assigns (SPEECH_GENERATION_START/END, speech_0,
TEXT_PROMPT_*) so the C# `NeuTtsConfig` constants can be re-verified against the checkpoint that is
actually on disk. As of this writing they were confirmed as:
    <|TEXT_REPLACE|>=151665  <|TEXT_PROMPT_START|>=151666  <|TEXT_PROMPT_END|>=151667
    <|SPEECH_REPLACE|>=151668  <|SPEECH_GENERATION_START|>=151669  <|SPEECH_GENERATION_END|>=151670
    <|speech_0|>=151671 ... <|speech_65535|>=217206   (eos used by generate() = 151670)

Upstream generation (neuttsair/neutts.py `_infer_torch`) — reproduced here for parity, DO NOT ADD a
repetition penalty; upstream uses NONE:
    backbone.generate(prompt, max_length=max_context(=2048), eos_token_id=<|SPEECH_GENERATION_END|>,
                      do_sample=True, temperature=1.0, top_k=50, use_cache=True, min_new_tokens=50)

Usage:
    python neutts_ref.py [--repo /path/to/neutts-air/checkout] [--ref-wav PATH --ref-txt PATH]
                         [--device cuda] [--seed 0]

Requires the official package on PYTHONPATH:  pip install neuttsair  (or clone the repo and run from it).
"""
import argparse
import os
import time
import wave
from pathlib import Path

import numpy as np

TARGET_TEXT = "The speech synthesizer is now working correctly."
OUT_DIR = "/tmp/hartsyinference_tts_to_stt"
OUT_PATH = os.path.join(OUT_DIR, "neutts_REF_python.wav")
SAMPLE_RATE = 24000
SPECIAL_TOKENS = [
    "<|TEXT_REPLACE|>", "<|TEXT_PROMPT_START|>", "<|TEXT_PROMPT_END|>",
    "<|SPEECH_REPLACE|>", "<|SPEECH_GENERATION_START|>", "<|SPEECH_GENERATION_END|>",
    "<|speech_0|>", "<|speech_65535|>",
]


def find_reference(args):
    """Locate the official dave sample (wav + its transcript text)."""
    if args.ref_wav and args.ref_txt:
        return Path(args.ref_wav), Path(args.ref_txt)
    roots = []
    if args.repo:
        roots.append(Path(args.repo))
    roots += [Path.cwd(), Path.cwd().parent, Path(__file__).resolve().parent]
    for root in roots:
        wav = root / "samples" / "dave.wav"
        txt = root / "samples" / "dave.txt"
        if wav.exists() and txt.exists():
            return wav, txt
    raise SystemExit(
        "Could not find samples/dave.wav + samples/dave.txt. Pass --repo <neutts-air checkout> "
        "or --ref-wav/--ref-txt explicitly. (Voice cloning is mandatory upstream.)")


def dump_token_ids(tokenizer):
    print("== tokenizer special-token ids (verify against NeuTtsConfig) ==")
    for tok in SPECIAL_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(tok)
        print(f"  {tok:32s} -> {tid}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("NEUTTS_REPO"))
    ap.add_argument("--ref-wav")
    ap.add_argument("--ref-txt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    try:
        from neuttsair.neutts import NeuTTSAir
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "Could not import neuttsair. Install the official package (pip install neuttsair) or run "
            f"this script from inside a neutts-air checkout. Underlying error: {e}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ref_wav, ref_txt = find_reference(args)
    ref_text = ref_txt.read_text(encoding="utf-8").strip()
    print(f"reference wav : {ref_wav}")
    print(f"reference text: {ref_text!r}")
    print(f"target text   : {TARGET_TEXT!r}")

    tts = NeuTTSAir(
        backbone_repo="neuphonic/neutts-air",
        backbone_device=args.device,
        codec_repo="neuphonic/neucodec",
        codec_device=args.device,
    )

    # Dump the real token ids so the C# constants can be re-verified against the on-disk checkpoint.
    tok = getattr(tts, "tokenizer", None)
    if tok is not None:
        dump_token_ids(tok)

    ref_codes = tts.encode_reference(str(ref_wav))

    t0 = time.time()
    wav = tts.infer(TARGET_TEXT, ref_codes, ref_text)
    elapsed = time.time() - t0

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    duration = len(wav) / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) if wav.size else 0.0
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0

    os.makedirs(OUT_DIR, exist_ok=True)
    pcm16 = np.clip(wav, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2")
    with wave.open(OUT_PATH, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm16.tobytes())

    print()
    print(f"saved     : {OUT_PATH}")
    print(f"samples   : {wav.size}")
    print(f"duration  : {duration:.2f} s @ {SAMPLE_RATE} Hz")
    print(f"gen time  : {elapsed:.2f} s ({elapsed / max(duration, 1e-9):.2f}x realtime)")
    print(f"RMS       : {rms:.5f}")
    print(f"peak      : {peak:.5f}")


if __name__ == "__main__":
    main()
