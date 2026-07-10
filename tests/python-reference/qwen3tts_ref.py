#!/usr/bin/env python3
"""Qwen3-TTS (12 Hz) CustomVoice — Python ground-truth reference.

Purpose
-------
Generate a WAV from the OFFICIAL Qwen3-TTS CustomVoice checkpoint so the pure-C#
HartsyInference engine can be A/B'd against it. Writes:
    /tmp/hartsyinference_tts_to_stt/qwen3tts_REF_python.wav
and prints wall-time, duration, and RMS.

DO NOT RUN ON THE ANALYSIS BOX (it OOMs). Run on a GPU machine.

Why this exists
---------------
The C# end-to-end test drove the CustomVoice checkpoint in VOICE-DESIGN mode
(speaker id = -1, no voice-description instruct text). CustomVoice checkpoints
are trained to ALWAYS receive a built-in speaker id; with no speaker and no
instruct they go out-of-distribution and emit EOS almost immediately -> ~2
frames -> silent output. The correct driver for a *-CustomVoice checkpoint is
CUSTOM-VOICE mode with a real speaker (e.g. "Ryan" / codec token 3061).

Checkpoint facts confirmed on disk (2026-07-09)
-----------------------------------------------
  ~/.cache/hartsyinference/models/Qwen--Qwen3-TTS-12Hz-{0.6B,1.7B}-CustomVoice/
    model.safetensors            -> 402 tensors, ALL under `talker.*`
                                    (talker.model.*, talker.code_predictor.*,
                                     talker.codec_head.*, talker.text_projection.*)
                                    NO `speaker_encoder.*`  (CustomVoice != clone,
                                    no ECAPA x-vector in this checkpoint)
    speech_tokenizer/model.safetensors -> decoder.* (271) + encoder.* (225)
  There is NO config.json, NO tokenizer, NO speaker map on disk. Config is
  implicit (baked into HartsyInference presets). So this script CANNOT read the
  official inference API off local files -- it must pull the official modeling
  code from the HF hub via trust_remote_code, or use a locally-cloned
  QwenLM/Qwen3-TTS repo.

ASSUMPTIONS (verify against the official README before trusting the output)
--------------------------------------------------------------------------
1. `transformers` on the analysis box is 4.27.4 (no native qwen3_tts). This
   script therefore assumes a RECENT transformers (pip install -U transformers)
   OR the official `QwenLM/Qwen3-TTS` package on PYTHONPATH.
2. The public API is one of the two patterns tried below (custom-code AutoModel,
   or the repo's own inference class). If neither matches the release you have,
   fix `synthesize()` to match the official README -- the KEY POINT this
   reference must preserve is: CustomVoice mode + speaker="Ryan", language
   "English", generating the fixed sentence. Do not fall back to voice-design.
3. Speaker "Ryan" (English) == codec token 3061 in the engine. Any of the
   verified built-ins works: Ryan(3061,en) Serena(3066,zh) Ono-Anna(2873,ja)
   Sohee(2864,ko).
"""

import os
import sys
import time
import wave
import struct

# ---- config -----------------------------------------------------------------
REPO = os.environ.get("QWEN3TTS_REPO", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
LOCAL = os.environ.get(
    "QWEN3TTS_LOCAL",
    os.path.expanduser(
        "~/.cache/hartsyinference/models/Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice"
    ),
)
# Prefer the local on-disk checkpoint if present; else the hub id. Custom modeling
# code (trust_remote_code) is still fetched from `REPO` on the hub.
MODEL_ID = LOCAL if os.path.isdir(LOCAL) and os.listdir(LOCAL) else REPO

TEXT = os.environ.get("QWEN3TTS_TEXT", "The speech synthesizer is now working correctly.")
SPEAKER = os.environ.get("QWEN3TTS_SPEAKER", "Ryan")        # CustomVoice built-in
LANGUAGE = os.environ.get("QWEN3TTS_LANG", "English")
OUT_DIR = "/tmp/hartsyinference_tts_to_stt"
OUT_WAV = os.path.join(OUT_DIR, "qwen3tts_REF_python.wav")
SAMPLE_RATE = 24_000                                        # 12.5 Hz frames * 1920


def write_wav_mono16(path, pcm, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    peak = max((abs(float(x)) for x in pcm), default=0.0)
    scale = 32767.0 / peak if peak > 1.0 else 32767.0
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for x in pcm:
            v = int(max(-32768, min(32767, round(float(x) * scale))))
            frames += struct.pack("<h", v)
        w.writeframes(bytes(frames))


def rms(pcm):
    if len(pcm) == 0:
        return 0.0
    s = 0.0
    for x in pcm:
        s += float(x) * float(x)
    return (s / len(pcm)) ** 0.5


def to_float_list(audio):
    """Coerce whatever the model returns (np array / torch tensor / list) to a
    flat python float list in [-1, 1]."""
    try:
        import numpy as np
        import torch
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().to(torch.float32).cpu().numpy()
        if isinstance(audio, np.ndarray):
            return audio.reshape(-1).astype("float32").tolist()
    except Exception:
        pass
    if hasattr(audio, "tolist"):
        flat = audio
        while hasattr(flat, "__len__") and len(flat) and hasattr(flat[0], "__len__"):
            flat = flat[0]
        return [float(x) for x in flat]
    return [float(x) for x in audio]


def synthesize():
    """Load the official CustomVoice model and synthesize TEXT in CUSTOM-VOICE
    mode with a built-in speaker. Returns (sample_rate, pcm_float_list).

    Tries the two most likely official entry points. Adjust to match the exact
    Qwen3-TTS release README if both fail -- but keep it CustomVoice + speaker.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # --- Pattern A: transformers custom-code (AutoProcessor + AutoModel) -------
    # Mirrors the standard Qwen multimodal release shape. The processor is where
    # speaker/language/mode conditioning is injected for CustomVoice.
    try:
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            MODEL_ID, trust_remote_code=True, torch_dtype=dtype
        ).to(device).eval()

        # CustomVoice conditioning. Exact kwarg names are release-specific; the
        # invariant is: pick a built-in speaker (NOT voice-design / clone).
        inputs = processor(
            text=TEXT,
            speaker=SPEAKER,
            language=LANGUAGE,
            mode="custom_voice",
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = model.generate(**inputs)  # waveform or dict/obj carrying it

        audio = getattr(out, "waveform", None)
        if audio is None and isinstance(out, dict):
            audio = out.get("waveform") or out.get("audio")
        if audio is None:
            audio = out
        sr = getattr(processor, "sampling_rate", None) or SAMPLE_RATE
        return int(sr), to_float_list(audio)
    except Exception as e_a:
        print(f"[ref] Pattern A (AutoModel custom-code) failed: {e_a!r}", file=sys.stderr)

    # --- Pattern B: the official QwenLM/Qwen3-TTS package ----------------------
    # `pip install -e .` the cloned QwenLM/Qwen3-TTS, or put it on PYTHONPATH.
    try:
        from qwen_tts import Qwen3TTS  # type: ignore

        tts = Qwen3TTS.from_pretrained(MODEL_ID)  # noqa
        audio = tts.generate(
            text=TEXT, speaker=SPEAKER, language=LANGUAGE, mode="custom_voice"
        )
        sr = getattr(tts, "sample_rate", SAMPLE_RATE)
        return int(sr), to_float_list(audio)
    except Exception as e_b:
        print(f"[ref] Pattern B (qwen_tts package) failed: {e_b!r}", file=sys.stderr)

    raise SystemExit(
        "Could not load Qwen3-TTS via either known API.\n"
        "  - Pattern A needs a recent `transformers` that ships the qwen3_tts\n"
        "    custom code (or trust_remote_code fetch from the hub).\n"
        "  - Pattern B needs the official QwenLM/Qwen3-TTS package importable.\n"
        "Reconcile synthesize() with the official README, keeping CUSTOM-VOICE\n"
        f"mode + speaker={SPEAKER!r} (do NOT use voice-design)."
    )


def main():
    print(f"[ref] model   : {MODEL_ID}")
    print(f"[ref] text    : {TEXT!r}")
    print(f"[ref] mode    : custom_voice  speaker={SPEAKER!r}  lang={LANGUAGE!r}")
    t0 = time.time()
    sr, pcm = synthesize()
    dt = time.time() - t0
    write_wav_mono16(OUT_WAV, pcm, sr)
    dur = len(pcm) / float(sr) if sr else 0.0
    print(f"[ref] wrote   : {OUT_WAV}")
    print(f"[ref] time    : {dt:.2f}s")
    print(f"[ref] samples : {len(pcm)}  ({dur:.2f}s @ {sr} Hz)")
    print(f"[ref] RMS     : {rms(pcm):.6f}")
    if rms(pcm) < 1e-4:
        print("[ref] WARNING: output is (near-)silent -- wrong mode/speaker?", file=sys.stderr)


if __name__ == "__main__":
    main()
