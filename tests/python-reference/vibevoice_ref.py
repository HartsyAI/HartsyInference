#!/usr/bin/env python3
"""Official-VibeVoice-1.5B SHORT-target reference generation.

Purpose
-------
Reproduce the "short-prompt instability" that the HartsyInference C# port shows, using
the *reference* Microsoft VibeVoice code path, so we have a ground-truth WAV + metrics to
compare the C# pipeline against. Generates a deliberately SHORT single-speaker utterance
("The speech synthesizer is now working correctly.") — the case the C# port garbles.

This mirrors demo/inference_from_file.py exactly for the parameters that matter:
    * cfg_scale = 1.3           (reference default; the C# port hardcodes 1.0 == no CFG)
    * generation_config = {'do_sample': False}   (greedy + token-constraint processor)
    * set_ddpm_inference_steps(10)               (reference demo default; config.json says 20)
    * is_prefill = True                          (voice cloning on)

Output
------
    /tmp/hartsyinference_tts_to_stt/vibevoice_REF_python.wav   (24 kHz mono)
Prints: generation time, audio duration, RMS.

DO NOT run under `dotnet test` / the IDE — this loads the full 5 GB model. Run standalone
on a GPU box:  python tests/python-reference/vibevoice_ref.py

Assumptions / notes
-------------------
  * Requires the reference `vibevoice` package on PYTHONPATH. Point VIBEVOICE_SRC at the
    checkout if it isn't pip-installed (a scratch checkout lives under
    .../scratchpad/vvcomm at authoring time).
  * Model weights: microsoft/VibeVoice-1.5B. Set VIBEVOICE_MODEL to a local dir (e.g.
    ~/.cache/hartsyinference/models/microsoft--VibeVoice-1.5B) or an HF repo id.
  * A voice-reference WAV is REQUIRED (voice cloning). Set VIBEVOICE_VOICE, else the script
    probes the reference demo `voices/` dir and the HartsyInference test-clips dir.
  * We do NOT set a manual torch seed here: the reference draws fresh torch.randn per
    diffusion frame — that decorrelated per-frame noise is exactly what the C# port's
    deterministic `new Random(step+1)` fails to reproduce. Leaving the RNG free keeps this
    faithful to the demo. Add `torch.manual_seed(...)` only if you want bit-repeatability.
"""

import os
import sys
import glob
import time

import numpy as np
import torch


# ---- Locate the reference package -------------------------------------------------------
_VV_SRC = os.environ.get("VIBEVOICE_SRC")
if _VV_SRC and _VV_SRC not in sys.path:
    sys.path.insert(0, _VV_SRC)
else:
    # Best-effort: the scratch checkout used during porting.
    for cand in glob.glob(os.path.expanduser(
            "/tmp/claude-*/**/scratchpad/vvcomm"), recursive=True):
        if os.path.isdir(os.path.join(cand, "vibevoice")):
            sys.path.insert(0, cand)
            break

try:
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
except ImportError as e:
    sys.exit(
        f"Could not import the reference vibevoice package ({e}).\n"
        "Install it or set VIBEVOICE_SRC=/path/to/VibeVoice checkout."
    )


# ---- Config -----------------------------------------------------------------------------
MODEL = os.environ.get(
    "VIBEVOICE_MODEL",
    os.path.expanduser("~/.cache/hartsyinference/models/microsoft--VibeVoice-1.5B"),
)
CFG_SCALE = float(os.environ.get("VIBEVOICE_CFG_SCALE", "1.3"))
DDPM_STEPS = int(os.environ.get("VIBEVOICE_DDPM_STEPS", "10"))
SHORT_TEXT = "The speech synthesizer is now working correctly."
OUT_DIR = "/tmp/hartsyinference_tts_to_stt"
OUT_PATH = os.path.join(OUT_DIR, "vibevoice_REF_python.wav")
SAMPLE_RATE = 24_000


def _find_voice() -> str:
    env = os.environ.get("VIBEVOICE_VOICE")
    if env and os.path.isfile(env):
        return env
    probes = []
    if _VV_SRC:
        probes.append(os.path.join(_VV_SRC, "demo", "voices"))
    probes += glob.glob(os.path.expanduser("/tmp/claude-*/**/scratchpad/vvcomm/demo/voices"),
                        recursive=True)
    probes.append(os.path.expanduser("~/.cache/hartsyinference/test-clips"))
    for d in probes:
        if not os.path.isdir(d):
            continue
        # Prefer an English single-speaker clip.
        for pat in ("en-*woman*.wav", "en-*man*.wav", "*.wav"):
            hits = sorted(glob.glob(os.path.join(d, pat)))
            if hits:
                return hits[0]
    sys.exit("No voice-reference WAV found. Set VIBEVOICE_VOICE=/path/to/voice.wav")


def main() -> None:
    voice = _find_voice()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Model : {MODEL}")
    print(f"Voice : {voice}")
    print(f"Device: {device}  dtype={dtype}  cfg_scale={CFG_SCALE}  ddpm_steps={DDPM_STEPS}")
    print(f"Text  : {SHORT_TEXT!r}")

    processor = VibeVoiceProcessor.from_pretrained(MODEL)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL,
        torch_dtype=dtype,
        device_map=(device if device in ("cuda", "cpu") else None),
        attn_implementation="sdpa",  # flash_attention_2 needs the extra dep; sdpa is fine here.
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=DDPM_STEPS)

    # Single speaker. The processor's script format is "Speaker 0: <text>".
    full_script = f"Speaker 0: {SHORT_TEXT}"
    inputs = processor(
        text=[full_script],
        voice_samples=[[voice]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=CFG_SCALE,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=True,
            is_prefill=True,
        )
    gen_time = time.time() - t0

    speech = outputs.speech_outputs[0]
    if speech is None:
        sys.exit("No audio produced (model emitted EOS with zero speech frames).")

    audio = speech.detach().to(torch.float32).cpu().numpy().reshape(-1)
    duration = audio.shape[-1] / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0

    os.makedirs(OUT_DIR, exist_ok=True)
    # save_audio handles normalization + WAV writing the same way the demo does.
    try:
        processor.save_audio(speech, output_path=OUT_PATH)
    except Exception:
        # Fallback: write PCM16 directly.
        import wave
        pcm = np.clip(audio, -1.0, 1.0)
        with wave.open(OUT_PATH, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes((pcm * 32767.0).astype("<i2").tobytes())

    print("---- RESULT ----")
    print(f"generation_time_s : {gen_time:.2f}")
    print(f"audio_duration_s  : {duration:.2f}")
    print(f"audio_samples     : {audio.shape[-1]}")
    print(f"rms               : {rms:.4f}")
    print(f"wav               : {OUT_PATH}")


if __name__ == "__main__":
    main()
