"""Dump F5-TTS's upstream reference preparation as a parity fixture for the engine port.

Runs f5_tts.infer.utils_infer.preprocess_ref_audio_text (pydub split_on_silence in three stages, the 12 s clip, and
remove_silence_edges with its -42 dBFS threshold, then 50 ms of trailing silence) over a fixed set of 16-bit signals,
and records for each the processed length, the SHA-1 of the processed 16-bit samples, and the RMS that
infer_batch_process uses to scale the reference and the output. The engine's F5 reference tests assert equality.

Signals are the spec format of zipvoice_frontend_dump.py (blocks of 10 ms: S silent, L loud, Q quiet, T and U just
under and over the -50 dBFS threshold) plus one real 24 kHz speech clip, copied into the test fixtures.

Usage (reference venv with f5_tts, pydub and ffmpeg):
    CUDA_VISIBLE_DEVICES= python tools/parity/f5_reference_dump.py \
        --clip /mnt/model-storage/scratch/tts-ref/sweep/kokoro_ref.wav \
        --out tests/HartsyInference.Audio.Tests/Fixtures/f5_reference_parity.json \
        --clip-out tests/HartsyInference.Audio.Tests/Fixtures/f5_reference_speech_24k.wav
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import wave

import numpy as np
from f5_tts.infer.utils_infer import preprocess_ref_audio_text

RATE = 24000
BLOCK = RATE // 100

CASES = [
    ("speech_like", "L 120 S 150 L 40 S 20 L 100 S 60 L 30"),
    ("long_over_12s", "L 300 S 100 L 300 S 100 L 300 S 100 L 300 S 100 L 200"),
    ("short_internal_pause", "L 200 S 15 L 200"),
    ("edge_silence_only", "S 80 L 100 S 60"),
    ("quiet_speech", "S 30 Q 200 S 20 Q 150 S 30"),
    ("threshold_blocks", "S 40 T 120 U 30 S 40 L 60"),
    ("all_silent", "S 300"),
    ("mixed_blocks_rms", "S 25 V 150 S 25 L 20"),
]


def build(spec):
    out = []
    for n, tok in enumerate(spec.split(" ")):
        pass
    samples = []
    tokens = spec.split(" ")
    total = 0
    for i in range(0, len(tokens), 2):
        kind, count = tokens[i], int(tokens[i + 1])
        size = count * BLOCK
        for k in range(size):
            g = total + k
            if kind == "L":
                v = (g * 7919) % 2001 + 2000
            elif kind == "Q":
                v = (g * 7919) % 201 - 100
            elif kind == "S":
                v = (g * 131) % 61 - 30
            elif kind == "T":
                v = 103
            elif kind == "U":
                v = 104
            elif kind == "V":
                v = 104 if k % BLOCK < 192 else 103
            else:
                raise ValueError(kind)
            samples.append(v)
        total += size
    return np.array(samples, dtype=np.int16)


def write_wav(path, pcm, rate=RATE):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def run_upstream(wav_path):
    tmp_text = "placeholder reference text."
    ref_audio, _ = preprocess_ref_audio_text(wav_path, tmp_text, show_info=lambda *_: None)
    with wave.open(ref_audio, "rb") as w:
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16)
    return pcm, rate


def case_entry(name, pcm_in, tmpdir):
    path = os.path.join(tmpdir, f"{name}.wav")
    write_wav(path, pcm_in)
    out, rate = run_upstream(path)
    wave_f = out.astype(np.float64) / 32768.0
    rms = float(np.sqrt(np.mean(np.square(wave_f)))) if len(wave_f) else 0.0
    return {
        "name": name,
        "in_len": int(len(pcm_in)),
        "rate": int(rate),
        "out_len": int(len(out)),
        "out_sha1": hashlib.sha1(out.tobytes()).hexdigest(),
        "rms": rms,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip-out", required=True)
    args = ap.parse_args()
    cases = []
    tmpdir = tempfile.mkdtemp(prefix="f5ref_")
    try:
        for name, spec in CASES:
            pcm = build(spec)
            cases.append({**case_entry(name, pcm, tmpdir), "spec": spec})
        with wave.open(args.clip, "rb") as w:
            assert w.getframerate() == RATE and w.getsampwidth() == 2 and w.getnchannels() == 1
            clip = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        shutil.copyfile(args.clip, args.clip_out)
        cases.append({**case_entry("speech_clip", clip, tmpdir), "spec": "file:" + os.path.basename(args.clip_out)})
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    with open(args.out, "w") as f:
        json.dump({"source": "f5_tts utils_infer.preprocess_ref_audio_text", "rate": RATE, "cases": cases}, f, indent=1)
    for c in cases:
        print(f"{c['name']:24} in={c['in_len']:7} out={c['out_len']:7} rms={c['rms']:.5f}")


if __name__ == "__main__":
    main()
