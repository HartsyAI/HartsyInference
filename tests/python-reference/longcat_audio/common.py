"""Shared helpers for the LongCat audio-feature dumpers: raw little-endian F32 files plus a meta.json of shapes.

The C# tests (tests/HartsyInference.Audio.Tests/LongCat/LongCatAudioReferenceTests.cs) read these files from the
directory named by HARTSY_LONGCAT_AUDIO_DIR. Run the scripts on WSL or a pod, never in the Windows venv.
"""
import json
import os

import numpy as np

# Pinned reference: meituan-longcat/LongCat-Video @ 58eb336edc5c8c63389838727e40b8c18434e54e
LONGCAT_COMMIT = "58eb336edc5c8c63389838727e40b8c18434e54e"
OUT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "longcat_audio_reference_tensors")


def write_f32(out_dir, name, array):
    os.makedirs(out_dir, exist_ok=True)
    array = np.ascontiguousarray(np.asarray(array, dtype="<f4"))
    array.tofile(os.path.join(out_dir, name + ".f32"))
    meta = read_meta(out_dir)
    meta[name] = list(array.shape)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)


def read_meta(out_dir):
    path = os.path.join(out_dir, "meta.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def set_meta(out_dir, key, value):
    os.makedirs(out_dir, exist_ok=True)
    meta = read_meta(out_dir)
    meta[key] = value
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)


def load_speech_16k(path):
    """A mono 16 kHz float32 clip. Pre-separated vocals are the intended input; the separator is not run here."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:
        import scipy.signal as ss

        audio = ss.resample_poly(audio, 16000, sr).astype("float32")
    return audio
