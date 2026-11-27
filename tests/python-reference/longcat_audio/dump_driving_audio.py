"""The driving speech as the reference hands it to the audio encoder on Avatar 1.5, with pyloudnorm itself.

run_demo_avatar_single_audio_to_video.py @ 58eb336 asks the separator for clip + 1 s of vocals (so the input is cut there), appends
zeros up to that length, and get_audio_embedding_whisper then loudness-normalises the whole array to -23 LUFS (_loudness_norm, off for
1.6). The separator itself is not run here: its cut is applied to the saved 16 kHz clip. The lines below transcribe the demo's and
_loudness_norm's; the meter and the gain are pyloudnorm's.

Also prints pyloudnorm's loudness of the synthetic signals the C# unit tests hold as known answers.

Usage: dump_driving_audio.py [--out DIR]
Needs numpy and pyloudnorm==0.1.1 (the reference's pin). Input: speech16k.f32 from dump_mel_128.py.
"""
import argparse
import math
import os

import numpy as np
import pyloudnorm as pyln

from common import OUT_DEFAULT, set_meta, write_f32

SR = 16000


def prepare(speech, frames, fps):
    duration = frames / fps + 1.0
    cut = math.ceil(math.ceil(duration * 44100) * SR / 44100)
    speech = speech[:cut]
    added = math.ceil((duration - len(speech) / SR) * SR)
    if added > 0:
        speech = np.append(speech, [0.] * added)
    loudness = pyln.Meter(SR).integrated_loudness(speech)
    if abs(loudness) <= 100:
        speech = pyln.normalize.loudness(speech, loudness, -23)
    return speech, loudness


def tone(seconds, amplitude, hertz):
    n = np.arange(int(round(seconds * SR)))
    return (amplitude * np.sin(2 * np.pi * hertz * n / SR)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    speech = np.fromfile(os.path.join(args.out, "speech16k.f32"), dtype="<f4")
    # 93 frames cut the 11 s clip; 253 frames need more than it holds, so it is padded.
    for name, frames in (("cut", 93), ("pad", 253)):
        prepared, loudness = prepare(speech, frames, 25)
        write_f32(args.out, f"prep_{name}", prepared)
        set_meta(args.out, f"prep_{name}_lufs", float(loudness))
        print(name, "frames", frames, "samples", len(prepared), "input loudness", loudness)

    known = {
        "tone": tone(2.0, 0.25, 440),
        "gated": np.concatenate([tone(1.0, 0.5, 440), np.zeros(SR, np.float32), tone(1.0, 0.01, 1000)]),
        "one block": tone(0.4, 0.1, 1000),
        "uneven": tone(1.23, 0.3, 200),
    }
    meter = pyln.Meter(SR)
    for name, signal in known.items():
        print("known", name, repr(float(meter.integrated_loudness(signal.astype(np.float64)))))


if __name__ == "__main__":
    main()
