"""Reference dump for Silero VAD v6, for SileroVadParityTests.

Uses the shipped ONNX under onnxruntime rather than torch: the file is already vendored beside
this script and the ONNX I/O contract is the one a port has to reproduce anyway.

Contract (from silero-vad's utils_vad.py OnnxWrapper):
    input : float32 [1, 576]  = 64 samples of carried context + 512 new samples
    state : float32 [2, 1, 128] = stacked LSTM (h, c), zero at stream start
    sr    : int64 scalar = 16000
    ->  output : float32 [1, 1] speech probability for the 512-sample chunk
        stateN : the next state, threaded back in

Usage:
    python silerovad_ref.py <silero_vad.onnx> <audio.wav> <out_dir>
"""
import os
import sys
import wave

import numpy as np
import onnxruntime as ort

WINDOW = 512      # samples scored per step (32 ms at 16 kHz)
CONTEXT = 64      # samples of previous audio prepended


def read_wav_mono16k(path):
    with wave.open(path) as w:
        assert w.getframerate() == 16000, f"expected 16 kHz, got {w.getframerate()}"
        assert w.getsampwidth() == 2, "expected 16-bit PCM"
        raw = w.readframes(w.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if w.getnchannels() == 2:
        audio = audio.reshape(-1, 2).mean(1)
    # Silero takes normalized audio, unlike the wake models which take int16 scale.
    return audio / 32768.0


def main():
    model_path, wav_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    session = ort.InferenceSession(model_path)
    audio = read_wav_mono16k(wav_path)

    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros(CONTEXT, dtype=np.float32)
    probs = []
    for start in range(0, len(audio) - WINDOW + 1, WINDOW):
        chunk = audio[start:start + WINDOW]
        frame = np.concatenate([context, chunk])[None, :].astype(np.float32)
        out, state = session.run(None, {
            "input": frame,
            "state": state,
            "sr": np.array(16000, dtype=np.int64),
        })
        probs.append(float(out.ravel()[0]))
        context = chunk[-CONTEXT:]

    samples = audio[:len(probs) * WINDOW].astype(np.float32)
    samples.tofile(os.path.join(out_dir, "silero_input.bin"))
    np.array(probs, dtype=np.float32).tofile(os.path.join(out_dir, "silero_probs.bin"))
    speech = sum(1 for p in probs if p >= 0.5)
    print(f"steps={len(probs)} speech_frames={speech} ({speech / max(1, len(probs)):.1%}) "
          f"min={min(probs):.6f} max={max(probs):.6f}")


if __name__ == "__main__":
    main()
