"""Reference dumps for the wake-word front-end, backbone and heads.

Runs openWakeWord's shipped ONNX graphs under onnxruntime and writes raw float32 fixtures
that WakeBackboneParityTests compares against the C# port.

The audio path deliberately mirrors openWakeWord's STREAMING chunking (1280 new samples with
480 samples of left context per mel call), not a whole-clip mel: the mel graph's dB floor is a
global max over whatever buffer it is handed, so the whole-clip path is a different computation
from the one production takes.

Usage:
    python wake_backbone_ref.py <wake_dir> <out_dir>

where <wake_dir> is the wake model root holding backbone/ and heads/.
"""
import os
import sys

import numpy as np
import onnxruntime as ort

CHUNK = 1280          # 80 ms, one score step
LEFT_CONTEXT = 480    # 160 * 3, what makes 1760 samples yield exactly 8 mel frames
WINDOW = 76           # mel frames per embedding
CONTEXT = 16          # embedding frames per head score


def write(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.tofile(path)
    print(f"  {os.path.basename(path)}: shape={list(arr.shape)} "
          f"min={arr.min():.6f} max={arr.max():.6f} mean={arr.mean():.6f}")


def main():
    wake_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    mel_sess = ort.InferenceSession(os.path.join(wake_dir, "backbone", "melspectrogram.onnx"))
    emb_sess = ort.InferenceSession(os.path.join(wake_dir, "backbone", "embedding_model.onnx"))

    # Deterministic int16-scaled audio. openWakeWord requires 16-bit PCM and converts to float
    # WITHOUT normalizing, so the magnitudes here matter to the log-mel output.
    rng = np.random.RandomState(1234)
    audio = (rng.uniform(-1.0, 1.0, CHUNK + LEFT_CONTEXT) * 8000.0).astype(np.int16).astype(np.float32)
    write(os.path.join(out_dir, "wake_mel_input.bin"), audio)

    mel = np.squeeze(mel_sess.run(None, {"input": audio[None, :]})[0]) / 10.0 + 2.0
    assert mel.shape == (8, 32), f"expected 8 mel frames, got {mel.shape}"
    write(os.path.join(out_dir, "wake_mel_output.bin"), mel)

    # A full embedding window, seeded independently so it does not depend on the mel fixture.
    window = rng.uniform(-1.0, 6.0, (WINDOW, 32)).astype(np.float32)
    write(os.path.join(out_dir, "wake_embedding_input.bin"), window)
    emb = emb_sess.run(None, {"input_1": window[None, :, :, None].astype(np.float32)})[0]
    write(os.path.join(out_dir, "wake_embedding_output.bin"), emb.reshape(-1))

    # One head score per shipped architecture variant: no LayerNorm, LayerNorm, and
    # LayerNorm behind a model.* prefix with a bundled verifier network.
    features = rng.uniform(-8.0, 8.0, (CONTEXT, 96)).astype(np.float32)
    write(os.path.join(out_dir, "wake_head_input.bin"), features)

    scores = []
    for name in ("oww_alexa_v0.1", "oww_hey_mycroft_v0.1", "oww_hey_jarvis_v0.1"):
        path = os.path.join(wake_dir, "heads", name + ".onnx")
        if not os.path.exists(path):
            print(f"  (skipping {name}: not present)")
            continue
        sess = ort.InferenceSession(path)
        out = sess.run(None, {sess.get_inputs()[0].name: features[None, :, :]})[0]
        print(f"  {name}: score={float(out.ravel()[0]):.8f}")
        scores.append(float(out.ravel()[0]))
    write(os.path.join(out_dir, "wake_head_scores.bin"), np.array(scores))

    # Whole-stream reference: every 80 ms score over real speech, driven through the same chunking
    # WakeDetectionPipeline uses. This is what catches ring-buffer and windowing mistakes that the
    # single-shot fixtures above cannot see.
    wav = sys.argv[3] if len(sys.argv) > 3 else None
    if wav and os.path.exists(wav):
        head_path = os.path.join(wake_dir, "heads", "oww_alexa_v0.1.onnx")
        head_sess = ort.InferenceSession(head_path)
        head_input = head_sess.get_inputs()[0].name
        pcm = read_wav_int16(wav)
        write(os.path.join(out_dir, "wake_stream_input.bin"), pcm)
        stream = stream_scores(mel_sess, emb_sess, head_sess, head_input, pcm)
        write(os.path.join(out_dir, "wake_stream_scores.bin"), np.array(stream))
        print(f"  stream steps={len(stream)} max_score={max(stream):.6f}")


def read_wav_int16(path):
    import wave
    with wave.open(path) as w:
        assert w.getnchannels() == 1 and w.getframerate() == 16000 and w.getsampwidth() == 2
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32)


def stream_scores(mel_sess, emb_sess, head_sess, head_input, pcm):
    """openWakeWord's streaming contract, minus the random-audio buffer seeding: scores are withheld
    until 76 real mel frames and 16 real embedding frames exist, matching WakeDetectionPipeline."""
    mel_ring, feat_ring, out = [], [], []
    history = np.zeros(0, dtype=np.float32)
    for start in range(0, len(pcm) - CHUNK + 1, CHUNK):
        chunk = pcm[start:start + CHUNK]
        history = np.concatenate([history[-LEFT_CONTEXT:], chunk]) if len(history) else chunk
        frames = np.squeeze(mel_sess.run(None, {"input": history[None, :]})[0]) / 10.0 + 2.0
        mel_ring.extend(frames)
        if len(mel_ring) < WINDOW:
            continue
        window = np.array(mel_ring[-WINDOW:], dtype=np.float32)
        emb = emb_sess.run(None, {"input_1": window[None, :, :, None]})[0].reshape(-1)
        feat_ring.append(emb)
        if len(feat_ring) < CONTEXT:
            continue
        feats = np.array(feat_ring[-CONTEXT:], dtype=np.float32)
        out.append(float(head_sess.run(None, {head_input: feats[None, :, :]})[0].ravel()[0]))
    return out


if __name__ == "__main__":
    main()
