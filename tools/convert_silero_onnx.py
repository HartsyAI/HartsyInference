"""Convert silero_vad.onnx's 16 kHz branch into the safetensors file SileroVad loads.

The engine has no ONNX graph executor — every forward pass is reimplemented in C# against IBackend — so an
ONNX file is only ever a weight container here. Silero's is an awkward one: its fifteen tensors are not graph
initializers but anonymous Constant nodes buried inside the `then_branch` subgraph of an `If` that switches on
sample rate, so OnnxWeightLoader cannot see them and there is nothing to name them by.

They are recovered by shape. Fourteen of the fifteen shapes are unique within the branch, which pins those
unambiguously; the two [512, 128] LSTM matrices and the two [512] LSTM biases are distinguished by the order
they appear in, which is PyTorch's (input-hidden before hidden-hidden). That last assumption is not taken on
faith — `--verify` runs the exported weights against onnxruntime through the same contract the C# port uses,
and a swapped pair fails it immediately and obviously.

    pip install onnx onnxruntime numpy safetensors
    python tools/convert_silero_onnx.py silero_vad.onnx Models/audio/wake/vad/silero_vad_16k.safetensors \
        --verify tests/python-reference/silerovad_reference/jfk.wav

Source the ONNX from silero-vad directly (https://github.com/snakers4/silero-vad, MIT), not from
openWakeWord's release, which ships a pinned old revision — see docs/Research/WAKE_WORD_DETECTION.md.
"""
import argparse
import sys
import wave

import numpy as np
import onnx
from onnx import numpy_helper

# Names SileroVad.LoadWeights looks up, in the order their shapes appear in the branch. Shapes are from
# docs/Research/WAKE_WORD_DETECTION.md and double as the assertion that this is the model we think it is.
EXPECTED = [
    ("stft_conv.weight", (258, 1, 256)),
    ("conv1.weight", (128, 129, 3)),
    ("conv1.bias", (128,)),
    ("conv2.weight", (64, 128, 3)),
    ("conv2.bias", (64,)),
    ("conv3.weight", (64, 64, 3)),
    ("conv3.bias", (64,)),
    ("conv4.weight", (128, 64, 3)),
    ("conv4.bias", (128,)),
    ("lstm_cell.weight_ih", (512, 128)),
    ("lstm_cell.weight_hh", (512, 128)),
    ("lstm_cell.bias_ih", (512,)),
    ("lstm_cell.bias_hh", (512,)),
    ("final_conv.weight", (1, 128, 1)),
    ("final_conv.bias", (1,)),
]

WINDOW = 512
CONTEXT = 64


def sixteen_k_branch(model):
    """The `then_branch` of the sample-rate `If` — the 16 kHz path."""
    for node in model.graph.node:
        if node.op_type != "If":
            continue
        for attr in node.attribute:
            if attr.name == "then_branch" and attr.g.ByteSize() > 0:
                return attr.g
    raise SystemExit("no If/then_branch in this ONNX; is it really silero_vad.onnx?")


def constants_in_order(graph):
    """Every Constant tensor in the branch, in graph order, skipping scalars and tiny shape helpers."""
    found = []
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            array = numpy_helper.to_array(attr.t)
            if array.dtype != np.float32 or array.size <= 1 and array.shape != (1,):
                continue
            found.append(array)
    return found


def extract(onnx_path):
    model = onnx.load(onnx_path)
    branch = sixteen_k_branch(model)
    candidates = constants_in_order(branch)

    wanted = [shape for _, shape in EXPECTED]
    picked, used = {}, set()
    for name, shape in EXPECTED:
        match = None
        for i, array in enumerate(candidates):
            if i in used or array.shape != shape:
                continue
            match = i
            break
        if match is None:
            raise SystemExit(f"no {shape} tensor left for {name}; this is not the expected architecture")
        used.add(match)
        picked[name] = np.ascontiguousarray(candidates[match])

    duplicated = [s for s in wanted if wanted.count(s) > 1]
    if duplicated:
        print(f"note: {len(set(duplicated))} shape(s) appear more than once and were assigned in graph order "
              f"({sorted(set(duplicated))}) — run --verify to confirm the assignment.", file=sys.stderr)
    return picked


def read_wav_mono16k(path):
    with wave.open(path) as w:
        if w.getframerate() != 16000 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: need 16 kHz 16-bit, got {w.getframerate()} Hz {w.getsampwidth() * 8}-bit")
        raw = w.readframes(w.getnframes())
        channels = w.getnchannels()
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if channels == 2:
        audio = audio.reshape(-1, 2).mean(1)
    return audio / 32768.0


def onnx_probabilities(onnx_path, audio):
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros(CONTEXT, dtype=np.float32)
    probs = []
    for start in range(0, len(audio) - WINDOW + 1, WINDOW):
        chunk = audio[start:start + WINDOW]
        frame = np.concatenate([context, chunk])[None, :].astype(np.float32)
        out, state = session.run(None, {"input": frame, "state": state,
                                        "sr": np.array(16000, dtype=np.int64)})
        probs.append(float(out.ravel()[0]))
        context = chunk[-CONTEXT:]
    return np.array(probs, dtype=np.float32)


def numpy_reference(weights, audio):
    """The architecture as documented, in numpy, so the exported tensors can be checked without the C# port.

    Right-only reflect pad of 64, fixed-DFT Conv1d STFT (kernel 256 / stride 128), magnitude, four-conv
    encoder, ReLU on the LSTM hidden state, LSTM cell, final conv, sigmoid.
    """
    def conv1d(x, w, b, stride=1, pad=0):
        if pad:
            x = np.pad(x, ((0, 0), (pad, pad)))
        cout, cin, k = w.shape
        t = (x.shape[1] - k) // stride + 1
        cols = np.stack([x[:, i * stride:i * stride + k] for i in range(t)], axis=-1)  # [cin, k, t]
        return np.einsum("okc,kct->ot", w.reshape(cout, cin, k), cols.reshape(cin, k, t)) + b[:, None]

    state_h = np.zeros(128, dtype=np.float32)
    state_c = np.zeros(128, dtype=np.float32)
    context = np.zeros(CONTEXT, dtype=np.float32)
    probs = []
    stft_w = weights["stft_conv.weight"]
    for start in range(0, len(audio) - WINDOW + 1, WINDOW):
        chunk = audio[start:start + WINDOW]
        frame = np.concatenate([context, chunk]).astype(np.float32)
        padded = np.concatenate([frame, frame[-2:-2 - 64:-1]])  # right-only reflect pad of 64
        spec = conv1d(padded[None, :], stft_w, np.zeros(stft_w.shape[0], dtype=np.float32), stride=128)
        half = spec.shape[0] // 2
        mag = np.sqrt(spec[:half] ** 2 + spec[half:] ** 2)
        x = mag
        for i in (1, 2, 3, 4):
            stride = 2 if i in (2, 3) else 1
            x = conv1d(x, weights[f"conv{i}.weight"], weights[f"conv{i}.bias"], stride=stride, pad=1)
            x = x * (x > 0)
        x = x.mean(axis=1)
        gates = weights["lstm_cell.weight_ih"] @ x + weights["lstm_cell.bias_ih"] \
            + weights["lstm_cell.weight_hh"] @ state_h + weights["lstm_cell.bias_hh"]
        i_g, f_g, g_g, o_g = np.split(gates, 4)
        sig = lambda v: 1.0 / (1.0 + np.exp(-v))
        state_c = sig(f_g) * state_c + sig(i_g) * np.tanh(g_g)
        state_h = sig(o_g) * np.tanh(state_c)
        relu_h = state_h * (state_h > 0)
        logit = float(weights["final_conv.weight"].reshape(-1) @ relu_h + weights["final_conv.bias"][0])
        probs.append(float(sig(logit)))
        context = chunk[-CONTEXT:]
    return np.array(probs, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("out")
    ap.add_argument("--verify", metavar="WAV", help="16 kHz mono WAV to check the export against onnxruntime")
    args = ap.parse_args()

    weights = extract(args.onnx)
    for name, shape in EXPECTED:
        print(f"  {name:26s} {tuple(weights[name].shape)}")

    if args.verify:
        audio = read_wav_mono16k(args.verify)
        expected = onnx_probabilities(args.onnx, audio)
        actual = numpy_reference(weights, audio)
        n = min(len(expected), len(actual))
        worst = float(np.max(np.abs(expected[:n] - actual[:n])))
        print(f"verify: {n} chunks, max abs diff {worst:.3e}")
        if worst > 1e-3:
            raise SystemExit(
                f"export does not reproduce onnxruntime (max abs diff {worst:.3e}). The two same-shaped LSTM "
                f"pairs are the likely culprit — they are assigned by graph order.")
        print("verify: OK")

    from safetensors.numpy import save_file
    save_file(weights, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
