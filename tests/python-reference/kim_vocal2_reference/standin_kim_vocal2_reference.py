"""Stand-in Kim_Vocal_2 reference for an environment without audio-separator and librosa. Prefer dump_kim_vocal2_reference.py when they
are installed: it drives the real audio-separator 0.30.2.

What is real here: onnxruntime on the real .onnx, torch.stft / torch.istft, and LongCat-Video's own ``_demix_batched`` loaded from its file.
What is a transcription: audio-separator 0.30.2's ``STFT`` class, ``run_model`` and ``spec_utils.normalize`` (MIT, Copyright (c) 2023
karaokenerds). The 16 kHz output is scipy ``resample_poly``, NOT librosa's soxr_hq, so it says nothing about LongCat's resampler.

Needs torch, onnx, onnxruntime, numpy and scipy only. The input must be 44.1 kHz stereo 16-bit PCM WAV.

    python standin_kim_vocal2_reference.py --longcat LongCat-Video --onnx Kim_Vocal_2.onnx --wav clip.wav --out ref_dir
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types
import wave

import numpy as np
import onnx
import onnxruntime as ort
import scipy.signal
import torch

EXPECTED_SHA256 = "ce74ef3b6a6024ce44211a07be9cf8bc6d87728cc852a68ab34eb8e58cde9c8b"


def normalize(wave, max_peak=1.0, min_peak=None):
    """audio_separator.separator.uvr_lib_v5.spec_utils.normalize."""
    maxv = np.abs(wave).max()
    if maxv > max_peak:
        wave *= max_peak / maxv
    elif min_peak is not None and maxv < min_peak:
        wave *= min_peak / maxv
    return wave


def load_longcat_demix(longcat):
    """Loads LongCat's vocal_separator.py by path, supplying only the two imports it names at module level."""
    spec_utils = types.ModuleType("audio_separator.separator.uvr_lib_v5.spec_utils")
    spec_utils.normalize = normalize
    for name in ("librosa", "audio_separator", "audio_separator.separator", "audio_separator.separator.uvr_lib_v5"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["audio_separator.separator.uvr_lib_v5"].spec_utils = spec_utils
    sys.modules[spec_utils.__name__] = spec_utils
    path = os.path.join(longcat, "longcat_video", "audio_process", "vocal_separator.py")
    module_spec = importlib.util.spec_from_file_location("longcat_vocal_separator", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


class Model:
    """The parts of audio-separator's MDXSeparator that LongCat's loop calls, with Kim_Vocal_2's constants."""

    n_fft, dim_f, dim_t, compensate, hop_length = 7680, 3072, 256, 1.009, 1024
    overlap, sample_rate, normalization_threshold = 0.25, 44100, 0.9
    torch_device = torch.device("cpu")

    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.window = torch.hann_window(self.n_fft, periodic=True)
        self.first = None

    def initialize_model_settings(self):
        self.trim = self.n_fft // 2
        self.chunk_size = self.hop_length * (self.dim_t - 1)

    def stft(self, x):
        batch = x.shape[:-2]
        channels, time = x.shape[-2:]
        s = torch.stft(x.reshape([-1, time]), n_fft=self.n_fft, hop_length=self.hop_length, window=self.window,
                       center=True, return_complex=False).permute([0, 3, 1, 2])
        s = s.reshape([*batch, channels, 2, -1, s.shape[-1]]).reshape([*batch, channels * 2, -1, s.shape[-1]])
        return s[..., : self.dim_f, :]

    def istft(self, x):
        batch = x.shape[:-3]
        channels, freq, time = x.shape[-3:]
        bins = self.n_fft // 2 + 1
        x = torch.cat([x, torch.zeros([*batch, channels, bins - freq, time])], -2)
        x = x.reshape([*batch, channels // 2, 2, bins, time]).reshape([-1, 2, bins, time]).permute([0, 2, 3, 1])
        spectrum = x[..., 0] + x[..., 1] * 1.0j
        y = torch.istft(spectrum, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=True)
        return y.reshape([*batch, 2, -1])

    def run_model(self, mix):
        spek = self.stft(mix)
        spek[:, :, :3, :] *= 0
        pred = self.session.run(None, {"input": spek.numpy()})[0]
        out = self.istft(torch.tensor(pred)).numpy()
        if self.first is None:
            self.first = {"wave": mix.numpy()[:1], "spek": spek.numpy()[:1], "pred": pred[:1], "istft": out[:1]}
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--longcat", required=True, help="LongCat-Video checkout at 58eb336edc5c8c63389838727e40b8c18434e54e")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.onnx, "rb") as f:
        data = f.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256:
        sys.exit(f"Kim_Vocal_2.onnx sha256 {digest} is not the pinned {EXPECTED_SHA256}")
    os.makedirs(args.out, exist_ok=True)

    def save(name, array):
        np.save(os.path.join(args.out, name + ".npy"), np.ascontiguousarray(array, dtype=np.float32))

    longcat = load_longcat_demix(args.longcat)
    model = Model(args.onnx)
    with wave.open(args.wav) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (44100, 2, 2):
            sys.exit("the clip must be 44.1 kHz stereo 16-bit PCM")
        pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    mix = (pcm.astype(np.float32) / 32768.0).reshape(-1, 2).T.copy()
    mix = normalize(mix, model.normalization_threshold, -model.normalization_threshold)
    save("mix_normalized", mix)
    source = longcat._demix_batched(model, mix, 2)
    save("stem_compensated", source)
    source = normalize(source, model.normalization_threshold, -model.normalization_threshold)
    save("stem_normalized", source)
    mono = source.mean(axis=0)
    save("mono_44k", mono)
    save("vocals_16k", scipy.signal.resample_poly(mono, 160, 441))
    for key, value in model.first.items():
        save("chunk0_" + key, value)

    # Every Add output of chunk 0: the 5 encoder blocks, the bottleneck, the 5 decoder blocks.
    graph = onnx.load(args.onnx)
    adds = [n.output[0] for n in graph.graph.node if n.op_type == "Add"]
    names = [f"encoding_blocks.{i}" for i in range(5)] + ["bottleneck_block"] + [f"decoding_blocks.{i}" for i in range(5)]
    if len(adds) != len(names):
        sys.exit(f"expected {len(names)} Add nodes, found {len(adds)}")
    for add in adds:
        graph.graph.output.append(onnx.helper.make_tensor_value_info(add, onnx.TensorProto.FLOAT, None))
    staged = ort.InferenceSession(graph.SerializeToString(), providers=["CPUExecutionProvider"])
    for name, value in zip(names, staged.run(adds, {"input": model.first["spek"].astype(np.float32)})):
        save("stage_" + name, value)

    meta = {
        "driver": "stand-in: no audio-separator, no librosa; vocals_16k is scipy resample_poly, not soxr_hq",
        "onnx_sha256": digest,
        "versions": {"onnxruntime": ort.__version__, "onnx": onnx.__version__, "torch": torch.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__},
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
