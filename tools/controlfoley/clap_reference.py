"""Reference activations of ControlFoley's CLAP audio embedding (laion_clap HTSAT audio tower + audio_projection).

  python clap_reference.py tiny <out_dir>
      Builds a tiny random official HTSAT_Swin_Transformer + audio_projection (released key names), runs the official
      get_audio_features / get_audio_embedding path on several waveforms and writes clap_tiny.safetensors + clap_tiny.json.
  python clap_reference.py convert <out.safetensors>
      Streams the audio tower of ext_weights/music_speech_audioset_epoch_15_esc_89.98.pt over HTTP range requests and keeps
      only the tensors the port reads (no 2.3 GB download).
  python clap_reference.py real <weights.safetensors> <reference.wav> <out_dir>
      Runs the official HTSAT-base tower with those weights on a 16 kHz mono clip (resampled with the torchaudio
      sinc_interp_hann algorithm) and writes clap_real_reference.safetensors/.json.

Needs torch, safetensors, numpy, laion_clap (pip install --no-deps laion_clap torchlibrosa; plus h5py ftfy braceexpand
webdataset wget progressbar2 pandas soundfile).
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from laion_clap.clap_module.htsat import HTSAT_Swin_Transformer, create_htsat_model
from laion_clap.clap_module.model import CLAPAudioCfp
from laion_clap.training.data import get_audio_features

MAX_LEN = 480000
SKIP = ("spectrogram_extractor", "logmel_extractor", "tscam_conv", "head.", "relative_position_index", "attn_mask",
        "num_batches_tracked", "audio_transform", "logit_scale", "text", "token_embedding", "positional_embedding",
        "ln_final")


def keep(key: str) -> bool:
    return (key.startswith("audio_branch.") or key.startswith("audio_projection.")) and not any(s in key for s in SKIP)


def audio_cfg(model_name: str, class_num: int) -> CLAPAudioCfp:
    return CLAPAudioCfp(model_type="HTSAT", model_name=model_name, sample_rate=48000, audio_length=1024, window_size=1024,
                        hop_size=480, fmin=50, fmax=14000, class_num=class_num, mel_bins=64, clip_samples=MAX_LEN)


class AudioTower(nn.Module):
    """The two CLAP members get_audio_embedding touches."""

    def __init__(self, branch: nn.Module, features: int, joint: int):
        super().__init__()
        self.audio_branch = branch
        self.audio_projection = nn.Sequential(nn.Linear(features, joint), nn.ReLU(), nn.Linear(joint, joint))

    @torch.no_grad()
    def embed(self, wave: torch.Tensor, cfg: CLAPAudioCfp):
        """Official hook path: get_audio_features (rand_trunc / repeatpad to 10 s at the model's nominal rate) then get_audio_embedding."""
        feats = get_audio_features({}, wave, MAX_LEN, "rand_trunc", "repeatpad", cfg)
        batch = {k: v.unsqueeze(0) for k, v in feats.items()}
        out = self.audio_branch(batch, mixup_lambda=None, device="cpu")["embedding"]
        out = self.audio_projection(out)
        mel = self.audio_branch.logmel_extractor(self.audio_branch.spectrogram_extractor(feats["waveform"][None]))[0, 0]
        return F.normalize(out, dim=-1)[0], mel


def randomize(module: nn.Module, seed: int):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if "spectrogram" in name or "logmel" in name:
                continue
            if name.endswith("norm.weight") or "norm1.weight" in name or "norm2.weight" in name or name.endswith("bn0.weight"):
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            elif "relative_position_bias_table" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
            elif p.dim() >= 2:
                p.copy_(torch.randn(p.shape, generator=g) * (1.2 / math.sqrt(p[0].numel())))
            else:
                p.copy_(0.02 * torch.randn(p.shape, generator=g))
        bn = module.audio_branch.bn0
        bn.running_mean.copy_(torch.randn(bn.running_mean.shape, generator=g) * 5)
        bn.running_var.copy_(torch.rand(bn.running_var.shape, generator=g) * 50 + 10)


def test_wave(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n, dtype=torch.float32) / 16000
    wave = 0.3 * torch.sin(2 * math.pi * (220 + 300 * t) * t) + 0.05 * torch.randn(n, generator=g)
    return wave * (0.5 + 0.5 * torch.sin(2 * math.pi * 2.5 * t) ** 2)


def tiny(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cfg = audio_cfg("tiny", 8)
    branch = HTSAT_Swin_Transformer(spec_size=256, patch_size=4, patch_stride=(4, 4), num_classes=8, embed_dim=16,
                                    depths=[2, 2, 2, 2], num_heads=[2, 2, 4, 4], window_size=8, config=cfg)
    tower = AudioTower(branch, branch.num_features, 24).eval()
    randomize(tower, 7)
    tensors = {k: v.detach().clone().float().contiguous() for k, v in tower.state_dict().items() if keep(k)}
    lengths = [40000, 100001, 31999]
    for i, n in enumerate(lengths):
        wave = test_wave(n, 100 + i)
        emb, mel = tower.embed(wave, cfg)
        tensors[f"in.audio{i}"] = wave
        tensors[f"ref.embedding{i}"] = emb
        tensors[f"ref.logmel{i}"] = mel.contiguous()
    save_file(tensors, os.path.join(out_dir, "clap_tiny.safetensors"))
    with open(os.path.join(out_dir, "clap_tiny.json"), "w") as f:
        json.dump({"embed_dim": 16, "depths": [2, 2, 2, 2], "heads": [2, 2, 4, 4], "joint": 24, "cases": len(lengths)}, f)
    print("wrote", out_dir)


def convert(out_path: str):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import io
    import pickle
    import zipfile

    import remote_pth as rp

    url = "https://huggingface.co/YJX-Xiaomi/ControlFoley/resolve/main/ext_weights/music_speech_audioset_epoch_15_esc_89.98.pt"
    f = rp.RangeFile(url)
    z = zipfile.ZipFile(f)
    prefix = z.namelist()[0].split("/")[0]

    class Stub:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, s):
            pass

    class U(rp._Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except pickle.UnpicklingError:
                return Stub

    obj = U(io.BytesIO(z.read(prefix + "/data.pkl"))).load()
    remote = rp.RemotePth.__new__(rp.RemotePth)
    remote._file, remote._zip, remote._prefix = f, z, prefix
    out = {}
    for key, lazy in obj["state_dict"].items():
        name = key[7:] if key.startswith("module.") else key
        if not keep(name):
            continue
        remote.index = {key: lazy}
        out[name] = torch.from_numpy(remote.read(key)).float().contiguous()
    save_file(out, out_path)
    print("wrote", out_path, len(out), "tensors")


def resample_sinc_hann(x: torch.Tensor, orig: int, new: int, width: int = 6, rolloff: float = 0.99) -> torch.Tensor:
    """torchaudio.functional.resample(sinc_interp_hann) for a 1-D signal."""
    g = math.gcd(orig, new)
    orig, new = orig // g, new // g
    base = min(orig, new) * rolloff
    w = int(math.ceil(width * orig / base))
    idx = torch.arange(-w, w + orig, dtype=torch.float64)[None, None] / orig
    t = (torch.arange(0, -new, -1, dtype=torch.float64)[:, None, None] / new + idx) * base
    t = t.clamp(-width, width)
    window = torch.cos(t * math.pi / width / 2) ** 2
    t = t * math.pi
    scale = base / orig
    kernels = torch.where(t == 0, torch.tensor(1.0, dtype=torch.float64), torch.sin(t) / t) * window * scale
    length = x.shape[-1]
    padded = F.pad(x.double()[None, None], (w, w + orig))
    out = F.conv1d(padded, kernels.float().double(), stride=orig)
    out = out.transpose(1, 2).reshape(-1)
    return out[: int(math.ceil(new * length / orig))].float()


def real(weights_path: str, wav_path: str, out_dir: str):
    import soundfile as sf

    os.makedirs(out_dir, exist_ok=True)
    cfg = audio_cfg("base", 527)
    branch = create_htsat_model(cfg)
    tower = AudioTower(branch, branch.num_features, 512).eval()
    state = load_file(weights_path)
    missing, unexpected = tower.load_state_dict(state, strict=False)
    bad = [k for k in missing if keep(k)]
    assert not bad and not unexpected, (bad, unexpected)
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    wave = torch.from_numpy(data)
    wave16 = resample_sinc_hann(wave, sr, 16000)
    emb, mel = tower.embed(wave16, cfg)
    save_file({"in.audio0": wave16.contiguous(), "ref.embedding0": emb.contiguous(), "ref.logmel0": mel.contiguous()},
              os.path.join(out_dir, "clap_real_reference.safetensors"))
    with open(os.path.join(out_dir, "clap_real_reference.json"), "w") as f:
        json.dump({"samples16k": int(wave16.shape[0]), "cases": 1}, f)
    print("embedding", emb.shape, float(emb.norm()))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "tiny":
        tiny(sys.argv[2])
    elif mode == "convert":
        convert(sys.argv[2])
    elif mode == "real":
        real(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit(__doc__)
