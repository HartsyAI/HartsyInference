"""Reference activations of ControlFoley's timbre encoder: the MusicGen-Style conditioner (MERT-v1-95M features, 8-layer
transformer, batch norm, RVQ with eval_q codebooks, 15x downsample, output projection) run exactly as
FeaturesUtils.encode_audio_with_music_model reaches it (official audiocraft StyleConditioner, imported without the rest of audiocraft).

  python style_reference.py tiny <out_dir>
      Tiny random official StyleConditioner (tiny HubertModel as the MERT stand-in) -> style_tiny.safetensors/.json.
  python style_reference.py convert <out.safetensors> <MERT pytorch_model.bin>
      Streams the self_wav conditioner out of facebook/musicgen-style state_dict.bin over HTTP range requests and merges the
      MERT weights (download m-a-p/MERT-v1-95M/pytorch_model.bin yourself, 377 MB).
  python style_reference.py real <weights.safetensors> <reference.wav> <out_dir>
      Official StyleConditioner with the real weights on the clip prepared as demo.py does (32 kHz, padded to 2 s) ->
      style_real_reference.safetensors/.json.

Environment: CONTROLFOLEY_REPO (default /home/user/xiaomi-research/controlfoley) points at the official checkout; needs torch,
transformers, safetensors, numpy, einops, julius, soundfile. torchaudio is not required.
"""
import importlib
import json
import math
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

REPO = os.environ.get("CONTROLFOLEY_REPO", "/home/user/xiaomi-research/controlfoley")
HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_RATE = 32000
MERT_CONV = dict(conv_kernel=[10, 3, 3, 3, 3, 2, 2], conv_stride=[5, 2, 2, 2, 2, 2, 2])


def import_audiocraft():
    """audiocraft's package __init__ drags in the whole model zoo; load only its conditioner modules, stubbing absent extras."""
    pkg = types.ModuleType("audiocraft")
    pkg.__path__ = [os.path.join(REPO, "lib", "audiocraft", "audiocraft")]
    sys.modules["audiocraft"] = pkg
    sys.path.insert(0, os.path.join(REPO, "lib", "audiocraft"))

    def stub(name):
        mod = types.ModuleType(name)

        def attr(n):
            if n.startswith("__"):
                raise AttributeError(n)
            return type(n, (), {"__init__": lambda self, *a, **k: None})

        mod.__getattr__ = attr
        mod.__path__ = []
        sys.modules[name] = mod

    for _ in range(40):
        try:
            return importlib.import_module("audiocraft.modules.conditioners")
        except ModuleNotFoundError as e:
            stub(e.name)
    raise RuntimeError("could not import audiocraft conditioners")


def hubert_config(hidden, layers, heads, ffn, conv_dim, pos_kernel, pos_groups):
    from transformers import HubertConfig

    return HubertConfig(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads, intermediate_size=ffn,
                        conv_dim=[conv_dim] * 7, conv_bias=False, feat_extract_norm="group", do_stable_layer_norm=False,
                        num_conv_pos_embeddings=pos_kernel, num_conv_pos_embedding_groups=pos_groups,
                        hidden_dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, feat_proj_dropout=0.0,
                        feat_proj_layer_norm=True, layer_norm_eps=1e-5, mask_time_prob=0.0, layerdrop=0.0, **MERT_CONV)


def legacy_hubert_state(model) -> dict:
    """HubertModel state with the pre-5.x weight-norm names the released MERT checkpoint (and the port) use."""
    out = {}
    for k, v in model.state_dict().items():
        k = k.replace("parametrizations.weight.original0", "weight_g").replace("parametrizations.weight.original1", "weight_v")
        if k != "masked_spec_embed":
            out[k] = v.detach().clone().contiguous()
    return out


def load_legacy_hubert(model, state: dict):
    renamed = {}
    for k, v in state.items():
        k = k.replace("weight_g", "parametrizations.weight.original0").replace("weight_v", "parametrizations.weight.original1")
        renamed[k] = v
    missing, unexpected = model.load_state_dict(renamed, strict=False)
    assert not unexpected and all(m == "masked_spec_embed" for m in missing), (missing, unexpected)


def patch_mert(model):
    import transformers

    transformers.AutoModel.from_pretrained = staticmethod(lambda *a, **k: model)


def randomize(module: nn.Module, seed: int, scale: float = 1.0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if "norm" in name and name.endswith("weight"):
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            elif p.dim() >= 2:
                p.copy_(torch.randn(p.shape, generator=g) * (scale / math.sqrt(p[0].numel())))
            elif "weight_g" in name or "original0" in name:
                p.copy_(1 + 0.2 * torch.rand(p.shape, generator=g))
            else:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))


def make_conditioner(ac, mert, dim, output_dim, n_q, bins, ds, transformer_scale):
    cond = ac.StyleConditioner(transformer_scale=transformer_scale, ds_factor=ds, encodec_n_q=4, n_q_out=n_q, eval_q=1,
                               q_dropout=False, bins=bins, batch_norm=True, rvq_threshold_ema_dead_code=0.1,
                               model_name="mert", sample_rate=SAMPLE_RATE, encodec_checkpoint="", length=3.0,
                               output_dim=output_dim, device="cpu", compute_mask=True, use_middle_of_segment=False,
                               ds_rate_compression=640, num_codebooks_lm=4)
    return cond.eval()


def wav_condition(ac, wave: torch.Tensor):
    return ac.WavCondition(wave[None, None].clone(), torch.tensor([wave.shape[-1]]), [SAMPLE_RATE], [None], [0.0])


@torch.no_grad()
def run_stages(ac, cond, wave: torch.Tensor, duration: float, eval_q: int):
    """Official forward plus the intermediates the port is checked against; returns (final, mert, pre_rvq, quantised)."""
    cond.set_params(eval_q=eval_q, excerpt_length=duration)
    x = wav_condition(ac, wave)
    final, mask = cond(x)
    sub = int(wave.shape[-1] - cond.length_subwav)
    assert sub == 0, f"excerpt {cond.length_subwav} != clip {wave.shape[-1]}: the official crop would be random"
    wav24 = ac.convert_audio(x.wav, from_rate=SAMPLE_RATE, to_rate=24000, to_channels=1)
    mert = cond.feat_extractor(wav24.squeeze(-2)).last_hidden_state
    h = cond.embed(mert)
    h = cond.transformer(h)
    h = cond.batch_norm(h.transpose(1, 2)).transpose(1, 2)
    cond.rvq.set_num_codebooks(eval_q)
    q = cond.rvq(h.transpose(1, 2), frame_rate=1.0).x.transpose(1, 2)
    ds = q[:, :: cond.ds_factor]
    manual = cond.output_proj(ds)
    lengths = x.length / cond._downsampling_factor()
    m = (torch.arange(manual.shape[1])[None, :] < lengths[:, None]).float()
    manual = manual * m[..., None]
    assert torch.allclose(manual, final, atol=1e-5), "manual chain disagrees with the official forward"
    return final[0], mert[0], h[0], ds[0]


def style_keys(cond) -> dict:
    out = {}
    for k, v in cond.state_dict().items():
        if "num_batches_tracked" in k or k.endswith("inited") or "cluster_size" in k or "embed_avg" in k:
            continue
        out["style." + k] = v.detach().clone().float().contiguous()
    return out


def tiny(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ac = import_audiocraft()
    from audiocraft.modules.transformer import StreamingTransformer
    from audiocraft.quantization import ResidualVectorQuantizer
    from transformers import HubertModel

    hidden, dim, out_dim, n_q, bins, ds = 40, 32, 24, 3, 16, 3
    mert = HubertModel(hubert_config(hidden, 2, 4, 80, 16, 16, 4)).eval()
    randomize(mert, 3)
    patch_mert(mert)
    cond = make_conditioner(ac, mert, 256, out_dim, n_q, bins, ds, "xsmall")
    cond.embed = nn.Linear(hidden, dim)
    cond.transformer = StreamingTransformer(d_model=dim, num_heads=4, num_layers=2, dim_feedforward=4 * dim, memory_efficient=True,
                                            activation="gelu", norm_first=True, causal=False, layer_scale=None, bias_ff=False,
                                            bias_attn=False)
    cond.rvq = ResidualVectorQuantizer(dim, n_q=n_q, q_dropout=False, bins=bins, threshold_ema_dead_code=0.1)
    cond.batch_norm = nn.BatchNorm1d(dim, affine=False)
    cond.output_proj = nn.Linear(dim, out_dim)
    cond.dim = dim
    cond = cond.eval()
    randomize(cond, 11, 1.5)
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for layer in cond.rvq.vq.layers:
            layer._codebook.embed.copy_(torch.randn(bins, dim, generator=g))
            layer._codebook.inited.fill_(1.0)
        cond.batch_norm.running_mean.copy_(torch.randn(dim, generator=g))
        cond.batch_norm.running_var.copy_(torch.rand(dim, generator=g) + 0.5)

    tensors = {"mert." + k: v for k, v in legacy_hubert_state(mert).items()}
    tensors.update(style_keys(cond))
    lengths = [16000, 24000]
    for i, n in enumerate(lengths):
        t = torch.arange(n, dtype=torch.float32) / SAMPLE_RATE
        wave = 0.4 * torch.sin(2 * math.pi * (300 + 700 * i) * t * (1 + t)) + 0.05 * torch.randn(n, generator=g)
        final, mer, pre, quant = run_stages(ac, cond, wave, n / SAMPLE_RATE, 1)
        tensors.update({f"in.wave{i}": wave, f"ref.mert{i}": mer.contiguous(), f"ref.pre{i}": pre.contiguous(),
                        f"ref.quant{i}": quant.contiguous(), f"ref.tokens{i}": final.contiguous()})
    save_file(tensors, os.path.join(out_dir, "style_tiny.safetensors"))
    with open(os.path.join(out_dir, "style_tiny.json"), "w") as f:
        json.dump({"hidden": hidden, "mert_layers": 2, "mert_heads": 4, "mert_ffn": 80, "mert_conv": 16, "pos_kernel": 16,
                   "pos_groups": 4, "dim": dim, "heads": 4, "layers": 2, "bins": bins, "codebooks": n_q, "ds": ds,
                   "out_dim": out_dim, "eval_q": 1, "cases": len(lengths)}, f)
    print("wrote", out_dir)


def convert(out_path: str, mert_bin: str):
    sys.path.insert(0, HERE)
    import io
    import pickle
    import zipfile

    import remote_pth as rp
    from clap_reference import read_strided

    f = rp.RangeFile("https://huggingface.co/facebook/musicgen-style/resolve/main/state_dict.bin")
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
    head = "condition_provider.conditioners.self_wav."
    out = {}
    for key, lazy in obj["best_state"].items():
        if not key.startswith(head):
            continue
        name = key[len(head):]
        if "num_batches_tracked" in name or name.endswith("inited") or "cluster_size" in name or "embed_avg" in name:
            continue
        out["style." + name] = torch.from_numpy(read_strided(remote, lazy)).float().contiguous()
    for k, v in torch.load(mert_bin, map_location="cpu", weights_only=True).items():
        if k != "masked_spec_embed":
            out["mert." + k] = v.float().contiguous()
    save_file(out, out_path)
    print("wrote", out_path, len(out), "tensors")


def real(weights_path: str, wav_path: str, out_dir: str):
    import soundfile as sf
    from transformers import HubertModel

    sys.path.insert(0, HERE)
    from clap_reference import resample_sinc_hann

    os.makedirs(out_dir, exist_ok=True)
    ac = import_audiocraft()
    state = load_file(weights_path)
    mert = HubertModel(hubert_config(768, 12, 12, 3072, 512, 128, 16)).eval()
    load_legacy_hubert(mert, {k[5:]: v for k, v in state.items() if k.startswith("mert.")})
    patch_mert(mert)
    cond = make_conditioner(ac, mert, 512, 1536, 6, 1024, 15, "default")
    missing, unexpected = cond.load_state_dict({k[6:]: v for k, v in state.items() if k.startswith("style.")}, strict=False)
    assert not unexpected and all(m.endswith(("inited", "cluster_size", "embed_avg", "num_batches_tracked")) for m in missing), (
        missing, unexpected)
    with torch.no_grad():
        for layer in cond.rvq.vq.layers:
            layer._codebook.inited.fill_(1.0)

    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    wave = resample_sinc_hann(torch.from_numpy(data), sr, SAMPLE_RATE)
    min_len, max_len = 2 * SAMPLE_RATE, 4 * SAMPLE_RATE
    if wave.shape[-1] < min_len:
        wave = torch.nn.functional.pad(wave, (0, min_len - wave.shape[-1]))
    wave = wave[:max_len]
    duration = wave.shape[-1] / SAMPLE_RATE
    final, mer, pre, quant = run_stages(ac, cond, wave, duration, 1)
    timbre = final.mean(dim=0)
    save_file({"in.wave0": wave.contiguous(), "ref.mert0": mer.contiguous(), "ref.pre0": pre.contiguous(),
               "ref.quant0": quant.contiguous(), "ref.tokens0": final.contiguous(), "ref.timbre0": timbre.contiguous()},
              os.path.join(out_dir, "style_real_reference.safetensors"))
    with open(os.path.join(out_dir, "style_real_reference.json"), "w") as f:
        json.dump({"samples32k": int(wave.shape[-1]), "duration": duration, "eval_q": 1, "cases": 1}, f)
    print("tokens", tuple(final.shape), "timbre norm", float(timbre.norm()))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "tiny":
        tiny(sys.argv[2])
    elif mode == "convert":
        convert(sys.argv[2], sys.argv[3])
    elif mode == "real":
        real(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit(__doc__)
