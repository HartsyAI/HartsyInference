"""Reference fixtures for the ControlFoley audio decoder (VAE decoder -> mel -> BigVGAN-v2 44.1 kHz) built from the
OFFICIAL classes of the ControlFoley repository (lib.autoencoder.vae, lib.bigvgan_v2, lib.mel_converter).

Usage:
  python audio_decoder_reference.py tiny <controlfoley checkout> <output dir>
      Tiny random VAE decoder + tiny BigVGAN (released key layout, weight-norm still present) and the activations of
      every stage -> <output dir>/audio_decoder_tiny.safetensors.
  python audio_decoder_reference.py real <controlfoley checkout> <output dir> <v1-44.pth> <bigvgan_generator.pt> \
          <bigvgan config.json> <example.wav>
      Converts the released decoder weights (decoder.* + data_mean/std, vocoder) to <output dir>/audio_decoder.safetensors
      (keys prefixed "vae." / "voc.") and writes <output dir>/audio_decoder_reference.safetensors with the waveform the
      official code decodes from a fixed random latent and the wav -> mel -> vocode round trip of the example clip.
Needs torch, safetensors, einops, librosa, huggingface_hub.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

mode, repo, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(repo))
out.mkdir(parents=True, exist_ok=True)

from lib.autoencoder.vae import VAE  # noqa: E402
from lib.bigvgan_v2.bigvgan import BigVGAN  # noqa: E402
from lib.bigvgan_v2.env import AttrDict  # noqa: E402
from lib.mel_converter import get_mel_converter  # noqa: E402


def c(t):
    return t.detach().clone().contiguous().float()


def hook(module, store, name):
    def fn(_m, _i, o):
        store[name] = c(o[0])
    module.register_forward_hook(fn)


def vae_taps(vae, store):
    d = vae.decoder
    hook(d.conv_in, store, "vae.conv_in")
    hook(d.mid.block_1, store, "vae.mid1")
    hook(d.mid.attn_1, store, "vae.mid_attn")
    hook(d.mid.block_2, store, "vae.mid2")
    for lvl in range(d.num_layers):
        hook(d.up[lvl].block[d.num_res_blocks], store, f"vae.level{lvl}")
        if hasattr(d.up[lvl], "upsample"):
            hook(d.up[lvl].upsample, store, f"vae.up{lvl}")


def voc_taps(voc, store):
    hook(voc.conv_pre, store, "voc.conv_pre")
    nk = voc.num_kernels
    acc = {}

    def mk(i, j):
        def fn(_m, _i, o):
            acc[(i, j)] = c(o[0])
            if j == nk - 1:
                store[f"voc.stage{i}"] = sum(acc[(i, k)] for k in range(nk)) / nk
        return fn

    for i in range(voc.num_upsamples):
        for j in range(nk):
            voc.resblocks[i * nk + j].register_forward_hook(mk(i, j))
    hook(voc.activation_post, store, "voc.act_post")


def raw_state(vae):
    return {"vae." + k: c(v) for k, v in vae.state_dict().items() if k.startswith("decoder.") or k.startswith("data_")}


@torch.inference_mode()
def run_vae(vae, latent_tc):
    """latent_tc: [T, latent_dim] as handed to FeaturesUtils.decode; returns the mel [1, 128, 2T]."""
    z = latent_tc.unsqueeze(0).transpose(1, 2)
    return vae.decode(z)


def randomize_vocoder(voc, gen):
    with torch.no_grad():
        params = dict(voc.named_parameters())
        for name, p in params.items():
            if name.endswith("original1") or name.endswith(".weight") or name.endswith(".bias"):
                p.copy_(torch.randn(p.shape, generator=gen) * (0.05 if p.ndim > 1 else 0.1))
            if name.endswith(".alpha") or name.endswith(".beta"):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.3)
        for name, p in params.items():
            if name.endswith("original0"):
                v = params[name[:-1] + "1"]
                norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.ndim - 1)))
                p.copy_(norm * (0.7 + 0.6 * torch.rand(norm.shape, generator=gen)))


def legacy_state(voc):
    """Saves weight-norm parameters the way the released checkpoint does (weight_g / weight_v)."""
    sd = {}
    for k, v in voc.state_dict().items():
        k = k.replace(".parametrizations.weight.original0", ".weight_g")
        k = k.replace(".parametrizations.weight.original1", ".weight_v")
        sd[k] = c(v)
    return sd


def tiny():
    torch.manual_seed(1234)
    gen = torch.Generator().manual_seed(7)
    vae = VAE(data_dim=128, embed_dim=4, hidden_dim=8)
    with torch.no_grad():
        for m in vae.modules():
            if m.__class__.__name__ == "MPConv1D":
                m.weight.copy_(torch.randn(m.weight.shape, generator=gen))
        vae.decoder.learnable_gain.fill_(0.3)
    weights = raw_state(vae)
    vae.eval()
    vae.remove_weight_norm()

    h = AttrDict({
        "resblock": "1", "upsample_rates": [8, 4, 2, 2, 2, 2], "upsample_kernel_sizes": [16, 8, 4, 4, 4, 4],
        "upsample_initial_channel": 128, "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]], "use_tanh_at_final": False,
        "use_bias_at_final": False, "activation": "snakebeta", "snake_logscale": True, "num_mels": 128,
    })
    voc = BigVGAN(h, use_cuda_kernel=False)
    randomize_vocoder(voc, gen)
    weights.update({"voc." + k: v for k, v in legacy_state(voc).items()})
    voc.remove_weight_norm()
    voc.eval()

    store = {}
    vae_taps(vae, store)
    voc_taps(voc, store)
    latent = torch.randn(7, 4, generator=gen)
    mel = run_vae(vae, latent)
    store["vae.mel"] = c(mel[0])
    with torch.inference_mode():
        wave = voc(mel)
    store["voc.wave"] = c(wave[0, 0])

    mc = get_mel_converter("44k")
    t = torch.arange(24000) / 44100.0
    clip = (0.6 * torch.sin(2 * np.pi * 440 * t) + 0.3 * torch.randn(24000, generator=gen)).clamp(-1, 1)
    clip[:2000] *= 1.8
    store["mel.input"] = c(clip)
    with torch.inference_mode():
        store["mel.output"] = c(mc(clip.unsqueeze(0))[0])
    store["mel.basis"] = c(mc.mel_basis)
    weights["ref.latent"] = c(latent)
    weights.update({"tap." + k: v for k, v in store.items()})
    save_file(weights, str(out / "audio_decoder_tiny.safetensors"))
    print({k: tuple(v.shape) for k, v in store.items()})


def real():
    vae_ckpt, voc_ckpt, voc_cfg, wav_path = sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
    sd = torch.load(vae_ckpt, map_location="cpu", weights_only=True)
    vae = VAE(data_dim=128, embed_dim=40, hidden_dim=512).eval()
    vae.load_state_dict(sd)
    weights = raw_state(vae)
    vae.remove_weight_norm()
    del vae.encoder

    h = AttrDict(json.loads(Path(voc_cfg).read_text()))
    voc = BigVGAN(h, use_cuda_kernel=False)
    gsd = torch.load(voc_ckpt, map_location="cpu", weights_only=True)["generator"]
    try:
        voc.load_state_dict(gsd)
    except RuntimeError:
        voc.remove_weight_norm()
        voc.load_state_dict(gsd)
    voc.remove_weight_norm()
    voc.eval()
    weights.update({"voc." + k: c(v) for k, v in gsd.items()})
    save_file(weights, str(out / "audio_decoder.safetensors"))

    ref = {}
    gen = torch.Generator().manual_seed(2024)
    latent = torch.randn(40, 40, generator=gen)
    mel = run_vae(vae, latent)
    with torch.inference_mode():
        wave = voc(mel)
    ref["latent"] = c(latent)
    ref["mel"] = c(mel[0])
    ref["wave"] = c(wave[0, 0])

    import librosa
    y, _ = librosa.load(wav_path, sr=44100, mono=True, duration=1.5)
    clip = torch.from_numpy(y).float()
    mc = get_mel_converter("44k")
    with torch.inference_mode():
        rt_mel = mc(clip.unsqueeze(0))
        rt_wave = voc(rt_mel)
    ref["rt_input"] = c(clip)
    ref["rt_mel"] = c(rt_mel[0])
    ref["rt_wave"] = c(rt_wave[0, 0])
    save_file(ref, str(out / "audio_decoder_reference.safetensors"))
    print({k: tuple(v.shape) for k, v in ref.items()})


{"tiny": tiny, "real": real}[mode]()
