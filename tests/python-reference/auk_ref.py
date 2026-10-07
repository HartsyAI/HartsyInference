#!/usr/bin/env python3
"""Stage-by-stage F32 reference dumps of Tencent AuK / AuK-Flash for the engine's real-weight parity test.

Runs the upstream modules (github.com/Tencent-Hunyuan/AuK) stage by stage instead of ``AukInfer.generate`` so only
one model is resident at a time and every random draw is captured: the VAE posterior eps and the initial ODE noise
are dumped and replayed by ``AukRealWeightParityTests``. Everything runs in F32 with autocast off (production
upstream runs the DiT under bf16 autocast); the thinker runs on the CPU in F32 so no stage is bf16-limited.

Environment: a venv with the upstream package (``pip install -e AuK --no-deps``), torch 2.7, transformers 4.57,
qwen-omni-utils, omegaconf, torchdiffeq, x_transformers, soundfile, librosa. ``--omni`` is a Qwen2.5-Omni-3B snapshot
directory (config, processor and tokenizer files plus the two thinker shards; an index listing only ``thinker.*``
keys avoids shard 3). Run with ``CUDA_VISIBLE_DEVICES`` naming the GPU for the DiT and VAE.

Layout: ``<out>/<case>/<name>.bin`` (raw little-endian F32 or I32) + ``<name>.json`` ({"shape", "dtype"}), with
``meta.json`` per case. Per-variant tensors live under ``<out>/<case>/<variant>/``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import warnings

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

CLONE_TEXT = "The quick brown fox jumps over the lazy dog."
CASES = {
    # Zero-shot clone: instruction + reference audio; both the thinker audio path and the VAE encoder are exercised.
    "clone": dict(
        instruction=f'Say the following with the same voice: "{CLONE_TEXT}"',
        audio=True,
        seconds=4.0,
    ),
    # Text-only instruct TTS with the short description that leaked into the speech in the live Swarm run.
    "instruct": dict(
        instruction='Generate speech based on the following description: "A warm middle-aged man speaking slowly and '
        'calmly.". The content to speak is: "The quick brown fox jumps over the lazy dog.".',
        audio=False,
        seconds=4.5,
    ),
}
FLASH_GRID = [0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0]
SAMPLE_RATE = 24_000
HOP = 480
LATENT = 64


def save(directory: str, name: str, value) -> None:
    os.makedirs(directory, exist_ok=True)
    t = value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)
    if t.dtype in (torch.int64, torch.int32, torch.bool):
        arr, dtype = t.to(torch.int32).numpy(), "i32"
    else:
        arr, dtype = t.to(torch.float32).numpy(), "f32"
    np.ascontiguousarray(arr).tofile(os.path.join(directory, name + ".bin"))
    with open(os.path.join(directory, name + ".json"), "w") as f:
        json.dump({"shape": list(arr.shape), "dtype": dtype}, f)


def git_head(path: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def messages_for(case: dict, audio_path: str | None) -> list:
    content = [{"type": "text", "text": case["instruction"]}]
    if audio_path:
        content.append({"type": "audio", "audio": audio_path})
    else:
        # infer_auk.AukInfer.generate appends the marker for text-only instruct TTS.
        content[0]["text"] += "|<no_prompt_audio>|"
    return [{"role": "user", "content": content}]


@torch.inference_mode()
def run_thinker(omni_dir: str, messages: list, out: str) -> list[torch.Tensor]:
    """Token ids, mel, audio tower rows and the 36 hidden states (HF ``output_hidden_states[1:]``), F32 on CPU."""
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    from auk.model.cfm_edit import CFMEdit

    processor = Qwen2_5OmniProcessor.from_pretrained(omni_dir)
    inputs = CFMEdit.build_cond_inputs([messages], processor)
    save(out, "input_ids", inputs["input_ids"][0])
    if "input_features" in inputs:
        valid = int(inputs["feature_attention_mask"][0].sum())
        save(out, "mel", inputs["input_features"][0, :, :valid])
        save(out, "pcm16k", torch.from_numpy(processor_audio(messages)))

    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(omni_dir, torch_dtype=torch.float32)
    thinker.visual = None
    thinker.eval()
    tower_rows = []
    hook = thinker.audio_tower.register_forward_hook(lambda _m, _i, o: tower_rows.append(o.last_hidden_state))
    outputs = thinker(**inputs, output_hidden_states=True)
    hook.remove()
    if tower_rows:
        save(out, "audio_tower", tower_rows[0])
    hidden = [h[0].float().clone() for h in outputs.hidden_states[1:]]
    for i, h in enumerate(hidden):
        save(out, f"hidden_{i:02d}", h)
    del thinker, outputs
    return hidden


def processor_audio(messages: list) -> np.ndarray:
    from qwen_omni_utils import process_mm_info

    audios, _, _ = process_mm_info([messages], use_audio_in_video=True)
    return np.asarray(audios[0], dtype=np.float32)


def fuse(hidden: list[torch.Tensor], layer_weights: torch.Tensor, layer_scale: torch.Tensor) -> torch.Tensor:
    """CFMEdit.encode_text's fusion, applied to the dumped hidden states."""
    d = hidden[0].shape[-1]
    stacked = torch.stack([F.layer_norm(h, [d]) for h in hidden], dim=0)
    weights = F.softmax(layer_weights.float(), dim=0)
    return (stacked * weights[:, None, None]).sum(dim=0) * layer_scale.float()


def load_vae(config, vae_path: str, device: str):
    from omegaconf import OmegaConf
    from auk.model.vae import load_vae_model
    from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAEConfig

    kwargs = OmegaConf.to_container(config.model.vae.model_init_kwargs, resolve=True)
    vae = load_vae_model(vae_name=config.model.vae.vae_name, vae_cfg=BigVGANFlowVAEConfig.from_dict(kwargs),
                         vae_ckpt=vae_path, map_location="cpu")
    return vae.to(device).float().eval()


@torch.inference_mode()
def encode_reference(vae, pcm24k: torch.Tensor, out: str, device: str, generator: torch.Generator) -> torch.Tensor:
    """BigVGANFlowVAE.encoding_and_normalization with the posterior eps drawn here and dumped."""
    sample = pcm24k.to(device)[None, None, :]
    stats = vae.audio_encoder(sample)
    mean, log_std = stats.chunk(2, 1)
    eps = torch.randn(mean.shape, generator=generator).to(device)
    latents = (mean + eps * torch.exp(log_std)).transpose(1, 2).float()
    latents = (latents - vae.global_mean.float()) / torch.sqrt(vae.global_log_std.float())
    save(out, "vae_eps", eps)
    save(out, "vae_mean", mean)
    save(out, "vae_log_std", log_std)
    save(out, "ref_latent", latents)
    return latents


def build_dit(config, ckpt: str, device: str):
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from auk.model.flux2_edit import Flux2Edit

    arch = OmegaConf.to_container(config.model.arch, resolve=True)
    arch["attn_backend"] = "torch"
    arch["checkpoint_activations"] = False
    dit = Flux2Edit(**arch, latent_dim=config.model.vae.latent_dim)
    state = load_file(ckpt, device="cpu")
    sub = {k[len("transformer."):]: v for k, v in state.items() if k.startswith("transformer.")}
    missing, unexpected = dit.load_state_dict(sub, strict=False)
    missing = [k for k in missing if "inv_freq" not in k]
    if missing or unexpected:
        raise RuntimeError(f"DiT key mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    if "transformer.rotary_embed.inv_freq" in state:
        dit.rotary_embed.inv_freq.copy_(state["transformer.rotary_embed.inv_freq"].float())
    dit = dit.to(device).float().eval()
    return dit, state["layer_weights"].float(), state["layer_scale"].float()


@torch.inference_mode()
def sample(dit, text: torch.Tensor, ref: torch.Tensor | None, y0: torch.Tensor, grid: list[float], cfg: float,
           out: str, device: str) -> torch.Tensor:
    """CFMEdit.sample's Euler loop with injected noise; dumps every velocity and state."""
    text = text[None].to(device)
    c_mask = torch.ones(text.shape[:2], dtype=torch.bool, device=device)
    ref_mask = None if ref is None else torch.ones(ref.shape[:2], dtype=torch.bool, device=device)
    x = y0.to(device)
    save(out, "t_grid", torch.tensor(grid, dtype=torch.float32))
    for i in range(len(grid) - 1):
        t = torch.tensor(grid[i], dtype=torch.float32, device=device)
        if cfg < 1e-5:
            v = dit(x=x, text=text, time=t, mask=None, c_mask=c_mask, ref=ref, ref_mask=ref_mask, cache=True)
        else:
            pred = dit(x=x, text=text, time=t, mask=None, c_mask=c_mask, ref=ref, ref_mask=ref_mask, cfg_infer=True,
                       cache=True)
            v_cond, v_uncond = torch.chunk(pred, 2, dim=0)
            if i == 0:
                save(out, "v_cond_00", v_cond)
                save(out, "v_uncond_00", v_uncond)
            v = v_cond + (v_cond - v_uncond) * cfg
        save(out, f"v_{i:02d}", v)
        x = x + (grid[i + 1] - grid[i]) * v
        save(out, f"x_{i + 1:02d}", x)
    dit.clear_cache()
    return x


def sway_grid(steps: int, coef: float) -> list[float]:
    t = torch.linspace(0, 1, steps + 1, dtype=torch.float32)
    t = t + coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
    return t.tolist()


@torch.inference_mode()
def decode(vae, latent: torch.Tensor, out: str, device: str) -> np.ndarray:
    raw = vae.denormalize(latent.to(device)).permute(0, 2, 1)
    pcm = vae.inference_from_latents(raw).float().cpu().reshape(-1)
    save(out, "pcm_out", pcm)
    sf.write(os.path.join(out, "out.wav"), pcm.numpy(), SAMPLE_RATE)
    return pcm.numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--auk", required=True, help="Folder with auk_base.safetensors and vae.safetensors")
    ap.add_argument("--flash", required=True, help="Folder with auk_flash.safetensors")
    ap.add_argument("--configs", required=True, help="Folder with base/config.yaml and flash/config.yaml")
    ap.add_argument("--omni", required=True, help="Qwen2.5-Omni-3B snapshot folder")
    ap.add_argument("--audio", required=True, help="Reference clip, mono; its 24 kHz length must be a multiple of 480")
    ap.add_argument("--out", required=True)
    ap.add_argument("--upstream", help="Upstream AuK checkout, for recording its commit")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import torchaudio
    from omegaconf import OmegaConf

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    variants = {
        "flash": dict(ckpt=os.path.join(args.flash, "auk_flash.safetensors"),
                      config=OmegaConf.load(os.path.join(args.configs, "flash", "config.yaml")), grid=FLASH_GRID, cfg=0.0),
        "base": dict(ckpt=os.path.join(args.auk, "auk_base.safetensors"),
                     config=OmegaConf.load(os.path.join(args.configs, "base", "config.yaml")), grid=sway_grid(32, -1.0),
                     cfg=2.0),
    }
    vae_path = os.path.join(args.auk, "vae.safetensors")

    wav, sr = torchaudio.load(args.audio)
    wav = wav.mean(0)
    pcm24k = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav[None])[0] if sr != SAMPLE_RATE else wav
    if pcm24k.numel() % HOP:
        raise SystemExit(f"Reference is {pcm24k.numel()} samples at 24 kHz; trim it to a multiple of {HOP} first.")

    for name in args.cases.split(","):
        case = CASES[name]
        out = os.path.join(args.out, name)
        os.makedirs(out, exist_ok=True)
        generator = torch.Generator().manual_seed(args.seed)
        print(f"[{name}] thinker", flush=True)
        hidden = run_thinker(args.omni, messages_for(case, args.audio if case["audio"] else None), out)
        frames = max(1, int(math.ceil(case["seconds"] * SAMPLE_RATE / HOP)))
        meta = dict(case=name, instruction=case["instruction"], seconds=case["seconds"], frames=frames,
                    audio=os.path.abspath(args.audio) if case["audio"] else None, seed=args.seed,
                    upstream=git_head(args.upstream) if args.upstream else None, torch=torch.__version__,
                    precision="f32, autocast off; thinker on CPU", variants={})

        ref = None
        if case["audio"]:
            save(out, "pcm24k", pcm24k)
            vae = load_vae(variants["base"]["config"], vae_path, device)
            ref = encode_reference(vae, pcm24k, out, device, generator)
            del vae
        y0 = torch.randn(1, frames, LATENT, generator=generator)
        save(out, "noise", y0)

        for vname, v in variants.items():
            vout = os.path.join(out, vname)
            print(f"[{name}/{vname}] DiT", flush=True)
            dit, layer_weights, layer_scale = build_dit(v["config"], v["ckpt"], device)
            text = fuse(hidden, layer_weights, layer_scale)
            save(vout, "fused", text)
            save(vout, "coeffs", F.softmax(layer_weights, 0) * layer_scale)
            save(vout, "inv_freq", dit.rotary_embed.inv_freq.float())
            latent = sample(dit, text, ref, y0, v["grid"], v["cfg"], vout, device)
            del dit
            torch.cuda.empty_cache()
            vae = load_vae(v["config"], vae_path, device)
            print(f"[{name}/{vname}] VAE decode", flush=True)
            decode(vae, latent, vout, device)
            del vae
            torch.cuda.empty_cache()
            meta["variants"][vname] = dict(steps=len(v["grid"]) - 1, cfg=v["cfg"], grid=v["grid"])
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[{name}] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
