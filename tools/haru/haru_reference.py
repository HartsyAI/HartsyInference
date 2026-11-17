"""Tiny random Haru checkpoint and the upstream (PyTorch, float32, CPU) outputs the engine is compared with.

    python haru_reference.py <path to the upstream repo's haru/ package dir> <out dir> [--seed 7]

Runs the unmodified upstream modules (`haru.models.haru.HaruModel`, the MeanFlow head and the VibeVoice codec) on a
checkpoint small enough to commit, with every tensor random so that no layer is an identity. Nothing is sampled: token ids,
latents, noise and audio are fixed, because the torch generator cannot be matched from C#. Writes:

  haru_tiny.safetensors           weights, upstream key names (lm.*, codec.*, head.*, *_audio_proj.*, latent_scale/bias)
  haru_tiny_expected.safetensors  in.* inputs and ref.* outputs
  config.json                     the config the model was built from
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

CONFIG = {
    "model_type": "haru",
    "dtype": "float32",
    "sample_rate": 24000,
    "frame_samples": 3200,
    "frame_rate": 7.5,
    "latent_dim": 16,
    "pad_token_id": 2000,
    "eop_token_id": 2001,
    "lm": {
        "model_type": "qwen3_5_text",
        "vocab_size": 2048,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "hidden_act": "silu",
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-06,
        "attention_bias": False,
        "attn_output_gate": True,
        "full_attention_interval": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 16,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "partial_rotary_factor": 0.25,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [1, 1, 0],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "tie_word_embeddings": True,
        "eos_token_id": 2002,
    },
    "codec": {
        "latent_dim": 16,
        "base_channels": 1,
        "ratios": [8, 5, 5, 4, 2, 2],
        "encoder_depths": [1, 1, 1, 1, 1, 1, 2],
        "decoder_depths": [2, 1, 1, 1, 1, 1, 1],
        "norm_eps": 1e-05,
        "latent_std": 0.5,
    },
    "head": {
        "hidden_size": 64,
        "latent_dim": 16,
        "num_layers": 2,
        "ffn_ratio": 3.0,
        "norm_eps": 1e-05,
        "interval_embed": 7.0,
        "num_steps": 1,
        "guidance_scale": 1.5,
        "guidance_range": [1.0, 2.0],
    },
}

CARD_TOKENS = 10
REFERENCE_FRAMES = 12
STEPS = 6
CLIP_SAMPLES = 5 * 3200 + 1234  # ends inside a frame
STREAM_FRAMES = 3


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("package", help="directory that contains the upstream `haru` package")
    parser.add_argument("out")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    sys.path.insert(0, args.package)
    from haru.config import HaruConfig
    from haru.models.haru import HaruModel
    from haru.models.vibevoice_codec import CodecStream

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    cfg = HaruConfig.from_dict(CONFIG)
    model = HaruModel(cfg).eval()
    state = model.state_dict()
    for name, value in state.items():
        scale = 0.05 if name == "lm.embed_tokens.weight" else 0.1 if value.dim() <= 1 else value.shape[-1] ** -0.5
        state[name] = torch.randn(value.shape) * scale
        if name.endswith(("gamma", ".norm.weight", "ffn_norm.weight")):
            state[name] += 1.0
    state["latent_scale"] = torch.tensor(0.2)
    state["latent_bias"] = torch.tensor(0.05)
    model.load_state_dict(state)
    model.eval().requires_grad_(False)

    rng = np.random.default_rng(args.seed)
    n_card, n = CARD_TOKENS, CARD_TOKENS + REFERENCE_FRAMES + STEPS
    text = rng.integers(10, 1900, size=n)
    text[n_card : n_card + REFERENCE_FRAMES] = cfg.pad_token_id
    d = cfg.latent_dim
    model_latents = (rng.standard_normal((n, d)) * 2.0).astype(np.float32)
    user_latents = (rng.standard_normal((n, d)) * 2.0).astype(np.float32)
    positions = np.arange(n)
    model_mask = positions >= n_card
    user_mask = positions >= n_card + REFERENCE_FRAMES
    noise = rng.standard_normal((n, d)).astype(np.float32)
    clip = (0.1 * rng.standard_normal(CLIP_SAMPLES)).astype(np.float32)
    frames = (0.1 * rng.standard_normal((1, STREAM_FRAMES * cfg.frame_samples))).astype(np.float32)
    latents = (rng.standard_normal((1, STREAM_FRAMES, d)) * 2.0).astype(np.float32)

    def t(array):
        return torch.as_tensor(array)

    # Backbone: the prompt in one call, then one position at a time.
    cache = model.new_cache()
    prefill = n - STEPS
    spans = [(0, prefill)] + [(i, i + 1) for i in range(prefill, n)]
    hidden = []
    for s, e in spans:
        emb = model.embed(
            t(text[None, s:e]), t(model_latents[None, s:e]), t(user_latents[None, s:e]),
            t(model_mask[None, s:e]), t(user_mask[None, s:e]),
        )
        hidden.append(model(emb, cache)[0])
    hidden = torch.cat(hidden)
    logits = model.logits(hidden[-STEPS:])
    stream_ids = torch.tensor([cfg.pad_token_id, cfg.eop_token_id])
    head_latents = model.unscale_latents(model.head.sample(hidden, t(noise), cfg.head.guidance_scale))

    stream = CodecStream(model.codec)
    expected = {
        "in.text": t(text).to(torch.int32),
        "in.model_latents": t(model_latents),
        "in.user_latents": t(user_latents),
        "in.model_mask": t(model_mask.astype(np.int32)),
        "in.user_mask": t(user_mask.astype(np.int32)),
        "in.noise": t(noise),
        "in.clip": t(clip),
        "in.frames": t(frames),
        "in.latents": t(latents),
        "in.prefill": torch.tensor(prefill, dtype=torch.int32),
        "ref.hidden": hidden.float(),
        "ref.logits": logits.float(),
        "ref.stream_logits": model.logits_fp32(hidden[-STEPS:], stream_ids).float(),
        "ref.head_latents": head_latents.float(),
        "ref.clip_mean": model.codec.encode(t(clip)).float(),
        "ref.clip_audio": model.codec.decode(t(latents)).float(),
        "ref.stream_mean": stream.encode(t(frames)).float(),
        "ref.stream_audio": torch.cat(
            [stream.decode(t(latents[:, i : i + 1])) for i in range(STREAM_FRAMES)], dim=1
        ).float(),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(out / "haru_tiny.safetensors"), metadata={"format": "pt"})
    save_file({k: v.contiguous() for k, v in expected.items()}, str(out / "haru_tiny_expected.safetensors"), metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(CONFIG, indent=2))
    print(f"wrote {out}: {len(model.state_dict())} tensors, {n} positions ({prefill} prefill + {STEPS} steps)")


if __name__ == "__main__":
    main()
