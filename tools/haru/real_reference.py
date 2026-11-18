"""Upstream Haru-4B (PyTorch, float32 compute, CPU) on the released checkpoint: the numbers the engine is compared with.

    python real_reference.py <upstream haru/ package dir> <dir with model.safetensors, config.json, tokenizer.json> <user.wav> <voice.wav> <out dir>
        [--frames 16]

The 10.5 GB BF16 checkpoint does not fit as float32 next to the engine, so the big language-model matrices stay BF16 in memory and are
widened per call (the same BF16 weights widened to F32, which is what the engine's CPU backend does); everything else is float32 from the
start. Stage 1 runs fixed token ids and latents through the model (hidden states, logits, the F32 PAD/EOP logits, the head). Stage 2 runs
`DuplexSession` greedily on a real user recording, a real voice clip and a character card, with `torch.randn` replaced by a seeded numpy
generator whose draws are recorded for the C# session to replay. Writes real_lm_expected.safetensors, real_session_expected.safetensors and
real_session_expected.json into the out dir.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

parser = argparse.ArgumentParser()
parser.add_argument("package")
parser.add_argument("model_dir")
parser.add_argument("user_wav")
parser.add_argument("voice_wav")
parser.add_argument("out")
parser.add_argument("--frames", type=int, default=16, help="user frames to step in stage 2")
args = parser.parse_args()
sys.path.insert(0, args.package)
model_dir = Path(args.model_dir)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

from haru.config import HaruConfig  # noqa: E402
from haru.models.haru import HaruModel  # noqa: E402
from haru.models.vibevoice_codec import CodecStream  # noqa: E402
import haru.session as session_module  # noqa: E402
from haru.audio import prepare_reference  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

torch.set_num_threads(max(1, (torch.get_num_threads())))


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---- widen BF16 matrices per call -------------------------------------------------------------------------------------------------
def _linear(self, x):
    weight = self.weight if self.weight.dtype == torch.float32 else self.weight.float()
    return F.linear(x.float(), weight, self.bias)


def _embedding(self, ids):
    return F.embedding(ids, self.weight).float()


torch.nn.Linear.forward = _linear
torch.nn.Embedding.forward = _embedding


def _logits(self, hidden):
    return F.linear(hidden.float(), self.lm.embed_tokens.weight.float())


def _logits_fp32(self, hidden, tokens):
    return hidden.float() @ self.lm.embed_tokens.weight[tokens].float().T


HaruModel.logits = _logits
HaruModel.logits_fp32 = _logits_fp32

log("building the model (BF16 parameters)")
config = HaruConfig.from_file(model_dir / "config.json")
torch.set_default_dtype(torch.bfloat16)
from transformers.initialization import no_init_weights  # noqa: E402

with no_init_weights():
    model = HaruModel(config)
torch.set_default_dtype(torch.float32)
params = dict(model.named_parameters())
buffers = dict(model.named_buffers())
with safe_open(str(model_dir / "model.safetensors"), framework="pt") as f:
    for name in f.keys():
        tensor = f.get_tensor(name)
        target = params.get(name, buffers.get(name))
        if target is None:
            raise KeyError(name)
        with torch.no_grad():
            target.copy_(tensor)
for name, p in model.named_parameters():
    big_lm = name.startswith("lm.") and p.dim() >= 2 and "conv1d" not in name
    if not big_lm:
        p.data = p.data.float()
for name, b in model.named_buffers():
    if b.is_floating_point():
        b.data = b.data.float()
model.eval().requires_grad_(False)
model.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
log("model ready")

rng = np.random.default_rng(5)

# ---- stage 1: fixed inputs ----------------------------------------------------------------------------------------------------------
CARD, REFERENCE, STEPS = 6, 8, 4
n = CARD + REFERENCE + STEPS
d = config.latent_dim
text = rng.integers(1000, 100000, size=n)
text[CARD : CARD + REFERENCE] = config.pad_token_id
model_latents = (rng.standard_normal((n, d)) * 2.0).astype(np.float32)
user_latents = (rng.standard_normal((n, d)) * 2.0).astype(np.float32)
positions = np.arange(n)
model_mask = positions >= CARD
user_mask = positions >= CARD + REFERENCE
noise = rng.standard_normal((n, d)).astype(np.float32)


def t(array):
    return torch.as_tensor(array)


with torch.no_grad():
    cache = model.new_cache()
    prefill = n - STEPS
    hidden = []
    for s, e in [(0, prefill)] + [(i, i + 1) for i in range(prefill, n)]:
        log(f"stage 1 positions {s}:{e}")
        emb = model.embed(t(text[None, s:e]), t(model_latents[None, s:e]), t(user_latents[None, s:e]), t(model_mask[None, s:e]), t(user_mask[None, s:e]))
        hidden.append(model(emb, cache)[0])
    hidden = torch.cat(hidden)
    logits = model.logits(hidden[-STEPS:])
    stream_ids = torch.tensor([config.pad_token_id, config.eop_token_id])
    head_latents = model.unscale_latents(model.head.sample(hidden, t(noise), config.head.guidance_scale))
save_file({
    "in.text": t(text).to(torch.int32), "in.model_latents": t(model_latents), "in.user_latents": t(user_latents),
    "in.model_mask": t(model_mask.astype(np.int32)), "in.user_mask": t(user_mask.astype(np.int32)), "in.noise": t(noise),
    "in.prefill": torch.tensor(prefill, dtype=torch.int32),
    "ref.hidden": hidden.float().contiguous(), "ref.logits": logits.float().contiguous(),
    "ref.stream_logits": model.logits_fp32(hidden[-STEPS:], stream_ids).float().contiguous(),
    "ref.head_latents": head_latents.float().contiguous(),
}, str(out / "real_lm_expected.safetensors"), metadata={"format": "pt"})
log("stage 1 written")


# ---- stage 2: a conversation on real audio ------------------------------------------------------------------------------------------
def load_wav(path: str, seconds: float | None = None) -> np.ndarray:
    w = wave.open(path)
    rate = w.getframerate()
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if seconds is not None:
        pcm = pcm[: int(seconds * rate)]
    grid = np.arange(int(len(pcm) * 24000 / rate)) * rate / 24000
    return np.interp(grid, np.arange(len(pcm)), pcm).astype(np.float32)


FRAME = config.frame_samples
user = load_wav(args.user_wav, seconds=args.frames * FRAME / 24000 + 0.1)[: args.frames * FRAME]
user = np.pad(user, (0, args.frames * FRAME - len(user)))
voice = prepare_reference(load_wav(args.voice_wav))
session_module.open_stream = lambda codec, **kwargs: CodecStream(codec)
draws: list[np.ndarray] = []
noise_rng = np.random.default_rng(23)


def fake_randn(*size, generator=None, device=None, dtype=None, **kwargs):
    shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else tuple(size)
    values = noise_rng.standard_normal(shape).astype(np.float32)
    draws.append(values.reshape(-1).copy())
    return torch.as_tensor(values, device=device, dtype=dtype)


torch.randn = fake_randn
card = "Haru is a friendly person who likes a quiet chat."
cfg = session_module.DuplexConfig(temperature=0.0, seed=1)
session = session_module.DuplexSession(model, system_prompt=card, reference=voice, config=cfg)
with torch.no_grad():
    frames = []
    started = time.time()
    frames.append(session.start())
    log(f"stage 2 start: prompt {session.prompt_length} positions, {time.time() - started:.1f} s")
    for i in range(args.frames - 1):
        frames.append(session.step(user[i * FRAME : (i + 1) * FRAME]))
        log(f"stage 2 frame {i + 1}/{args.frames - 1}: token {frames[-1].token} {frames[-1].text!r} {frames[-1].seconds:.1f} s")
tensors = {
    "in.user": torch.from_numpy(user), "in.voice": torch.from_numpy(voice),
    "run.tokens": torch.tensor([f.token for f in frames], dtype=torch.int32),
    "run.audio": torch.from_numpy(np.stack([f.audio for f in frames])),
    "run.prompt_length": torch.tensor(session.prompt_length, dtype=torch.int32),
}
for i, values in enumerate(draws):
    tensors[f"run.draw.{i}"] = torch.from_numpy(values)
save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / "real_session_expected.safetensors"), metadata={"format": "pt"})
(out / "real_session_expected.json").write_text(json.dumps({"card": card, "text": [f.text for f in frames]}, ensure_ascii=False, indent=1))
log(f"stage 2 written: {len(frames)} frames, {len(draws)} draws, tokens {[f.token for f in frames]}")
