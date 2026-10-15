#!/usr/bin/env python3
"""Stage dumps of upstream VibeVoice-1.5B (fp32, CPU) for the engine's first-frame parity test.

The run reproduces HartsyInference's VibeVoicePipeline.SynthesizeCore on identical inputs: the engine's raw-text
prompt, the -25 dBFS voice prompt zero-padded to whole 3200-sample latent frames, posterior-mean voice latents
(no sampling), and the engine's per-frame diffusion noise (xorshift32 + Box-Muller, seeded with seed ^ 0x51ED270B).
Nothing numeric is changed; hooks only record each stage.

Stages recorded (row-major float32 unless noted): voice_mean [N,64], voice_norm [N,64], voice_connected [N,1536],
tokens (int32) and speech_mask (uint8), prefill_embeds [T,1536], prefill_hidden [T,1536], cond_1 [1536],
neg_hidden_1 [1536]; then for every diffusion frame f: cond_f [1536], neg_hidden_f [1536], noise_f [64],
denoise_f_stepK [64], z_f [64], raw_f [64], frame_f [3200], sem_f [128], acoustic_embed_f [1536], semantic_embed_f [1536].

usage: vibevoice_first_frame_dump.py --weights DIR --model-dir DIR --voice WAV --text TEXT --seed N --out DIR
                                     [--steps 12] [--native-prompt]
  --weights     microsoft--VibeVoice-1.5B directory (safetensors shards + config)
  --model-dir   directory with upstream preprocessor_config.json pointing at a Qwen2.5 tokenizer
  --native-prompt  keep upstream's own " Speaker 0: <text>" spacing (default: the engine's " Speaker 0:<text>")
"""
import argparse
import contextlib
import io
import json
import math
import os
import sys
import types
import wave

import numpy as np
import torch

REF_STEPS = 10
CFG_SCALE = 1.3
VOICE_PAD = 3200
M32 = 0xFFFFFFFF


class EngineRng:
    """Mirror of HartsyInference.Audio.Dsp.DeterministicRng (uint32 arithmetic, Box-Muller cosine branch)."""

    def __init__(self, seed: int):
        self.state = (((seed & M32) * 2654435761) + 0x9E3779B9) & M32 | 1

    def _step(self, x: int) -> int:
        x ^= (x << 13) & M32
        x ^= x >> 17
        x ^= (x << 5) & M32
        return x & M32

    def uniform(self) -> float:
        self.state = self._step(self.state)
        return (self.state & 0xFFFFFF) / 16777216.0

    def gaussian(self) -> float:
        u1 = self.uniform()
        u2 = self.uniform()
        if u1 < 1e-7:
            u1 = 1e-7
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def padded_normalized_voice(path: str) -> np.ndarray:
    """Engine voice path: normalize the unpadded clip to -25 dBFS, then zero-pad to whole latent frames."""
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 24000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    gain = 10 ** (-25.0 / 20.0) / (float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) + 1e-6)
    x = x * np.float32(gain)
    n_lat = int(math.ceil(x.shape[0] / float(VOICE_PAD)))
    out = np.zeros(n_lat * VOICE_PAD, dtype=np.float32)
    out[: x.shape[0]] = x
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--native-prompt", action="store_true")
    ap.add_argument("--threads", type=int, default=0, help="torch intra-op threads (0 = min(8, cpu count))")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    torch.set_num_threads(args.threads if args.threads > 0 else max(1, min(8, os.cpu_count() or 1)))
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    import vibevoice.modular.modular_vibevoice_tokenizer as mvt

    REC = {}
    FRAME = [0]
    LM_CALLS = [0]
    ENG = EngineRng((args.seed ^ 0x51ED270B) & M32)

    def rec(name: str, value) -> None:
        if name in REC:
            return
        arr = value.detach().to(torch.float32).cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        REC[name] = np.ascontiguousarray(arr)

    # Upstream's sample() draws posterior noise for voice prompts; the engine uses the mean (Encode, no Sample).
    orig_sample = mvt.VibeVoiceTokenizerEncoderOutput.sample

    def mean_sample(self, dist_type="fix"):
        if dist_type == "gaussian":
            return self.mean, self.std
        return orig_sample(self, dist_type=dist_type)

    mvt.VibeVoiceTokenizerEncoderOutput.sample = mean_sample

    proc = VibeVoiceProcessor.from_pretrained(args.model_dir)
    if not args.native_prompt:
        orig_parse = type(proc)._parse_script

        def nospace_parse(self, script):
            return [(spk, txt[1:] if txt.startswith(" ") else txt) for spk, txt in orig_parse(self, script)]

        proc._parse_script = types.MethodType(nospace_parse, proc)
    voice = padded_normalized_voice(args.voice)
    proc.db_normalize = False

    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        args.weights, torch_dtype=torch.float32, device_map=None, attn_implementation="sdpa")
    model.eval()
    model.set_ddpm_inference_steps(num_steps=REF_STEPS)

    acoustic = model.model.acoustic_tokenizer
    semantic = model.model.semantic_tokenizer
    orig_enc = acoustic.encode
    orig_dec = acoustic.decode
    orig_sem_enc = semantic.encode
    orig_psi = type(model)._process_speech_inputs

    def psi(self, speech_tensors, speech_masks, speech_type="audio"):
        feats, connected = orig_psi(self, speech_tensors, speech_masks, speech_type)
        rec("voice_norm", feats[0])
        rec("voice_connected", connected)
        return feats, connected

    model._process_speech_inputs = types.MethodType(psi, model)

    def enc(*a, **k):
        out = orig_enc(*a, **k)
        rec("voice_mean", out.mean[0])
        return out

    acoustic.encode = enc

    def dec(latent, *a, **k):
        f = FRAME[0]
        rec(f"raw_{f}", latent[0, 0])
        audio = orig_dec(latent, *a, **k)
        rec(f"frame_{f}", audio.reshape(-1))
        return audio

    acoustic.decode = dec

    def sem_enc(*a, **k):
        out = orig_sem_enc(*a, **k)
        rec(f"sem_{FRAME[0]}", out.mean.reshape(-1))
        return out

    semantic.encode = sem_enc

    def connector_hook(name_prefix):
        def hook(module, inputs, output):
            f = FRAME[0]
            if f == 0 or inputs[0].shape[1] != 1:
                return
            rec(f"{name_prefix}_embed_{f}", output[0, 0])
        return hook

    model.model.acoustic_connector.register_forward_hook(connector_hook("acoustic"))
    model.model.semantic_connector.register_forward_hook(connector_hook("semantic"))

    def lm_pre(module, pos_args, kwargs):
        LM_CALLS[0] += 1
        emb = kwargs.get("inputs_embeds", pos_args[0] if pos_args else None)
        if LM_CALLS[0] == 1:
            rec("prefill_embeds", emb[0])

    def lm_post(module, pos_args, output):
        k = LM_CALLS[0]
        h = output.last_hidden_state
        if k == 1:
            rec("prefill_hidden", h[0])
            rec("cond_1", h[0, -1])
        elif k == 2:
            rec("neg_hidden_1", h[0, -1])
        elif k == 3:
            rec("cond_2", h[0, -1])
        elif k == 4:
            rec("neg_hidden_2", h[0, -1])

    model.model.language_model.register_forward_pre_hook(lm_pre, with_kwargs=True)
    model.model.language_model.register_forward_hook(lm_post)

    @torch.no_grad()
    def sample_speech_tokens_dump(self, condition, neg_condition, cfg_scale=3.0):
        FRAME[0] += 1
        f = FRAME[0]
        self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)
        n = torch.tensor([ENG.gaussian() for _ in range(64)], dtype=torch.float32)
        rec(f"noise_{f}", n)
        rec(f"cond_{f}", condition[0])
        rec(f"neg_hidden_{f}", neg_condition[0])
        condition = torch.cat([condition, neg_condition], dim=0).to(self.model.prediction_head.device)
        speech = torch.stack([n, n]).to(condition)
        for step, t in enumerate(self.model.noise_scheduler.timesteps, start=1):
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = self.model.prediction_head(combined, t.repeat(combined.shape[0]).to(combined), condition=condition)
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample
            rec(f"denoise_{f}_step{step}", speech[0])
        out = speech[: len(speech) // 2]
        rec(f"z_{f}", out[0])
        return out

    type(model).sample_speech_tokens = sample_speech_tokens_dump

    inputs = proc(text=[f"Speaker 0: {args.text}"], voice_samples=[[voice]], padding=True,
                  return_tensors="pt", return_attention_mask=True)
    REC["tokens"] = inputs["input_ids"][0].numpy().astype("<i4")
    REC["speech_mask"] = inputs["speech_input_mask"][0].numpy().astype("u1")
    dev = {k: (v.to("cpu") if torch.is_tensor(v) else v) for k, v in inputs.items()}

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = model.generate(**dev, max_new_tokens=args.steps, cfg_scale=CFG_SCALE, tokenizer=proc.tokenizer,
                             generation_config={"do_sample": False}, verbose=True, is_prefill=True,
                             max_length_times=100)
    audio = out.speech_outputs[0]
    stages = []
    for name, arr in REC.items():
        path = os.path.join(args.out, f"{name}.bin")
        arr.tofile(path)
        dtype = "i32" if arr.dtype == np.dtype("<i4") else ("u8" if arr.dtype == np.dtype("u1") else "f32")
        stages.append({"name": name, "shape": list(arr.shape), "dtype": dtype, "file": os.path.basename(path)})
    manifest = {
        "weights": os.path.abspath(args.weights),
        "voice_wav": os.path.abspath(args.voice),
        "text": args.text,
        "seed": args.seed,
        "steps": args.steps,
        "cfg_scale": CFG_SCALE,
        "ddpm_steps": REF_STEPS,
        "prompt": "native" if args.native_prompt else "engine",
        "stop": [line for line in buf.getvalue().splitlines() if "reached" in line][-1:],
        "frames_generated": 0 if audio is None else round(audio.numel() / 3200.0, 2),
        "stages": stages,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"wrote {len(stages)} stages to {args.out}; frames={manifest['frames_generated']} stop={manifest['stop']}")


if __name__ == "__main__":
    sys.exit(main())
