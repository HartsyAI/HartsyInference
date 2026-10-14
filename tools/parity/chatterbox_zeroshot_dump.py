"""Dumps upstream Chatterbox zero-shot conditioning for the engine's ChatterboxZeroShotParityTests.

Runs ChatterboxTTS.prepare_conditionals (resemble-ai/chatterbox src/chatterbox/tts.py) on a reference clip, CPU,
float32, and writes what it derives plus the 16 kHz samples it derived them from. Then records teacher-forced T3
speech logits with that conditioning for FORCED token runs (taken from upstream samples of the same clip), the prefill
built as T3.inference builds it (CFG pair with the uncond text embedding zeroed, the extra BOS embedding t3.py appends).

Writes into <out_dir>:
  ref24.f32               float32 reference at 24 kHz (librosa.load(sr=24000), what embed_ref gets before its 10 s cut)
  ref16_librosa.f32       float32 librosa.resample(ref24 -> 16 kHz), what the voice encoder and the T3 tokens see
  ve.f32                  float32 [256] voice-encoder embedding
  t3_cond_tokens.i32      int32 T3 cond-prompt speech tokens
  gen_prompt_tokens.i32   int32 embed_ref prompt tokens
  gen_prompt_feat_tm.f32  float32 [frames, 80] embed_ref prompt mel (time-major)
  gen_xvec.f32            float32 [192] CAM++ x-vector
  tf_<n>.f32              float32 [steps, 2, 8194] raw cond/uncond speech logits, steps = len(forced) + 1
  meta.json               shapes, and per tf_<n> the text and forced tokens

    python tools/parity/chatterbox_zeroshot_dump.py <chatterbox ckpt dir> <reference.wav> <out_dir>

The Perth watermarker is not part of conditioning; when resemble-perth is not installed a no-op stand-in is used.
"""
import json
import os
import sys
import types
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

try:
    import perth  # noqa: F401
except ImportError:
    stub = types.ModuleType("perth")
    stub.PerthImplicitWatermarker = lambda: types.SimpleNamespace(apply_watermark=lambda wav, sample_rate=None: wav)
    sys.modules["perth"] = stub

from chatterbox.models.s3gen import S3Gen
from chatterbox.models.t3 import T3
from chatterbox.models.tokenizers import EnTokenizer
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.tts import ChatterboxTTS, punc_norm

EXAGGERATION = 0.5
# Upstream generate() samples of the parity clip: "Yes." seeds 6 and 1, "Okay, see you." seed 1 (EOS dropped).
FORCED = [
    ("Yes.", [3704, 4218, 1898, 3454, 2752, 2276, 6244, 1084, 1703, 731, 5400, 6374, 6373, 6376, 6377, 1029, 1788,
              5916, 3729, 3003, 4299]),
    ("Yes.", [1759, 2895, 2923, 3649, 1904, 3119, 2930, 1032, 2253, 2487, 739, 1494, 4218, 1876, 4299, 5055, 6486, 426,
              2885, 5789, 2420, 890, 1370, 389, 2909, 2804, 3505, 4224, 2112, 4299]),
    ("Okay, see you.", [1759, 2031, 758, 353, 2910, 81, 2364, 5270, 5272, 4787, 2636, 5075, 4723, 2922, 4299, 4218,
                        3975, 2112, 3648, 4671, 6373, 6379, 2979, 314, 566, 3481, 4858, 4606, 3146, 3143, 461, 716,
                        2861, 3589, 4215, 4296, 4299, 6486, 6405]),
]


def load_model(ckpt: Path) -> ChatterboxTTS:
    """ChatterboxTTS.from_local without the built-in voice (zero-shot only needs the three modules)."""
    ve = VoiceEncoder()
    ve.load_state_dict(load_file(ckpt / "ve.safetensors"))
    t3 = T3()
    state = load_file(ckpt / "t3_cfg.safetensors")
    t3.load_state_dict(state["model"][0] if "model" in state.keys() else state)
    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(ckpt / "s3gen.safetensors"), strict=False)
    model = ChatterboxTTS(t3.eval(), s3gen.eval(), ve.eval(), EnTokenizer(str(ckpt / "tokenizer.json")), "cpu")
    return model


def teacher_forced(model: ChatterboxTTS, text: str, forced: list) -> tuple:
    t3, cond = model.t3, model.conds.t3
    ids = model.tokenizer.text_to_tokens(punc_norm(text))
    tokens = torch.cat([ids, ids], dim=0)
    tokens = F.pad(tokens, (1, 0), value=t3.hp.start_text_token)
    tokens = F.pad(tokens, (0, 1), value=t3.hp.stop_text_token)
    bos = torch.tensor([[t3.hp.start_speech_token]])
    embeds, _ = t3.prepare_input_embeds(t3_cond=cond, text_tokens=tokens, speech_tokens=bos.expand(2, 1), cfg_weight=0.5)
    bos_embed = t3.speech_emb(bos) + t3.speech_pos_emb.get_fixed_embedding(0)
    out = t3.tfmr(inputs_embeds=torch.cat([embeds, torch.cat([bos_embed, bos_embed])], dim=1), use_cache=True)
    logits = [t3.speech_head(out.last_hidden_state[:, -1])]
    for i, token in enumerate(forced):
        step = t3.speech_emb(torch.tensor([[token]])) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
        out = t3.tfmr(inputs_embeds=torch.cat([step, step]), past_key_values=out.past_key_values, use_cache=True)
        logits.append(t3.speech_head(out.last_hidden_state[:, -1]))
    return torch.stack(logits, 0), tokens[0].tolist()


@torch.inference_mode()
def main() -> None:
    ckpt, ref, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    model = load_model(ckpt)
    model.prepare_conditionals(ref, exaggeration=EXAGGERATION)
    c = model.conds

    def f32(name, t):
        np.asarray(t, dtype=np.float32).tofile(os.path.join(out, name + ".f32"))

    def i32(name, t):
        np.asarray(t, dtype=np.int32).tofile(os.path.join(out, name + ".i32"))

    ref24, _ = librosa.load(ref, sr=24000)
    ref16 = librosa.resample(ref24, orig_sr=24000, target_sr=16000)
    f32("ref24", ref24)
    f32("ref16_librosa", ref16)
    f32("ve", c.t3.speaker_emb.reshape(-1))
    i32("t3_cond_tokens", c.t3.cond_prompt_speech_tokens.reshape(-1))
    i32("gen_prompt_tokens", c.gen["prompt_token"].reshape(-1))
    f32("gen_prompt_feat_tm", c.gen["prompt_feat"][0])
    f32("gen_xvec", c.gen["embedding"].reshape(-1))
    meta = dict(t3_cond_tokens=c.t3.cond_prompt_speech_tokens.shape[-1], gen_prompt_tokens=c.gen["prompt_token"].shape[-1],
                prompt_feat=list(c.gen["prompt_feat"].shape[1:]), ref24=len(ref24), ref16=len(ref16), tf={})
    for n, (text, forced) in enumerate(FORCED):
        logits, ids = teacher_forced(model, text, forced)
        f32(f"tf_{n}", logits)
        meta["tf"][f"tf_{n}"] = dict(text=text, ids=ids, forced=forced)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "tf"}))


if __name__ == "__main__":
    main()
