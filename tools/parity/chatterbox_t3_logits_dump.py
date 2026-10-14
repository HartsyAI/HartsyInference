"""Dumps upstream Chatterbox T3 teacher-forced speech logits for the engine's T3 parity test.

Builds the prefill exactly as upstream T3.inference (resemble-ai/chatterbox src/chatterbox/models/t3/t3.py) does for
ChatterboxTTS.generate with the default voice (conds.pt): text = [START] + EnTokenizer(punc_norm(text)) + [STOP],
batch of two for CFG with the second row's text embedding zeroed, then the extra BOS embedding t3.py appends. Then
feeds FORCED speech tokens one at a time (position i + 1, as the sampling loop does) and records the raw cond and
uncond speech logits before every step. CPU, float32.

Writes into <out_dir>:
  t3_logits_<name>.f32   float32 [steps, 2, 8194] (row 0 cond, row 1 uncond), steps = len(FORCED) + 1
  t3_logits_meta.json    text, text token ids, forced tokens, exaggeration, cond-prompt length

    python tools/parity/chatterbox_t3_logits_dump.py <chatterbox ckpt dir> <out_dir>
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from chatterbox.models.t3 import T3
from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.tokenizers import EnTokenizer
from chatterbox.tts import Conditionals, punc_norm

TEXTS = {"yes": "Yes.", "okay": "Okay, see you."}
FORCED = [3784, 4325, 2133, 6486, 160, 1863, 4377, 4325]
EXAGGERATION = 0.5


def load_conds(path: str) -> Conditionals:
    """Reads conds.pt as upstream ships it (torch pickle) or as the engine's safetensors repack of it."""
    with open(path, "rb") as f:
        is_pickle = f.read(2) == b"PK"
    if is_pickle:
        return Conditionals.load(path)
    flat = load_file(path)
    t3 = {k[3:]: v for k, v in flat.items() if k.startswith("t3.")}
    gen = {k[4:]: v for k, v in flat.items() if k.startswith("gen.")}
    return Conditionals(T3Cond(**t3), gen)


@torch.inference_mode()
def dump(t3: T3, conds: Conditionals, tokenizer: EnTokenizer, text: str) -> tuple:
    ids = tokenizer.text_to_tokens(punc_norm(text))
    tokens = torch.cat([ids, ids], dim=0)
    tokens = F.pad(tokens, (1, 0), value=t3.hp.start_text_token)
    tokens = F.pad(tokens, (0, 1), value=t3.hp.stop_text_token).long()
    initial = t3.hp.start_speech_token * torch.ones_like(tokens[:, :1])
    embeds, _ = t3.prepare_input_embeds(t3_cond=conds.t3, text_tokens=tokens, speech_tokens=initial, cfg_weight=0.5)
    bos = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long)
    bos_embed = t3.speech_emb(bos) + t3.speech_pos_emb.get_fixed_embedding(0)
    inputs = torch.cat([embeds, torch.cat([bos_embed, bos_embed])], dim=1)
    model = T3HuggingfaceBackend(config=t3.cfg, llama=t3.tfmr, speech_enc=t3.speech_emb, speech_head=t3.speech_head)
    out = model(inputs_embeds=inputs, past_key_values=None, use_cache=True, output_hidden_states=True, return_dict=True)
    steps = [out.logits[:, -1, :].float().numpy()]
    past = out.past_key_values
    for i, tok in enumerate(FORCED):
        e = t3.speech_emb(torch.tensor([[tok]])) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
        out = model(inputs_embeds=torch.cat([e, e]), past_key_values=past, output_hidden_states=True, return_dict=True)
        past = out.past_key_values
        steps.append(out.logits[:, -1, :].float().numpy())
    return tokens[0].tolist(), np.stack(steps).astype(np.float32)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: chatterbox_t3_logits_dump.py <ckpt dir> <out_dir>")
    ckpt, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0)
    t3 = T3()
    t3.load_state_dict(load_file(os.path.join(ckpt, "t3_cfg.safetensors")))
    t3.eval()
    conds = load_conds(os.path.join(ckpt, "conds.pt"))
    conds.t3.emotion_adv = EXAGGERATION * torch.ones(1, 1, 1)
    tokenizer = EnTokenizer(os.path.join(ckpt, "tokenizer.json"))
    meta = {"forced": FORCED, "exaggeration": EXAGGERATION,
            "cond_prompt_len": int(conds.t3.cond_prompt_speech_tokens.shape[-1]), "texts": {}}
    for name, text in TEXTS.items():
        ids, logits = dump(t3, conds, tokenizer, text)
        logits.tofile(os.path.join(out_dir, f"t3_logits_{name}.f32"))
        meta["texts"][name] = {"text": text, "ids": ids, "steps": int(logits.shape[0])}
        print(name, ids, logits.shape, "argmax cond step0", int(logits[0, 0].argmax()))
    with open(os.path.join(out_dir, "t3_logits_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
