#!/usr/bin/env python3
"""Dump the MusicGen-small upstream reference tensors for MusicGenParityTests.

Produces a single `refs.safetensors` with fixed-input -> fixed-output IO for the three risky pieces:
  * T5-base text encoder hidden states (t5_ids/t5_mask -> t5_hidden)
  * decoder teacher-forced logits (dec_codes + dec_cross -> dec_logits, [K,T,vocab])
  * EnCodec 32 kHz decode (codes -> encodec_wav)

Run inside a venv with: pip install 'transformers==4.44.2' 'torch' 'numpy<2' safetensors sentencepiece
Usage:  TORCHDYNAMO_DISABLE=1 python dump_musicgen_reference.py --output /path/to/refs.safetensors

Cast only the submodule under test to f32 (NEVER .float() the whole model) per the parity-loop memory rules.
Then point the C# test at it:  MUSICGEN_REF=/path/to/refs.safetensors dotnet test ... --filter MusicGenParity
"""
import argparse, os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
import numpy as np
import torch
from transformers import MusicgenForConditionalGeneration, AutoTokenizer, EncodecModel
from safetensors.numpy import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--prompt", default="upbeat 80s synthwave with driving drums")
    ap.add_argument("--seqlen", type=int, default=256)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").eval()
    tok = AutoTokenizer.from_pretrained("facebook/musicgen-small")
    d = {}

    # ---- (a) T5 text encoder ----
    enc = tok([args.prompt], return_tensors="pt", padding="max_length",
              max_length=args.seqlen, truncation=True)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    with torch.no_grad():
        te = model.text_encoder.float()
        t5_hidden = te(input_ids=ids, attention_mask=mask).last_hidden_state
    d["t5_ids"] = np.ascontiguousarray(ids.numpy().astype(np.int64))
    d["t5_mask"] = np.ascontiguousarray(mask.numpy().astype(np.int64))
    d["t5_hidden"] = np.ascontiguousarray(t5_hidden.numpy().astype(np.float32))

    # ---- (b) decoder teacher-forced logits ----
    T = 6
    torch.manual_seed(1)
    dec_codes = torch.randint(0, 2048, (1, 4, T), dtype=torch.long)
    S = 8
    torch.manual_seed(2)
    dec_cross = torch.randn(1, S, 1024)  # POST enc_to_dec_proj (hidden=1024)
    with torch.no_grad():
        decf = model.decoder.float()
        out = decf(input_ids=dec_codes.reshape(4, T),
                   encoder_hidden_states=dec_cross,
                   encoder_attention_mask=torch.ones(1, S, dtype=torch.long))
    d["dec_codes"] = np.ascontiguousarray(dec_codes.numpy().astype(np.int64))
    d["dec_cross"] = np.ascontiguousarray(dec_cross.numpy().astype(np.float32))
    d["dec_logits"] = np.ascontiguousarray(out.logits.float().numpy().astype(np.float32))  # [K,T,vocab]

    # ---- (c) EnCodec 32 kHz decode ----
    codec = EncodecModel.from_pretrained("facebook/encodec_32khz").float().eval()
    Tc = 50
    torch.manual_seed(0)
    codes = torch.randint(0, 2048, (1, 4, Tc), dtype=torch.long)
    with torch.no_grad():
        wav = codec.decode(codes.unsqueeze(0), [None])[0]  # [B,1,samples]
    d["codes"] = np.ascontiguousarray(codes.numpy().astype(np.int64))
    d["encodec_wav"] = np.ascontiguousarray(wav.squeeze(0).squeeze(0).numpy().astype(np.float32))

    save_file(d, args.output)
    print("wrote", args.output, {k: (str(v.dtype), v.shape) for k, v in d.items()})


if __name__ == "__main__":
    main()
