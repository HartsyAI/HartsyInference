"""
Ideogram 4 text-encoder reference: real Qwen3-VL-8B-Instruct, 13-layer tap.

Produces the GROUND-TRUTH conditioning features the C# `Ideogram4Pipeline` should match,
so we can isolate the wrong-image bug to either the encoder (this) or the DiT weights.

It replicates the C# path EXACTLY:
  * chat template = apply_chat_template([{user: prompt}], add_generation_prompt=True),
    tokenized with add_special_tokens=False  (matches upstream pipeline_ideogram4.py and
    our Qwen3Tokenizer.EncodeChat(includeThinkBlock:false) — Qwen3-VL emits NO <think> block).
  * tap = outputs of decoder layers (0,3,6,...,33,35) == HF hidden_states[1,4,...,36]
    (no final model.norm on the intermediates), concatenated channel-wise -> [1, L, 13*4096=53248].

Run on the CLOUD box (needs the 8B encoder; ~16GB bf16 download or pass a local path):
  pip install torch transformers accelerate safetensors numpy
  python3 tests/python-reference/dump_qwen3vl_ideogram4_encoder.py \
      --model Qwen/Qwen3-VL-8B-Instruct \
      --prompt "a photo of a cat"

Then in SwarmUI generate the SAME prompt and compare:
  * [Ideogram4] promptTokens count/ids   vs  this script's "token ids"
      -> mismatch  => tokenizer / chat-template bug
  * [Ideogram4] textFeatures mean/std/min/max/first  vs  this script's stats
      -> mismatch  => encoder forward / weight-load (fp8) bug
      -> match      => encoder is fine; bug is in the DiT real-weight load

For an element-wise diff, set IDEOGRAM4_DEBUG_DIR before generating (C# writes
layers/textFeatures.bin) and point --csbin at it; this script writes the same raw
little-endian F32 layout to <out>/textFeatures_ref.bin.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

# Decoder-layer outputs Ideogram 4 taps (upstream constants.QWEN3_VL_ACTIVATION_LAYERS).
ACT_LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct",
                    help="HF id or local path to the Qwen3-VL-8B-Instruct (language tower).")
    ap.add_argument("--prompt", default="a photo of a cat")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "ideogram4_reference_tensors", "encoder"))
    ap.add_argument("--csbin", default=None,
                    help="Optional path to the C# textFeatures.bin dump for an element-wise diff.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        from transformers import AutoModel
    except Exception:
        AutoModel = None

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    print("=== chat-templated text ===")
    print(repr(text))
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    ids_list = input_ids[0].tolist()
    print(f"\ntoken ids count={len(ids_list)}")
    print("ids =", ",".join(str(i) for i in ids_list))

    print(f"\nLoading model {args.model} (this can take a while / download ~16GB)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    model.eval()

    # The 13-layer tap lives on the LANGUAGE tower. For a VL wrapper the LM is under
    # .model.language_model (or .language_model); plain CausalLM exposes hidden_states directly.
    with torch.no_grad():
        out = model(input_ids=input_ids.to(model.device), output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple length num_layers+1; hs[0]=embeddings, hs[i+1]=decoder layer i output
    print(f"\nhidden_states entries: {len(hs)} (expect num_layers+1 = 37)")

    # HF index for "output of decoder layer L" is L+1.
    feats = [hs[l + 1][0].float() for l in ACT_LAYERS]   # each [L, 4096]
    concat = torch.cat(feats, dim=-1).contiguous()        # [L, 13*4096]
    arr = concat.cpu().numpy().astype(np.float32)
    L, D = arr.shape
    print(f"\ntextFeatures shape [1, {L}, {D}]  (expect D=53248)")

    flat = arr.reshape(-1)
    finite = flat[np.isfinite(flat)]
    mean = float(finite.mean())
    std = float(finite.std())
    print(f"[REF] textFeatures [{L}x{D}] mean={mean:.5f} std={std:.5f} "
          f"min={float(finite.min()):.4f} max={float(finite.max()):.4f} "
          f"nan/inf={int((~np.isfinite(flat)).sum())} "
          f"first=[{flat[0]:.4f}, {flat[1]:.4f}, {flat[2]:.4f}, {flat[3]:.4f}]")

    # Raw little-endian F32, same layout as Ideogram4DebugDump.
    raw_path = os.path.join(args.out, "textFeatures_ref.bin")
    arr.tofile(raw_path)
    np.save(os.path.join(args.out, "textFeatures_ref.npy"), arr)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"prompt": args.prompt, "token_ids": ids_list, "shape": [1, L, D]}, f, indent=2)
    print(f"\nwrote {raw_path}")

    if args.csbin and os.path.exists(args.csbin):
        cs = np.fromfile(args.csbin, dtype=np.float32)
        if cs.size != arr.size:
            print(f"\n[DIFF] size mismatch: C#={cs.size} ref={arr.size} "
                  f"(token-count or dim differs — check token ids first)")
        else:
            d = np.abs(cs - flat)
            denom = np.abs(flat).mean() + 1e-9
            print(f"\n[DIFF] avg_err={d.mean():.3e} max_err={d.max():.3e} "
                  f"rel_avg={d.mean()/denom:.3e}  "
                  f"({'MATCH' if d.mean()/denom < 1e-2 else 'MISMATCH -> encoder bug'})")


if __name__ == "__main__":
    sys.exit(main())
