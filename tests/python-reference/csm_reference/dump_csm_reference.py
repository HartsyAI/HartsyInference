"""CSM-1B backbone + c0-head parity oracle (unsloth/csm-1b, transformers CsmForConditionalGeneration).
Also converts the transformers checkpoint to the C# sesame key layout. Verifies the 1B Llama backbone +
codebook-0 head + audio/text embedding tables + context construction. The Mimi codec is verified separately."""
import os, torch
from transformers import CsmForConditionalGeneration
from safetensors.torch import save_file

OUT = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("CSM_DIR", "unsloth/csm-1b")
m = CsmForConditionalGeneration.from_pretrained(SRC, torch_dtype=torch.float32).eval()
print("loaded", type(m).__name__)

sd = m.state_dict()
NCB, VOCAB = 32, 2051
H = 2048

# ── Build the fixed context: N audio frames, each = sum of 32 codebook embeddings (offset cb*VOCAB + code). ──
emb = sd["depth_decoder.model.embed_tokens.weight"]   # [32*2051, 2048]
torch.manual_seed(0)
N = 4
codes = torch.randint(0, 1024, (N, NCB), dtype=torch.long)   # arbitrary in-range codes per frame
ctx = torch.zeros(1, N, H)
for t in range(N):
    for cb in range(NCB):
        ctx[0, t] += emb[cb * VOCAB + codes[t, cb]]

with torch.no_grad():
    bb = m.backbone_model(inputs_embeds=ctx)
    last = bb.last_hidden_state[:, -1]            # [1, 2048]
    c0_logits = m.lm_head(last).float()           # [1, 2051]
    c0 = int(c0_logits[0].argmax())
    # Depth decoder: teacher-forced codes c0..c30 → logits for codebooks 1..31. The model embeds position i as
    # codebook-(i-1) of input_ids[i] and overwrites position 0 with the backbone hidden; head predicts cb at pos i.
    forced = [c0] + [(17 * (i + 1)) % 2048 for i in range(1, 31)]   # c0..c30 (31 codes)
    dd_input = torch.tensor([[0] + forced], dtype=torch.long)        # [1, 32]; pos 0 overwritten
    dd = m.depth_decoder(input_ids=dd_input, backbone_last_hidden_state=last)
    dec_logits = dd.logits.float()                # [1, 31, 2051] for codebooks 1..31
print("c0 argmax", c0, "| dec logits", tuple(dec_logits.shape), "argmax(cb1)", int(dec_logits[0, 0].argmax()))

save_file({
    "ctx_codes": codes.to(torch.int32).contiguous(),   # [N, 32]
    "c0_logits": c0_logits.contiguous(),               # [1, 2051]
    "c0_argmax": c0_logits[0].argmax().to(torch.int32).reshape(1).contiguous(),
    "forced": torch.tensor(forced, dtype=torch.int32).contiguous(),   # [31] = c0..c30
    "dec_logits": dec_logits.contiguous(),             # [1, 31, 2051]
    "dec_argmax": dec_logits[0].argmax(-1).to(torch.int32).contiguous(),  # [31]
}, os.path.join(OUT, "csm_ref_io.safetensors"))
print("wrote csm_ref_io.safetensors")

# ── Convert weights to the C# sesame layout. ──
w = {}
for k, v in sd.items():
    if k.startswith("codec_model."):
        continue
    v = v.float().contiguous()
    if k.startswith("backbone_model."):
        w["backbone." + k[len("backbone_model."):]] = v.clone()
    elif k.startswith("depth_decoder.model.layers.") or k == "depth_decoder.model.norm.weight":
        w["decoder." + k[len("depth_decoder.model."):]] = v.clone()
    elif k == "embed_text_tokens.weight":
        w["text_embeddings.weight"] = v.clone()
    elif k == "lm_head.weight":
        w["codebook0_head.weight"] = v.clone()
    elif k == "depth_decoder.model.inputs_embeds_projector.weight":
        w["projection.weight"] = v.clone()
    elif k == "depth_decoder.model.embed_tokens.weight":
        for i in range(NCB):
            w[f"audio_embeddings.{i}.weight"] = v[i * VOCAB:(i + 1) * VOCAB].clone().contiguous()
    elif k == "depth_decoder.codebooks_head.weight":   # [31, 1024, 2051] = [head, in, out]
        for i in range(v.shape[0]):
            w[f"audio_head.{i}.weight"] = v[i].t().clone().contiguous()   # -> [2051, 1024] (out, in)

save_file(w, os.path.join(OUT, "csm_weights.safetensors"))
print("wrote csm_weights.safetensors:", len(w), "tensors")
