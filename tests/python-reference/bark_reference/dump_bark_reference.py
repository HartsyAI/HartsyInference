"""Bark semantic + coarse stage logits parity oracle (suno/bark) via transformers BarkModel.
Teacher-forced logits over a fixed token sequence per stage; the C# BarkCausalStage.ForwardLogits matches.
The EnCodec-24k codec half is verified separately."""
import os, torch
from transformers import BarkModel
from safetensors.torch import save_file

OUT = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("BARK_DIR", "suno/bark")
m = BarkModel.from_pretrained(SRC, torch_dtype=torch.float32).eval()
print("loaded BarkModel")

torch.manual_seed(0)
d = {}

def stage_logits(name, model, in_vocab, ids):
    t = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        # BarkCausalModel.forward: input_embeds + position_embeds -> layers -> lm_head
        out = model(input_ids=t)
        logits = out.logits if hasattr(out, "logits") else out[0]
    logits = logits.float().contiguous()   # [1, T, out_vocab]
    print(name, "logits", tuple(logits.shape), "argmax(last)", int(logits[0, -1].argmax()))
    d[f"{name}_ids"] = t.to(torch.int32).contiguous()
    d[f"{name}_logits"] = logits
    d[f"{name}_argmax"] = logits[0].argmax(-1).to(torch.int32).contiguous()

# Semantic: in-range semantic-input ids (vocab 129600). Use small ids + a couple control tokens.
stage_logits("semantic", m.semantic, 129_600, [10048, 100, 5000, 12345, 0, 10000])
# Coarse: in==out vocab 12096.
stage_logits("coarse", m.coarse_acoustics, 12_096, [12048, 50, 1024, 6000, 0, 2050])

save_file(d, os.path.join(OUT, "bark_ref_io.safetensors"))
print("wrote bark_ref_io.safetensors")

# Fine stage: forward(codebook_idx, input_ids[B,T,8]) → logits for that codebook.
T = 6
grid = torch.randint(0, 1024, (1, T, 8), dtype=torch.long)
cb_idx = 2
with torch.no_grad():
    fl = m.fine_acoustics(cb_idx, input_ids=grid).logits.float().contiguous()   # [1, T, fine_vocab]
print("fine logits", tuple(fl.shape), "cb_idx", cb_idx, "argmax(last)", int(fl[0, -1].argmax()))
d["fine_grid"] = grid.to(torch.int32).contiguous()
d["fine_cb_idx"] = torch.tensor([cb_idx], dtype=torch.int32)
d["fine_logits"] = fl
d["fine_argmax"] = fl[0].argmax(-1).to(torch.int32).contiguous()
save_file(d, os.path.join(OUT, "bark_ref_io.safetensors"))   # re-save with fine added

# Dump semantic + coarse + fine stage weights as safetensors (HF Bark key layout the C# expects).
sd = m.state_dict()
w = {k: v.detach().clone().float().contiguous() for k, v in sd.items()
     if k.startswith("semantic.") or k.startswith("coarse_acoustics.") or k.startswith("fine_acoustics.")}
save_file(w, os.path.join(OUT, "bark_weights.safetensors"))
print("wrote bark_weights.safetensors:", len(w), "tensors")

