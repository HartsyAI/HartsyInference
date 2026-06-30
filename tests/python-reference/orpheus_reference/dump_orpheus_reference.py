"""Orpheus-3B LM logits parity oracle (unsloth/orpheus-3b-0.1-ft, ungated mirror of canopylabs/orpheus-3b-0.1-ft).
Runs the Llama-3.2-3B backbone on a fixed token sequence and dumps teacher-forced logits for the C# Qwen2Model
(Audio package) to match. bf16 on CPU (~6 GB) — does NOT f32 the whole 3B model (would OOM)."""
import os, sys, torch
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file

OUT = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("ORPHEUS_DIR", "unsloth/orpheus-3b-0.1-ft")
m = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16).eval()
print("loaded", type(m).__name__, "vocab", m.config.vocab_size, "layers", m.config.num_hidden_layers)

# Fixed sequence: Orpheus human-frame tokens + a few arbitrary in-range ids (LM forward is content-agnostic).
ids = torch.tensor([[128259, 128000, 1820, 4062, 14198, 39935, 128009, 128260, 128257, 1234, 5678, 9012]], dtype=torch.long)
with torch.no_grad():
    logits = m(ids).logits.float()        # [1, T, vocab]
T = ids.shape[1]
last = logits[0, -1]
print("logits", tuple(logits.shape), "argmax(last)", int(last.argmax()), "max", float(last.max()))

save_file({
    "input_ids": ids.to(torch.int32).contiguous(),
    "logits": logits.contiguous(),             # [1, T, vocab]
    "argmax": logits[0].argmax(-1).to(torch.int32).contiguous(),  # [T]
}, os.path.join(OUT, "orpheus_ref_io.safetensors"))
print("wrote orpheus_ref_io.safetensors (T=%d)" % T)
