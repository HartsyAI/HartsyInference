"""Mimi split-EMA-RVQ decode parity oracle: codes[1,32,T] -> quantized embeddings[1,512,T]."""
import os, torch
from transformers import MimiModel
from safetensors.torch import save_file

OUT = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(1)
model = MimiModel.from_pretrained("kyutai/mimi").eval()
q = model.quantizer
nq = model.config.num_quantizers
T = 16
codes = torch.randint(0, model.config.codebook_size, (1, nq, T), dtype=torch.long)
with torch.no_grad():
    quant = q.decode(codes)            # [1, 512, T]
print("codes", tuple(codes.shape), "quant", tuple(quant.shape), "nsem", model.config.num_semantic_quantizers)
save_file({
    "codes": codes.to(torch.int32).contiguous(),
    "quant": quant.float().contiguous(),
}, os.path.join(OUT, "mimi_rvq_ref.safetensors"))
print("wrote rvq ref")
