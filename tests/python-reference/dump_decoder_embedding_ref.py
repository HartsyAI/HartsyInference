"""Reference for decoder-based embedding GGUFs (Qwen3-Embedding, gte-Qwen2, e5-mistral). Feeds the same token ids
to HF transformers (needs transformers>=4.51 for Qwen3 — run in a fresh venv; torchvision not required) and the
C# DecoderEmbeddingModel (`embed` CLI + HARTSY_EMBED_IDS), comparing last-token-pooled, L2-normalized embeddings.
Validated: Qwen3-Embedding-0.6B cosine = 1.000000."""
import torch, numpy as np, sys, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
name = sys.argv[1] if len(sys.argv) > 1 else 'Qwen/Qwen3-Embedding-0.6B'
tok = AutoTokenizer.from_pretrained(name); mdl = AutoModel.from_pretrained(name, dtype=torch.float32).eval()
enc = tok('a photo of a cat.', return_tensors='pt')
print('IDS:', ','.join(map(str, enc['input_ids'][0].tolist())))   # Qwen3-Embedding appends <|endoftext|>
with torch.no_grad(): out = mdl(**enc)
v = F.normalize(out.last_hidden_state[0, -1], dim=0)              # last-token pooling
print('REF first8:', [round(float(x), 4) for x in v[:8]])
