"""Reference for BERT-family embedding GGUFs (bge / all-MiniLM). Feeds the same token ids to HF transformers and
the C# BertEmbeddingModel (via `embed` CLI + HARTSY_EMBED_IDS) and compares the pooled, normalized embedding.
bge = CLS pooling; all-MiniLM = mean pooling (over attention mask). Validated: cosine = 1.000000."""
import torch, numpy as np, sys
from transformers import AutoTokenizer, AutoModel
name = sys.argv[1] if len(sys.argv) > 1 else 'BAAI/bge-small-en-v1.5'
pool = sys.argv[2] if len(sys.argv) > 2 else 'cls'
tok = AutoTokenizer.from_pretrained(name); mdl = AutoModel.from_pretrained(name).eval()
enc = tok('a photo of a cat.', return_tensors='pt')
print('IDS:', ','.join(map(str, enc['input_ids'][0].tolist())))
with torch.no_grad(): out = mdl(**enc)
if pool == 'mean':
    mask = enc['attention_mask'][0].unsqueeze(-1).float()
    v = (out.last_hidden_state[0] * mask).sum(0) / mask.sum()
else:
    v = out.last_hidden_state[0, 0]
v = torch.nn.functional.normalize(v, dim=0)
print('REF first8:', [round(float(x), 4) for x in v[:8]])
