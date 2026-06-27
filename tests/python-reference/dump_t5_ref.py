"""Reference for T5/FLAN-T5 seq2seq GGUFs. Validates the C# T5Model encoder (last_hidden_state) + first decoder
logits vs HF T5ForConditionalGeneration (transformers >=4.x, shared input ids). Validated: flan-t5-small encoder
cosine=1.0, decoder first-token cosine=1.0 + argmax 644 ("Das"); e2e "Das Haus ist schön."."""
import torch, numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
tok=T5Tokenizer.from_pretrained('google/flan-t5-small')
mdl=T5ForConditionalGeneration.from_pretrained('google/flan-t5-small',torch_dtype=torch.float32).eval()
ein=tok('translate English to German: The house is wonderful.', return_tensors='pt')
print('IN_IDS:',','.join(map(str,ein['input_ids'][0].tolist())))
with torch.no_grad():
    enc=mdl.encoder(**ein).last_hidden_state
    lg=mdl(input_ids=ein['input_ids'], decoder_input_ids=torch.tensor([[0]])).logits[0,-1]
print('enc_last first6:',[round(float(x),4) for x in enc[0,-1][:6]])
print('dec argmax=',int(lg.argmax()),repr(tok.decode([int(lg.argmax())])))
