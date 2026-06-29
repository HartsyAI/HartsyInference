import os, sys, json, glob, types, importlib.util, gc
os.environ['TORCHDYNAMO_DISABLE']='1'; os.environ['TORCHINDUCTOR_DISABLE']='1'
import torch, numpy as np
torch.set_grad_enabled(False)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m
pkg=types.ModuleType('hmpkg'); pkg.__path__=[]; sys.modules['hmpkg']=pkg
load('hmpkg.configuration_heartmula','/tmp/heartlib/src/heartlib/heartmula/configuration_heartmula.py')
modmod=load('hmpkg.modeling_heartmula','/tmp/heartlib/src/heartlib/heartmula/modeling_heartmula.py')
cfgmod=sys.modules['hmpkg.configuration_heartmula']
HeartMuLa=modmod.HeartMuLa; HeartMuLaConfig=cfgmod.HeartMuLaConfig
snap=glob.glob('/home/kalebbroo/.cache/huggingface/hub/models--HeartMuLa--HeartMuLa-oss-3B/snapshots/*')[0]
cfg=HeartMuLaConfig(**{k:v for k,v in json.load(open(snap+'/config.json')).items() if k in ('backbone_flavor','decoder_flavor','text_vocab_size','audio_vocab_size','audio_num_codebooks','muq_dim')})
with torch.device('meta'):
    model=HeartMuLa(cfg)
model=model.to(torch.bfloat16)
from safetensors import safe_open
sd={}
for st in sorted(glob.glob(snap+'/*.safetensors')):
    with safe_open(st,framework='pt') as f:
        for k in f.keys(): sd[k]=f.get_tensor(k).to(torch.bfloat16)
missing,unexpected=model.load_state_dict(sd, strict=False, assign=True)
print('missing',len(missing),missing[:8]); print('unexpected',len(unexpected),unexpected[:8], flush=True)
del sd; gc.collect()
# any param still on meta?
meta=[n for n,p in model.named_parameters() if p.is_meta]
print('still-meta params:',meta[:8],'count',len(meta), flush=True)
model=model.to('cpu')  # materialize any remaining (buffers)
# build rope caches (skipped during meta construction)
for m in model.modules():
    if hasattr(m,'rope_init') and hasattr(m,'is_cache_built'):
        m.is_cache_built=False; m.rope_init()
model.setup_caches(1)
P=9;S=6
tokens=torch.zeros(1,S,P,dtype=torch.long); text_ids=[128000,100,200,300,400,128001]
for i,v in enumerate(text_ids): tokens[0,i,-1]=v
tmask=torch.zeros_like(tokens,dtype=torch.bool); tmask[:,:,-1]=True
input_pos=torch.arange(S).unsqueeze(0)
embeds=model._embed_tokens(tokens,uncond_mask=None)
h=(embeds*tmask.unsqueeze(-1)).sum(dim=2,dtype=embeds.dtype).to(torch.bfloat16)
h=model.backbone(h,input_pos=input_pos,mask=model.backbone_causal_mask[input_pos,:])
last_h=h[:,-1,:].to(torch.bfloat16)
c0=model.codebook0_head(last_h).float().cpu().numpy()
print('c0 argmax',int(c0.argmax()),'max',float(c0.max()), flush=True)
fixed=[11,22,33,44,55,66,77,88]
model.decoder.reset_caches()
curr_h=torch.cat([last_h.unsqueeze(1),model._embed_audio(0,torch.tensor([[fixed[0]]]))],dim=1).to(embeds.dtype)
curr_pos=torch.arange(0,curr_h.size(1)).unsqueeze(0); dec=[]
for i in range(1,cfg.audio_num_codebooks):
    dh=model.decoder(model.projection(curr_h),input_pos=curr_pos,mask=model.decoder_causal_mask[curr_pos,:])
    ci=torch.mm(dh[:,-1,:].to(torch.bfloat16),model.audio_head[i-1]); dec.append(ci.float().cpu().numpy()[0])
    curr_h=model._embed_audio(i,torch.tensor([[fixed[i]]])).to(embeds.dtype); curr_pos=curr_pos[:,-1:]+1
dec=np.stack(dec)
print('dec argmaxes',[int(x.argmax()) for x in dec], flush=True)
np.savez('/tmp/heartmula_ref/ref.npz',text_ids=np.array(text_ids),fixed_codes=np.array(fixed),
    last_h=last_h.float().cpu().numpy(),c0_logits=c0,dec_logits=dec)
np.savez('/tmp/heartmula_ref/emb.npz',
    text_emb_128000=model.text_embeddings(torch.tensor(128000)).float().cpu().numpy(),
    audio_emb_b0_c11=model._embed_audio(0,torch.tensor([[11]]))[0,0].float().cpu().numpy(),
    audio_emb_b3_c44=model._embed_audio(3,torch.tensor([[44]]))[0,0].float().cpu().numpy())
print('SAVED', flush=True)
