"""Hunyuan3D-2 ShapeVAE (decode-only) parity oracle. Loads the hy3dgen decode classes (post_kl + Transformer
resblocks + CrossAttentionDecoder geo_decoder) standalone — bypassing the diffusers-broken package __init__ by
registering minimal fake parent packages — runs decode on a fixed latent + fixed query points, and dumps the
occupancy. Also extracts the vae weights (F32) for the C# loader. Env HY3D_VAE, HY3D_REPO."""
import os, sys, types, numpy as np, torch, torch.nn as nn, importlib.util as u
from safetensors import safe_open
from safetensors.torch import save_file
REPO=os.environ.get("HY3D_REPO","/tmp/Hunyuan3D-2")
VAE=os.environ.get("HY3D_VAE","/tmp/hy3d/hunyuan3d-vae-v2-0/model.fp16.safetensors")
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"hunyuan3d_reference_tensors")
AB=os.path.join(REPO,"hy3dgen/shapegen/models/autoencoders")

# minimal fake package tree so the relative imports in attention_blocks resolve
for name in ["hy3dgen","hy3dgen.shapegen","hy3dgen.shapegen.utils","hy3dgen.shapegen.models","hy3dgen.shapegen.models.autoencoders"]:
    m=types.ModuleType(name); m.__path__=[]; sys.modules[name]=m
sys.modules["hy3dgen.shapegen.utils"].logger=types.SimpleNamespace(info=lambda *a,**k:None,warning=lambda *a,**k:None)
def load(modname, path):
    spec=u.spec_from_file_location(modname, path); mod=u.module_from_spec(spec); sys.modules[modname]=mod; spec.loader.exec_module(mod); return mod
load("hy3dgen.shapegen.models.autoencoders.attention_processors", os.path.join(AB,"attention_processors.py"))
ab=load("hy3dgen.shapegen.models.autoencoders.attention_blocks", os.path.join(AB,"attention_blocks.py"))

# extract + save vae weights (F32)
w={}
with safe_open(VAE,"pt") as f:
    for k in f.keys(): w[k]=f.get_tensor(k).float().contiguous()
save_file(w, os.path.join(OUT,"vae_weights.safetensors")); print("extracted vae weights:", len(w))

# build decode path
post_kl=nn.Linear(64,1024,bias=True)
transformer=ab.Transformer(n_ctx=3072,width=1024,layers=16,heads=16,qkv_bias=False,qk_norm=True)
fourier=ab.FourierEmbedder(num_freqs=8,include_pi=False)
geo=ab.CrossAttentionDecoder(fourier_embedder=fourier,out_channels=1,num_latents=3072,width=1024,heads=16,qkv_bias=False,qk_norm=True)
sd={}
for k,v in w.items():
    if k.startswith("post_kl."): sd.setdefault("post_kl",{})[k[len("post_kl."):]]=v
for name,mod,pre in [("transformer",transformer,"transformer."),("geo",geo,"geo_decoder.")]:
    md={k[len(pre):]:v for k,v in w.items() if k.startswith(pre)}
    miss,unexp=mod.load_state_dict(md,strict=False); print(name,"missing",len(miss),"unexpected",len(unexp), miss[:2])
post_kl.load_state_dict({"weight":w["post_kl.weight"],"bias":w["post_kl.bias"]})
for m in (post_kl,transformer,geo): m.eval()

torch.manual_seed(7)
latent=torch.randn(1,3072,64)
N=4096
queries=(torch.rand(1,N,3)*2-1)*1.01
with torch.no_grad():
    x=post_kl(latent); x=transformer(x)
    occ=geo(queries=queries, latents=x)   # [1,N,1]
save_file({"latent":latent.contiguous(),"queries":queries.contiguous(),"occupancy":occ.float().contiguous()},
          os.path.join(OUT,"vae_ref_io.safetensors"))
print("occupancy",tuple(occ.shape),"std",occ.std().item(),"mean",occ.mean().item(),"min",occ.min().item(),"max",occ.max().item())
