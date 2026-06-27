import numpy as np, torch, torch.nn.functional as F
from gguf import GGUFReader
DUMP='/tmp/claude-1000/-home-kalebbroo-Desktop-Projects-SharpInference/eb7ffcf1-a730-4648-9a49-2e6f0df16099/scratchpad/llavadump'
r=GGUFReader('/tmp/llava/mmproj-model-f16.gguf')
W={t.name: torch.from_numpy(t.data.astype(np.float32).copy()) for t in r.tensors}
HID,LAYERS,HEADS,INTER,PATCH,IMG=1024,23,16,4096,14,336
GRID=(IMG-PATCH)//PATCH+1; NP=GRID*GRID; SEQ=NP+1; D=HID//HEADS; EPS=1e-5
def ln(x,w,b): return F.layer_norm(x,(x.shape[-1],),w,b,EPS)
def linT(x,name):
    y=x@W[name+'.weight'].T
    if name+'.bias' in W: y=y+W[name+'.bias']
    return y
def qgelu(x): return x*torch.sigmoid(1.702*x)
def load(tag,shape): return torch.from_numpy(np.fromfile(f'{DUMP}/cs_{tag}.f32',dtype=np.float32).reshape(shape))
def cmp(tag,py):
    cs=load(tag,tuple(py.shape)); d=(py-cs).abs()
    cor=np.corrcoef(py.flatten().numpy(),cs.flatten().numpy())[0,1]
    print(f"{tag:8s} py[mean={py.mean():.4f} max={py.abs().max():.3f}] cs[mean={cs.mean():.4f} max={cs.abs().max():.3f}] maxdiff={d.max():.4f} corr={cor:.5f}")
px=load('pixels',(1,3,IMG,IMG))
conv=F.conv2d(px, W['v.patch_embd.weight'].reshape(HID,3,PATCH,PATCH), None, stride=PATCH)  # no bias
patches=conv.reshape(1,HID,NP).transpose(1,2)                       # [1,576,1024]
seq=torch.cat([W['v.class_embd'].view(1,1,HID), patches],dim=1)     # prepend CLS -> [1,577,1024]
seq=seq+W['v.position_embd.weight'].reshape(1,SEQ,HID)
seq=ln(seq,W['v.pre_ln.weight'],W['v.pre_ln.bias'])                  # pre-LN
cmp('seq',seq[0])
h=seq
for i in range(LAYERS):
    p=f'v.blk.{i}'
    x=ln(h,W[f'{p}.ln1.weight'],W[f'{p}.ln1.bias'])
    q=linT(x,f'{p}.attn_q').reshape(1,SEQ,HEADS,D).transpose(1,2)
    k=linT(x,f'{p}.attn_k').reshape(1,SEQ,HEADS,D).transpose(1,2)
    v=linT(x,f'{p}.attn_v').reshape(1,SEQ,HEADS,D).transpose(1,2)
    a=F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(1,SEQ,HID)
    h=h+linT(a,f'{p}.attn_out')
    x=ln(h,W[f'{p}.ln2.weight'],W[f'{p}.ln2.bias'])
    up=qgelu(linT(x,f'{p}.ffn_down'))   # ffn_down=fc1(up); CLIP quick-gelu
    h=h+linT(up,f'{p}.ffn_up')          # ffn_up=fc2(down)
    if i==0: cmp('blk0',h[0])
cmp('postln',h[0])   # no post-LN
# LLaVA projector: drop CLS, mm.0 -> gelu(tanh) -> mm.2
pf=h[:,1:,:]
mid=linT(pf,'mm.0'); act=F.gelu(mid,approximate='tanh'); emb=linT(act,'mm.2')
cmp('embeds',emb[0])
