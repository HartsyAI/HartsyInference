import numpy as np, torch, torch.nn.functional as F
from gguf import GGUFReader
DUMP='/tmp/claude-1000/-home-kalebbroo-Desktop-Projects-SharpInference/eb7ffcf1-a730-4648-9a49-2e6f0df16099/scratchpad/smolvlmdump'
r=GGUFReader('/tmp/smolvlm/mmproj-SmolVLM2-2.2B-Instruct-f16.gguf')
W={t.name: torch.from_numpy(t.data.astype(np.float32).copy()) for t in r.tensors}
HID,LAYERS,HEADS,INTER,PATCH,IMG,S=1152,27,16,4304,14,384,3
GRID=(IMG-PATCH)//PATCH+1; NP=GRID*GRID; D=HID//HEADS; EPS=1e-5
def ln(x,w,b): return F.layer_norm(x,(x.shape[-1],),w,b,EPS)
def linT(x,name):
    y=x@W[name+'.weight'].T
    if name+'.bias' in W: y=y+W[name+'.bias']
    return y
def load(tag,shape): return torch.from_numpy(np.fromfile(f'{DUMP}/cs_{tag}.f32',dtype=np.float32).reshape(shape))
def cmp(tag,py):
    cs=load(tag,tuple(py.shape)); d=(py-cs).abs()
    cor=np.corrcoef(py.flatten().numpy(),cs.flatten().numpy())[0,1]
    print(f"{tag:8s} py[mean={py.mean():.4f} max={py.abs().max():.3f}] cs[mean={cs.mean():.4f} max={cs.abs().max():.3f}] maxdiff={d.max():.4f} corr={cor:.5f}")
px=load('pixels',(1,3,IMG,IMG))
conv=F.conv2d(px, W['v.patch_embd.weight'].reshape(HID,3,PATCH,PATCH), W['v.patch_embd.bias'], stride=PATCH)
seq=conv.reshape(1,HID,NP).transpose(1,2).contiguous()+W['v.position_embd.weight'].reshape(1,NP,HID)
cmp('seq',seq[0])
h=seq
for i in range(LAYERS):
    p=f'v.blk.{i}'
    x=ln(h,W[f'{p}.ln1.weight'],W[f'{p}.ln1.bias'])
    q=linT(x,f'{p}.attn_q').reshape(1,NP,HEADS,D).transpose(1,2)
    k=linT(x,f'{p}.attn_k').reshape(1,NP,HEADS,D).transpose(1,2)
    v=linT(x,f'{p}.attn_v').reshape(1,NP,HEADS,D).transpose(1,2)
    a=F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(1,NP,HID)
    h=h+linT(a,f'{p}.attn_out')
    x=ln(h,W[f'{p}.ln2.weight'],W[f'{p}.ln2.bias'])
    up=F.gelu(linT(x,f'{p}.ffn_down'),approximate='tanh')   # ffn_down=fc1(up)
    h=h+linT(up,f'{p}.ffn_up')                              # ffn_up=fc2(down)
    if i==0: cmp('blk0',h[0])
nrm=ln(h,W['v.post_ln.weight'],W['v.post_ln.bias'])
cmp('postln',nrm[0])
# idefics3 pixel-shuffle
x=nrm.view(1,GRID,GRID,HID).view(1,GRID,GRID//S,HID*S).permute(0,2,1,3)
x=x.reshape(1,GRID//S,GRID//S,HID*S*S).permute(0,2,1,3).reshape(1,(GRID//S)**2,HID*S*S)
emb=linT(x,'mm.model.fc')   # nn.Linear, no bias
cmp('embeds',emb[0])
