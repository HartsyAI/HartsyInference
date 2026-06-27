import numpy as np, torch, torch.nn.functional as F
from gguf import GGUFReader
DUMP='/tmp/claude-1000/-home-kalebbroo-Desktop-Projects-SharpInference/eb7ffcf1-a730-4648-9a49-2e6f0df16099/scratchpad/vlmdump'
r=GGUFReader('/tmp/gemma3vdl/mmproj-model-f16.gguf')
W={t.name: torch.from_numpy(t.data.astype(np.float32).copy()) for t in r.tensors}
HID,LAYERS,HEADS,INTER,PATCH,IMG=1152,27,16,4304,14,896
GRID=IMG//PATCH; NP=GRID*GRID; D=HID//HEADS; EPS=1e-6
def ln(x,w,b): return F.layer_norm(x,(x.shape[-1],),w,b,EPS)
def linT(x,name):  # nn.Linear: gguf .data is [out,in]; y = x @ W.T + b
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
    up=F.gelu(linT(x,f'{p}.ffn_down'),approximate='tanh')   # ffn_down = fc1 (UP, 1152->4304)
    h=h+linT(up,f'{p}.ffn_up')                              # ffn_up   = fc2 (DOWN, 4304->1152)
    if i==0: cmp('blk0',h[0])
nrm=ln(h,W['v.post_ln.weight'],W['v.post_ln.bias'])
cmp('postln',nrm[0])
pooled=F.avg_pool2d(nrm.transpose(1,2).reshape(1,HID,GRID,GRID),4).reshape(1,HID,256).transpose(1,2)
sen=pooled/torch.sqrt(pooled.pow(2).mean(-1,keepdim=True)+EPS)*W['mm.soft_emb_norm.weight']
emb=sen@W['mm.input_projection.weight']   # raw param [in,out]; y = x @ W
cmp('embeds',emb[0])
print("REF embeds first8:", emb[0,0,:8].tolist())
