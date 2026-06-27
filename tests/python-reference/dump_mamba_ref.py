import numpy as np
from gguf import GGUFReader
r=GGUFReader('/tmp/mamba/mamba-130m-hf.F16.gguf')
W={t.name: t.data.astype(np.float32) for t in r.tensors}
D,DI,DS,K,DTR=768,1536,16,4,48; EPS=1e-5
ids=[510,5347,273,6181,310]; L=len(ids)
emb=W['token_embd.weight']  # gguf data shape (vocab, 768)
h=np.stack([emb[i] for i in ids])  # [L,768]
def rms(x,w): return x/np.sqrt((x**2).mean(-1,keepdims=True)+EPS)*w
xn=rms(h, W['blk.0.attn_norm.weight'])              # [L,768]
# in_proj: gguf data (3072,768)=[out,in]; y = xn @ Wt.T
inp=W['blk.0.ssm_in.weight']                         # (3072,768)
xz=xn@inp.T                                          # [L,3072]
x=xz[:,:DI]; z=xz[:,DI:]                             # [L,1536] each
# conv1d depthwise causal kernel4 + silu; conv data (1536,4)=[ch,k]
cw=W['blk.0.ssm_conv1d.weight']; cb=W['blk.0.ssm_conv1d.bias']
xc=np.zeros_like(x)
for s in range(L):
    for j in range(K):
        ti=s-(K-1)+j
        if ti>=0: xc[s]+=cw[:,j]*x[ti]
    xc[s]+=cb
xc=xc/(1+np.exp(-xc))                                # silu
np.save('/tmp/mamba/ref_xc.npy', xc[-1])
# x_proj: data (80,1536)=[out,in]
xp=W['blk.0.ssm_x.weight']
xd=xc@xp.T                                           # [L,80]
dt=xd[:,:DTR]; B=xd[:,DTR:DTR+DS]; C=xd[:,DTR+DS:]   # [L,48],[L,16],[L,16]
# dt_proj data (1536,48)=[out,in] + bias; softplus
dtp=W['blk.0.ssm_dt.weight']; dtb=W['blk.0.ssm_dt.bias']
delta=dt@dtp.T+dtb                                   # [L,1536]
delta=np.where(delta>20,delta,np.log1p(np.exp(delta)))
# scan: A=-exp(A_log) data (1536,16)
A=-np.exp(W['blk.0.ssm_a']); Dp=W['blk.0.ssm_d']
state=np.zeros((DI,DS)); y=np.zeros((L,DI))
for s in range(L):
    dA=np.exp(delta[s][:,None]*A)                    # [DI,DS]
    dBx=delta[s][:,None]*B[s][None,:]*xc[s][:,None]  # [DI,DS]
    state=dA*state+dBx
    y[s]=(state*C[s][None,:]).sum(-1)+Dp*xc[s]
np.save('/tmp/mamba/ref_y.npy', y[-1])
y=y*(z/(1+np.exp(-z)))                               # gate
op=W['blk.0.ssm_out.weight']                         # (768,1536)=[out,in]
mix=y@op.T                                           # [L,768]
out0=h+mix
np.save('/tmp/mamba/ref_out0.npy', out0[-1])
print('ref block0 last first6:',[round(float(v),4) for v in out0[-1][:6]])
print('ref xc last first4:',[round(float(v),4) for v in xc[-1][:4]])
