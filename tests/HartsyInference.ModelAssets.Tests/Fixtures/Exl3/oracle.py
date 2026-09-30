import numpy as np, sys
d=sys.argv[1]; IN=int(sys.argv[2]); OUT=int(sys.argv[3])
tr=np.fromfile(f"{d}/trellis.i16",np.int16).reshape(IN//16,OUT//16,32)
suh=np.fromfile(f"{d}/suh.f16",np.float16); svh=np.fromfile(f"{d}/svh.f16",np.float16)
def decode_tile(t16):
    u16=t16.view(np.uint16).astype(np.uint32)   # 32 words
    w32=u16[0::2]|(u16[1::2]<<16)               # 16 uint32
    K=2
    # bit b (0..511) MSB first within uint32 word
    bits=np.zeros(512,np.uint8)
    for j in range(16):
        for k in range(32): bits[j*32+k]=(int(w32[j])>>(31-k))&1
    vals=np.zeros(256,np.float32); out=np.zeros(256,np.float16)
    for t in range(256):
        end=(t+1)*K
        st=0
        for k in range(16): st=(st<<1)|int(bits[(end-16+k)%512])
        x=np.uint32((st*0xCBAC1FED)&0xFFFFFFFF)
        x=np.uint32((int(x)&0x8fff8fff)^0x3b603b60)
        lo=np.array([int(x)&0xFFFF],np.uint16).view(np.float16)[0]; hi=np.array([int(x)>>16],np.uint16).view(np.float16)[0]
        out[t]=np.float16(lo+hi)
    return out
def tile_perm(p):
    lane,e=p//8,p%8
    r=[(lane%4)*2,(lane%4)*2+1,(lane%4)*2+8,(lane%4)*2+9]
    c0=lane//4; c1=c0+8
    return (r[e%4], c0 if e<4 else c1)
W_hat=np.zeros((IN,OUT),np.float16)
for a in range(IN//16):
    for b in range(OUT//16):
        o=decode_tile(tr[a,b])
        for p in range(256):
            r,c=tile_perm(p); W_hat[a*16+r,b*16+c]=o[p]
ref=np.fromfile(f"{d}/w_hat.f16",np.float16).reshape(IN,OUT)
print("W_hat bit-exact vs upstream kernel:", np.array_equal(W_hat.view(np.uint16),ref.view(np.uint16)))
# Hadamard oracle fp64
def had(n):
    H=np.array([[1.0]])
    while H.shape[0]<n: H=np.block([[H,H],[H,-H]])
    return H
H=had(128)/np.sqrt(128)
X=W_hat.astype(np.float64)
for i in range(0,IN,128):
    X[i:i+128,:]=H@X[i:i+128,:]
for o in range(0,OUT,128):
    X[:,o:o+128]=X[:,o:o+128]@H
X=X*suh.astype(np.float64)[:,None]*svh.astype(np.float64)[None,:]
W=np.fromfile(f"{d}/w_fused.f16",np.float16).reshape(IN,OUT).astype(np.float64)
err=np.abs(W-X); print("fused kernel vs fp64 oracle: max abs",err.max(),"max|W|",np.abs(X).max(),"rel-to-max",err.max()/np.abs(X).max())
np.save(f"{d}/W_fp64.npy",X)
