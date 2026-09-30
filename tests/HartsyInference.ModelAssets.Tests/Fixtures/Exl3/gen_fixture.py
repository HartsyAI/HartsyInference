import ctypes, sys, numpy as np, os
IN=int(sys.argv[1]); OUT=int(sys.argv[2]); seed=int(sys.argv[3]); outdir=sys.argv[4]
rng=np.random.default_rng(seed)
K=2
trellis=rng.integers(-32768,32768,size=(IN//16,OUT//16,16*K),dtype=np.int64).astype(np.int16)
suh=(rng.choice([-1.0,1.0],IN)*rng.uniform(0.5,2.0,IN)).astype(np.float16)
svh=(rng.choice([-1.0,1.0],OUT)*rng.uniform(0.5,2.0,OUT)).astype(np.float16)
cu=ctypes.CDLL("libcuda.so.1")
def ck(r,w=""):
    if r: raise RuntimeError(f"{w} cu error {r}")
ck(cu.cuInit(0)); dev=ctypes.c_int(); ck(cu.cuDeviceGet(ctypes.byref(dev),0))
name=ctypes.create_string_buffer(100); cu.cuDeviceGetName(name,100,dev); print("device:",name.value.decode())
ctx=ctypes.c_void_p(); ck(cu.cuCtxCreate_v2(ctypes.byref(ctx),0,dev),"ctx")
mod=ctypes.c_void_p(); ptx=open("h.ptx","rb").read()+b"\0"
ck(cu.cuModuleLoadData(ctypes.byref(mod),ptx),"load")
def fn(n):
    f=ctypes.c_void_p(); ck(cu.cuModuleGetFunction(ctypes.byref(f),mod,n.encode()),n); return f
def dalloc(nb):
    p=ctypes.c_uint64(); ck(cu.cuMemAlloc_v2(ctypes.byref(p),nb)); return p
def h2d(arr):
    p=dalloc(arr.nbytes); ck(cu.cuMemcpyHtoD_v2(p,arr.ctypes.data_as(ctypes.c_void_p),arr.nbytes)); return p
dt,ds,dv=h2d(trellis),h2d(suh),h2d(svh)
nW=IN*OUT*2; dw=dalloc(nW); dW=dalloc(nW)
def launch(f,grid,block,args):
    arr=(ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.pointer(a),ctypes.c_void_p) for a in args])
    ck(cu.cuLaunchKernel(f,grid[0],grid[1],grid[2],block,1,1,0,None,arr,None),"launch")
    ck(cu.cuCtxSynchronize(),"sync")
i32=lambda v: ctypes.c_int(v)
launch(fn("k_recon"),(OUT//128,IN//16,1),256,[dw,dt,i32(OUT//16),i32(0)])
launch(fn("k_recon_had"),(OUT//128,IN//128,1),256,[dW,dt,ds,dv,i32(OUT//16),i32(0)])
w_hat=np.empty(IN*OUT,np.float16); W=np.empty(IN*OUT,np.float16)
ck(cu.cuMemcpyDtoH_v2(w_hat.ctypes.data_as(ctypes.c_void_p),dw,nW)); ck(cu.cuMemcpyDtoH_v2(W.ctypes.data_as(ctypes.c_void_p),dW,nW))
os.makedirs(outdir,exist_ok=True)
trellis.tofile(f"{outdir}/trellis.i16"); suh.tofile(f"{outdir}/suh.f16"); svh.tofile(f"{outdir}/svh.f16")
w_hat.tofile(f"{outdir}/w_hat.f16"); W.tofile(f"{outdir}/w_fused.f16")
print("done", IN, OUT, np.abs(w_hat.astype(np.float32)).max(), np.abs(W.astype(np.float32)).max())
