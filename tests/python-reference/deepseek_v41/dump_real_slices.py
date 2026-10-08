import json, torch, numpy as np
import struct
D='/mnt/model-storage/Models/llm/deepseek-v4.1-flash/'
idx=json.load(open(D+'model.safetensors.index.json'))['weight_map']
FP4=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],dtype=torch.float32)
DT={'F8_E4M3':torch.float8_e4m3fn,'F8_E8M0':torch.float8_e8m0fnu,'BF16':torch.bfloat16,'I8':torch.int8,'F32':torch.float32}
HDR={}
def sl(key,r0,r1):
    fn=D+idx[key]
    if fn not in HDR:
        with open(fn,'rb') as fh:
            n=struct.unpack('<Q',fh.read(8))[0]; HDR[fn]=(json.loads(fh.read(n)),8+n)
    h,base=HDR[fn]; t=h[key]; shape=t['shape']; dt=DT[t['dtype']]
    rowb=int(np.prod(shape[1:]))*torch.empty(0,dtype=dt).element_size()
    with open(fn,'rb') as fh:
        fh.seek(base+t['data_offsets'][0]+r0*rowb); buf=bytearray(fh.read((r1-r0)*rowb))
    return torch.frombuffer(buf,dtype=dt).reshape(r1-r0,*shape[1:])
def fp8(key,r0,r1):
    w=sl(key,r0,r1).float(); s=sl(key.replace('.weight','.scale'),r0//32,(r1+31)//32).float()
    s=s.repeat_interleave(32,0).repeat_interleave(32,1)[:r1-r0-0 if False else None]
    s=s[:w.shape[0],:w.shape[1]]
    return w*s
def fp4(key,r0,r1):
    b=sl(key,r0,r1).view(torch.uint8); s=sl(key.replace('.weight','.scale'),r0,r1).float()
    v=torch.stack([FP4[(b&15).long()],FP4[(b>>4).long()]],-1).flatten(1)
    return v*s.repeat_interleave(32,1)
def bf16(key,r0,r1): return sl(key,r0,r1).float()
jobs=[('fp8','layers.0.attn.wkv.weight',0,64),('fp8','layers.0.attn.wo_a.weight',8128,8192),
      ('fp8','layers.2.attn.wq_b.weight',32704,32768),('fp8','layers.1.engram.wkv.weight',25568,25600),
      ('fp4','layers.0.ffn.experts.0.w1.weight',0,64),('fp4','layers.0.ffn.experts.0.w2.weight',5056,5120),
      ('fp4','layers.0.ffn.experts.383.w3.weight',2240,2304),
      ('bf16','embed.weight',0,8),('bf16','embed.weight',129272,129280),('bf16','layers.2.attn.compressor.wgate.weight',0,32)]
man=[]
for kind,key,r0,r1 in jobs:
    t={'fp8':fp8,'fp4':fp4,'bf16':bf16}[kind](key,r0,r1).contiguous()
    n=f"{len(man)}.f32"; t.numpy().astype('<f4').tofile('/'.join(['REAL',n]).replace('REAL',__import__('sys').argv[1]))
    man.append(dict(kind=kind,key=key,r0=r0,r1=r1,cols=t.shape[1],file=n,sum=float(t.double().sum()),absmax=float(t.abs().max())))
json.dump(man,open(__import__('sys').argv[1]+'/manifest.json','w'),indent=1)
print(json.dumps(man)[:600])
