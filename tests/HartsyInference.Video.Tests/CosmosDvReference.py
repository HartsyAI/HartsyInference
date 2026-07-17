"""Cosmos-Tokenize1-DV8x16x16 encoder parity-reference generator (off-ship, Python/torch CPU).

Reconstructs the DV encoder forward in F32 from the shipped weights and validates it is BIT-EXACT to NVIDIA's
`encoder.jit` conv-body (the jit runs bf16 and cannot execute on CPU-only torch — bf16 avg_pool3d is unimplemented —
so we cross-check against the jit's own submodules cast to F32, which match to max|Δ|=0.0). It then emits the fixtures
consumed by CosmosTokenizerTests.CosmosDvTokenizer_Encode_ParityVsEncoderJit:
    cosmos_input.{bin,hdr}            [1,3,9,64,64] f32   fixed synthetic clip in [-1,1]
    cosmos_ref_continuous.{bin,hdr}  [1,6,2,4,4]  f32     pre-FSQ latent (== jit conv-body in F32)
    cosmos_ref_indices.{bin,hdr}     [1,2,4,4]    i32     FSQ token indices

Run: ~/hfvenv/bin/python CosmosDvReference.py   (needs the model.pt + encoder.jit under ~/models_cosmos_dv).
Point the C# test at the output dir via COSMOS_DV_REF and the checkpoint via COSMOS_DV_MODEL.
NOTE: encoder.jit is the bf16 export of the checkpoint's `network.*` set (verified: jit weights == network.* exactly),
so the C# tokenizer loads `network.*`; the residual F32-vs-bf16 gap is a rare FSQ half-integer rounding tie.
"""
import math, torch, torch.nn.functional as F

DEV='cpu'

def load_ema_f32(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)['model']
    sd = {}
    for k,v in ck.items():
        if k.startswith('ema.network-'):
            nk = k[len('ema.network-'):].replace('-', '.')  # encoder.conv_in.0.conv3d.weight
            sd[nk] = v.float()
    return sd

class Ref:
    def __init__(self, sd):
        self.sd = sd
        self.wav = torch.tensor([0.7070,0.7070])  # will override from jit buffer if provided
    # ---- factorized causal conv helpers ----
    def _cw(self, name):  # returns (w,b)
        return self.sd[name+'.weight'], self.sd[name+'.bias']
    def spatial_conv(self, x, name):  # kernel (1,3,3), pad H,W by 1, no temporal
        w,b = self._cw(name)
        x = F.pad(x, [1,1,1,1,0,0], 'constant', 0.0)
        return F.conv3d(x, w, b, stride=1, padding=0)
    def temporal_conv(self, x, name, reps=2, stride=1):  # kernel (3,1,1), causal replicate left
        w,b = self._cw(name)
        xp = x[:,:,:1].repeat(1,1,reps,1,1)
        x = torch.cat([xp, x], 2)
        return F.conv3d(x, w, b, stride=(stride,1,1), padding=0)
    def onexone(self, x, name):
        w,b = self._cw(name)
        return F.conv3d(x, w, b, stride=1, padding=0)
    def factconv(self, x, base):  # base like 'encoder.conv_in'
        x = self.spatial_conv(x, base+'.0.conv3d')
        x = self.temporal_conv(x, base+'.1.conv3d', reps=2, stride=1)
        return x
    # ---- groupnorm num_groups=1 per-frame ----
    def gnorm(self, x, name):
        w = self.sd[name+'.norm.weight']; b = self.sd[name+'.norm.bias']
        B,C,T,H,W = x.shape
        y = x.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        y = F.group_norm(y, 1, w, b, 1e-6)
        return y.reshape(B,T,C,H,W).permute(0,2,1,3,4)
    def silu(self,x): return x*torch.sigmoid(x)
    # ---- resblock ----
    def resblock(self, x, base):
        h = self.gnorm(x, base+'.norm1'); h = self.silu(h)
        h = self.factconv(h, base+'.conv1')
        h = self.gnorm(h, base+'.norm2'); h = self.silu(h)
        h = self.factconv(h, base+'.conv2')
        if base+'.nin_shortcut.conv3d.weight' in self.sd:
            x = self.onexone(x, base+'.nin_shortcut.conv3d')
        return x + h
    # ---- attention (per frame) scale = C^-0.5 ----
    def attn(self, x, base):
        h = self.gnorm(x, base+'.norm')
        q = self.onexone(h, base+'.q.conv3d'); k = self.onexone(h, base+'.k.conv3d'); v = self.onexone(h, base+'.v.conv3d')
        B,C,T,H,W = q.shape
        def r(t): return t.permute(0,2,1,3,4).reshape(B*T, C, H*W)
        qq = r(q).permute(0,2,1); kk = r(k); vv = r(v)
        w_ = torch.bmm(qq, kk) * (C**-0.5)
        w_ = torch.softmax(w_, 2)
        hh = torch.bmm(vv, w_.permute(0,2,1)).reshape(B,T,C,H,W).permute(0,2,1,3,4)
        return x + self.onexone(hh, base+'.proj_out.conv3d')
    def attn_temporal(self, x, base):  # causal temporal self-attention (folds H,W into batch, attends over T)
        h = self.gnorm(x, base+'.norm')
        q = self.onexone(h, base+'.q.conv3d'); k = self.onexone(h, base+'.k.conv3d'); v = self.onexone(h, base+'.v.conv3d')
        B,C,T,H,W = q.shape
        def r(t): return t.permute(0,3,4,1,2).reshape(B*H*W, C, T).permute(0,2,1)  # [BHW, T, C]
        qq, kk, vv = r(q), r(k), r(v)
        w_ = torch.bmm(qq, kk.permute(0,2,1)) * (C**-0.5)   # [BHW, T, T]
        mask = torch.tril(torch.ones_like(w_))
        w_ = w_.masked_fill(mask==0, float('-inf'))
        w_ = torch.softmax(w_, 2)
        hh = torch.bmm(w_, vv)                               # [BHW, T, C]
        hh = hh.permute(0,2,1).reshape(B,H,W,C,T).permute(0,3,4,1,2)  # [B,C,T,H,W]
        return x + self.onexone(hh, base+'.proj_out.conv3d')
    # ---- downsample ----
    def downsample(self, x, base, spatiotemporal):
        # spatial
        xp = F.pad(x, [0,1,0,1,0,0], 'constant', 0.0)
        w,b = self._cw(base+'.conv1.conv3d')
        c1 = F.conv3d(xp, w, b, stride=(1,2,2), padding=0)
        ap = F.avg_pool3d(xp, (1,2,2),(1,2,2),(0,0,0))
        x0 = c1 + ap
        if spatiotemporal:
            x1 = torch.cat([x0[:,:,:1], x0], 2)
            w2,b2 = self._cw(base+'.conv2.conv3d')
            xpp = x1[:,:,:1].repeat(1,1,1,1,1)
            x1p = torch.cat([xpp, x1], 2)
            c2 = F.conv3d(x1p, w2, b2, stride=(2,1,1), padding=0)
            ap2 = F.avg_pool3d(x1, (2,1,1),(2,1,1),(0,0,0))
            x0 = c2 + ap2
        return self.onexone(x0, base+'.conv3.conv3d')
    # ---- patcher (2-level Haar via grouped conv, band-major) ----
    def dwt(self, x, do_repeat):
        if do_repeat:
            xi = x[:,:,:1]; xv = x[:,:,1:]
            x = torch.cat([xi.repeat_interleave(4, dim=2), xv], 2)
        C = x.shape[1]; n = 2
        wav = self.wav.float()
        hl = wav.flip(0).reshape(1,1,-1).repeat(C,1,1)
        hh = (wav * torch.tensor([1.0,-1.0])).reshape(1,1,-1).repeat(C,1,1)
        x = F.pad(x, [0,1,0,1,0,1], 'reflect')
        def cv(inp, filt, dim):  # dim in {2,3,4}
            shp=[C,1,1,1,1]; shp[dim]=n
            f = filt.reshape(shp)
            st=[1,1,1]; st[dim-2]=2
            return F.conv3d(inp, f, None, stride=st, groups=C)
        xl = cv(x, hl, 2); xh = cv(x, hh, 2)
        xll= cv(xl,hl,3); xlh=cv(xl,hh,3); xhl=cv(xh,hl,3); xhh=cv(xh,hh,3)
        b=[cv(xll,hl,4),cv(xll,hh,4),cv(xlh,hl,4),cv(xlh,hh,4),
           cv(xhl,hl,4),cv(xhl,hh,4),cv(xhh,hl,4),cv(xhh,hh,4)]
        out = torch.cat(b,1) / (math.sqrt(2.0)*1.0)
        return out
    def patcher(self, x):
        x = self.dwt(x, True)
        x = self.dwt(x, False)
        return x
    # ---- full encoder ----
    def encode(self, x):
        x = self.patcher(x)
        x = self.factconv(x, 'encoder.conv_in')
        # down.0
        x = self.resblock(x,'encoder.down.0.block.0'); x=self.resblock(x,'encoder.down.0.block.1')
        x = self.downsample(x,'encoder.down.0.downsample', True)
        # down.1
        x = self.resblock(x,'encoder.down.1.block.0'); x=self.resblock(x,'encoder.down.1.block.1')
        x = self.downsample(x,'encoder.down.1.downsample', False)
        # down.2 (no downsample)
        x = self.resblock(x,'encoder.down.2.block.0'); x=self.resblock(x,'encoder.down.2.block.1')
        # mid
        x = self.resblock(x,'encoder.mid.block_1')
        x = self.attn(x,'encoder.mid.attn_1.0'); x=self.attn_temporal(x,'encoder.mid.attn_1.1')
        x = self.resblock(x,'encoder.mid.block_2')
        x = self.gnorm(x,'encoder.norm_out'); x=self.silu(x)
        x = self.factconv(x, 'encoder.conv_out')
        # quant_conv 1x1x1
        x = self.onexone(x, 'quant_conv.conv3d')
        return x  # [B,6,T,H,W] pre-FSQ continuous
    # ---- FSQ ----
    def fsq(self, z):  # z [B,6,T,H,W]
        levels = torch.tensor([8,8,8,5,5,5], dtype=torch.float32)
        x = z.permute(0,2,3,4,1)  # [B,T,H,W,6]
        eps = 1e-3
        half_l = (levels-1)*(1+eps)/2
        offset = torch.where(levels%2==0, 0.5, 0.0)
        shift = torch.atanh(offset/half_l)
        zb = torch.tanh(x+shift)*half_l - offset
        zhat = torch.round(zb)
        half_w = torch.floor(levels/2)
        codes = zhat/half_w
        digit = codes*half_w + half_w
        basis = torch.tensor([1,8,64,512,2560,12800], dtype=torch.float32)
        idx = (digit*basis).sum(-1)  # [B,T,H,W]
        return idx.to(torch.int32), codes  # codes = normalized continuous

def jit_body_f32(jit, patched):
    """Run the jit encoder conv-body (conv_in..conv_out) + quant_conv in f32 on a pre-patched f32 input."""
    e = jit.encoder
    x = e.conv_in.forward(patched)
    d = e.down
    x = getattr(d,"0").downsample.forward(getattr(getattr(d,"0").block,"1").forward(getattr(getattr(d,"0").block,"0").forward(x)))
    x = getattr(d,"1").downsample.forward(getattr(getattr(d,"1").block,"1").forward(getattr(getattr(d,"1").block,"0").forward(x)))
    x = getattr(getattr(d,"2").block,"1").forward(getattr(getattr(d,"2").block,"0").forward(x))
    x = e.mid.block_1.forward(x)
    x = getattr(e.mid.attn_1,"1").forward(getattr(e.mid.attn_1,"0").forward(x))
    x = e.mid.block_2.forward(x)
    x = e.norm_out.forward(x); x = x*torch.sigmoid(x)
    x = e.conv_out.forward(x)
    x = jit.quant_conv.forward(x)
    return x

if __name__=='__main__':
    import numpy as np
    jit = torch.jit.load('/home/hartsy/models_cosmos_dv/encoder.jit', map_location='cpu').eval()
    wav = jit.encoder.patcher3d.wavelets.float().clone()
    # fixed synthetic input
    g = torch.Generator().manual_seed(1234)
    T,H,W = 9, 64, 64
    x = (torch.rand(1,3,T,H,W, generator=g)*2-1)

    # ---- (1) validate transcription vs jit conv-body, using the JIT's own weights in f32 ----
    sd_jit = {k: v.float() for k,v in jit.state_dict().items()}
    ref_j = Ref(sd_jit); ref_j.wav = wav
    jitf = torch.jit.load('/home/hartsy/models_cosmos_dv/encoder.jit', map_location='cpu').eval().float()
    with torch.no_grad():
        patched = ref_j.patcher(x)                 # my f32 patcher
        cont_jitbody = jit_body_f32(jitf, patched) # jit's conv body in f32 on my patcher output
        cont_ref_j = ref_j.encode(x)               # my full transcription (jit weights f32)
    print("== transcription check (jit-weights f32) ==")
    print("  patcher out shape:", tuple(patched.shape))
    print("  my-body vs jit-body maxdiff:", (cont_ref_j-cont_jitbody).abs().max().item())
    idx_a,_ = ref_j.fsq(cont_ref_j); idx_b,_ = ref_j.fsq(cont_jitbody)
    print("  token match my-body vs jit-body:", (idx_a==idx_b).float().mean().item())

    # ---- (2) authoritative artifacts: network(=jit) weights in f32 (what C# loads) ----
    cont_ref = cont_ref_j
    idx_ref, codes_ref = ref_j.fsq(cont_ref)
    print("== network/jit-weights F32 reference (authoritative for C#) ==")
    print("  continuous shape:", tuple(cont_ref.shape), " indices shape:", tuple(idx_ref.shape))
    np.save('cosmos_input.npy', x.numpy().astype('float32'))
    np.save('cosmos_ref_continuous.npy', cont_ref.numpy().astype('float32'))  # pre-FSQ [B,6,T,H,W]
    np.save('cosmos_ref_indices.npy', idx_ref.numpy().astype('int32'))        # [B,T,H,W]
    np.save('cosmos_ref_codes.npy', codes_ref.permute(0,4,1,2,3).numpy().astype('float32'))  # dequant [B,6,T,H,W]
    print("saved npy artifacts; sample indices:", idx_ref.reshape(-1)[:12].tolist())
    # also emit as raw binary + a header for easy C# ingestion
    def dump(name, arr):
        arr = np.ascontiguousarray(arr)
        with open(name+'.hdr','w') as f: f.write(','.join(map(str,arr.shape))+'|'+str(arr.dtype))
        arr.tofile(name+'.bin')
    dump('cosmos_input', x.numpy().astype('float32'))
    dump('cosmos_ref_continuous', cont_ref.numpy().astype('float32'))
    dump('cosmos_ref_indices', idx_ref.numpy().astype('int32'))
    print("dumped .bin/.hdr for C#")
