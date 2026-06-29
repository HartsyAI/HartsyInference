"""Dump HeartCodec per-component + full-decode reference IO for the C# parity tests.

Loads the real HeartCodec-oss-20260123 checkpoint via the upstream heartlib (cloned at /tmp/heartlib),
casts the (small, ~300M) codec to f32, and dumps fixed-input -> fixed-output tensors to a .safetensors
that the C# HeartCodecParityTests reads. All randomness (initial latent) is made deterministic with a
fixed torch seed so the C# side can reproduce the exact noise.

Components dumped:
  rvq_codes        [Q=8, T]  int64    fixed input code grid
  rvq_out          [T, 512]           ResidualVQ.get_output_from_indices(codes.T) -> project_out
  cond_emb         [2T, 512]          cond_feature_emb + nearest x2 interpolation (the estimator mu cond)
  est_x            [2T, 256]          a fixed estimator input latent (deterministic randn)
  est_t            scalar             a fixed flow time
  est_v            [2T, 256]          estimator velocity = LlamaTransformer(cat[x, incontext(=0), mu], t)
  scalar_in        [256, L]           a fixed deterministic scalar_model.decode input (latent.T)
  scalar_out       [L*960]            scalar_model.decode(scalar_in) waveform (1 band)
  full_codes       [Q=8, T]  int64    fixed grid for the full detokenize
  full_wav         [N]                codec.detokenize(full_codes) (deterministic noise) -> 48k waveform
"""
import os, sys, glob, json, types, importlib.util, math
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCHINDUCTOR_DISABLE'] = '1'
import torch, numpy as np
from safetensors.torch import save_file
torch.set_grad_enabled(False)

HEARTLIB = '/tmp/heartlib/src/heartlib'
OUT = '/tmp/heartmula_ref/heartcodec_ref.safetensors'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

pkg = types.ModuleType('hc'); pkg.__path__ = []; sys.modules['hc'] = pkg
mpkg = types.ModuleType('hc.models'); mpkg.__path__ = []; sys.modules['hc.models'] = mpkg
load('hc.models.transformer', f'{HEARTLIB}/heartcodec/models/transformer.py')
load('hc.models.sq_codec', f'{HEARTLIB}/heartcodec/models/sq_codec.py')
load('hc.models.flow_matching', f'{HEARTLIB}/heartcodec/models/flow_matching.py')
cfgmod = load('hc.configuration_heartcodec', f'{HEARTLIB}/heartcodec/configuration_heartcodec.py')
# patch the relative imports inside modeling_heartcodec
mod_src = open(f'{HEARTLIB}/heartcodec/modeling_heartcodec.py').read()
mod_src = mod_src.replace('from .models.flow_matching import FlowMatching',
                          'from hc.models.flow_matching import FlowMatching')
mod_src = mod_src.replace('from .models.sq_codec import ScalarModel',
                          'from hc.models.sq_codec import ScalarModel')
mod_src = mod_src.replace('from .configuration_heartcodec import HeartCodecConfig',
                          'from hc.configuration_heartcodec import HeartCodecConfig')
modmod = types.ModuleType('hc.modeling_heartcodec')
exec(compile(mod_src, 'modeling_heartcodec.py', 'exec'), modmod.__dict__)
sys.modules['hc.modeling_heartcodec'] = modmod
HeartCodec = modmod.HeartCodec
HeartCodecConfig = cfgmod.HeartCodecConfig

snap = glob.glob(os.path.expanduser(
    '~/.cache/huggingface/hub/models--HeartMuLa--HeartCodec-oss-20260123/snapshots/*'))[0]
cfg = HeartCodecConfig(**json.load(open(snap + '/config.json')))
with torch.device('meta'):
    model = HeartCodec(cfg)
sd = {}
for st in sorted(glob.glob(snap + '/*.safetensors')):
    from safetensors import safe_open
    with safe_open(st, framework='pt') as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k).float()
missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
print('missing', len(missing), missing[:8])
print('unexpected', len(unexpected), unexpected[:8])
model = model.eval()

fm = model.flow_matching
sm = model.scalar_model
out = {}

# ---- 1. RVQ decode ----
T = 24
g = torch.Generator().manual_seed(1234)
codes = torch.randint(0, cfg.codebook_size, (cfg.num_quantizers, T), generator=g, dtype=torch.long)  # [Q,T]
# vq_embed.get_output_from_indices expects indices [b, n, q]
idx = codes.transpose(0, 1).unsqueeze(0)  # [1, T, Q]
rvq_out = fm.vq_embed.get_output_from_indices(idx)  # [1, T, 512]
out['rvq_codes'] = codes.to(torch.int64).contiguous()
out['rvq_out'] = rvq_out[0].contiguous()  # [T, 512]
print('rvq_out', rvq_out.shape)

# ---- 2. cond emb (cond_feature_emb + nearest x2) ----
cond = fm.cond_feature_emb(rvq_out)  # [1, T, 512]
cond = torch.nn.functional.interpolate(cond.permute(0, 2, 1), scale_factor=2, mode='nearest').permute(0, 2, 1)
out['cond_emb'] = cond[0].contiguous()  # [2T, 512]
print('cond_emb', cond.shape)

# ---- 3. estimator velocity (single forward, guidance_scale==1 path: cat[x, incontext, mu]) ----
T2 = 2 * T
gx = torch.Generator().manual_seed(777)
est_x = torch.randn(1, T2, cfg.out_channels, generator=gx)  # [1, 2T, 256]
incontext = torch.zeros(1, T2, cfg.out_channels)
# mu here = cond masked by zero_cond? In inference_codes, mask>0.5 everywhere within latent_length.
# For the parity dump use the raw cond as mu (the conditioning seen by estimator when all-active).
est_t = torch.tensor([0.3])
est_in = torch.cat([est_x, incontext, cond], 2)  # [1, 2T, 256+256+512=1024]
est_v = fm.estimator(est_in, timestep=est_t)  # [1, 2T, 256]
out['est_x'] = est_x[0].contiguous()
out['est_cond'] = cond[0].contiguous()
out['est_t'] = est_t.contiguous()
out['est_v'] = est_v[0].contiguous()
print('est_v', est_v.shape)

# ---- 4. scalar_model.decode ----
gl = torch.Generator().manual_seed(555)
L = 40
scalar_in = torch.randn(1, cfg.out_channels // 2, L, generator=gl)  # [1, 128, L]  (latent after reshape)
scalar_out = sm.decode(scalar_in)  # [1, 1, L*960]
out['scalar_in'] = scalar_in[0].contiguous()  # [128, L]
out['scalar_out'] = scalar_out[0, 0].contiguous()  # [L*960]
print('scalar_out', scalar_out.shape)

# ---- 5. full single-segment decode (the meat of detokenize, without the overlap-add segmentation).
# Replicates: inference_codes(one segment) -> reshape latent [B,2T,256]->[2B,T,128] -> scalar.decode.
# Uses a fixed initial-latent noise so the C# side can reproduce the exact ODE start. The CFM ODE here
# integrates from t=0..1 in num_steps with guidance_scale (CFG), scenario start (no incontext).
gf = torch.Generator().manual_seed(2024)
Tf = 24
full_codes = torch.randint(0, cfg.codebook_size, (cfg.num_quantizers, Tf), generator=gf, dtype=torch.long)
num_steps = 10
guidance = 1.25
# initial latent noise [B=1, 2*Tf, 256] (the estimator runs at 2x the code frame rate)
gn = torch.Generator().manual_seed(31415)
init_noise = torch.randn(1, 2 * Tf, cfg.out_channels, generator=gn)
out['full_init_noise'] = init_noise[0].contiguous()  # [2Tf, 256]

# Build mu exactly like inference_codes with latent_masks all active (latent_length = 2*Tf, no incontext).
idxf = full_codes.transpose(0, 1).unsqueeze(0)  # [1, Tf, Q]
qfe = fm.vq_embed.get_output_from_indices(idxf)
qfe = fm.cond_feature_emb(qfe)
qfe = torch.nn.functional.interpolate(qfe.permute(0, 2, 1), scale_factor=2, mode='nearest').permute(0, 2, 1)
mu = qfe  # all active
incontext = torch.zeros_like(init_noise)
# solve_euler with guidance>1 (CFG branch)
x = init_noise.clone()
t_span = torch.linspace(0, 1, num_steps + 1)
t = t_span[0]; dt = t_span[1] - t_span[0]
for step in range(1, len(t_span)):
    dphi = fm.estimator(
        torch.cat([torch.cat([x, x], 0),
                   torch.cat([incontext, incontext], 0),
                   torch.cat([torch.zeros_like(mu), mu], 0)], 2),
        timestep=t.unsqueeze(-1).repeat(2))
    un, co = dphi.chunk(2, 0)
    dphi = un + guidance * (co - un)
    x = x + dt * dphi
    t = t + dt
    if step < len(t_span) - 1:
        dt = t_span[step + 1] - t
latents = x  # [1, 2Tf, 256]
# reshape [B, 2Tf, 256] -> [B, 2Tf, 2, 128] -> permute [B,2,2Tf,128] -> [2B, 2Tf, 128]
lat = latents.reshape(1, latents.shape[1], 2, latents.shape[2] // 2).permute(0, 2, 1, 3)
lat = lat.reshape(2, latents.shape[1], latents.shape[2] // 2)  # [2, 2Tf, 128]
cur = sm.decode(lat.transpose(1, 2)).squeeze(1)  # [2, samples]
out['full_codes'] = full_codes.to(torch.int64).contiguous()
out['full_latents'] = latents[0].contiguous()  # [2Tf, 256]
out['full_wav'] = cur.contiguous()  # [2, samples]
print('full_latents', latents.shape, 'full_wav', cur.shape)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
save_file({k: v.contiguous() for k, v in out.items()}, OUT)
print('SAVED', OUT)
for k, v in out.items():
    print(' ', k, tuple(v.shape), v.dtype)
