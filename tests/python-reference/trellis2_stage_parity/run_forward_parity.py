"""The structure decoder and one texture-flow forward of upstream TRELLIS.2 on the CPU (fp32) against the C# forward on the same inputs.

Usage: python -I run_forward_parity.py <TRELLIS.2 clone> <models dir> <shape decoder path without extension> <capture dir> [--flow-only] [--bf16]
Reads forward.tns (written by csharp/Forward). --bf16 runs the flow in bfloat16 as its checkpoint config says; the default is float32.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import setup, load_model, read_tns, stats, timed  # noqa: E402

upstream, models, shape_vae, cap = sys.argv[1:5]
torch = setup(upstream, models, shape_vae)
from safetensors.torch import load_file  # noqa: E402
from trellis2 import models as upstream_models  # noqa: E402
from trellis2.modules import sparse as sp  # noqa: E402

d = read_tns(f"{cap}/forward.tns")
print("threads", torch.get_num_threads(), flush=True)

if "--flow-only" not in sys.argv:
    ss = load_model(os.path.join(models, "ss_dec_conv3d_16l8_fp16"))
    out = timed("structure decoder", lambda: ss(torch.from_numpy(d["ss_z"])))
    print(f"  C# {float(d['ss_ms'][0]) / 1000:.2f}s", flush=True)
    stats("structure decoder occupancy logits (1x1x64^3)", d["ss_out"], out.numpy())
    del ss

path = os.path.join(models, "slat_flow_imgshape2tex_dit_1_3B_512_bf16")
if "--bf16" in sys.argv:
    cfg = json.load(open(path + ".json"))
    flow = getattr(upstream_models, cfg["name"])(**cfg["args"])
    flow.load_state_dict(load_file(path + ".safetensors"), strict=False)
    flow.eval()
else:
    flow = load_model(path)
coords = torch.from_numpy(d["flow_coords"]).int()
x = sp.SparseTensor(torch.from_numpy(d["flow_x"][0]), coords)
shape = sp.SparseTensor(torch.from_numpy(d["flow_shape_norm"][0]), coords)
cond = torch.from_numpy(d["flow_cond"])[None]
v = timed(f"texture flow forward ({'bf16' if '--bf16' in sys.argv else 'fp32'}, {coords.shape[0]} tokens)",
          lambda: flow(x, torch.tensor([500.0]), cond, concat_cond=shape))
print(f"  C# {float(d['flow_ms'][0]) / 1000:.2f}s", flush=True)
stats("texture flow velocity (one step, t = 500)", d["flow_out"][0], v.feats.float().numpy())
