"""Fish-Speech 1.5 firefly codec DECODE per-stage dump for C# parity debugging.
Reuses indices from firefly_ref_io.safetensors and captures every decode sub-stage:
FSQ dequant (pre-upsample), each quantizer upsample stage, head conv_pre, each head
up+resblock, and final pre-tanh + audio."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file, load_file
from einops import rearrange
import torch.nn.functional as F
from fsmod.firefly import HiFiGANGenerator
from fsmod.fsq import DownsampleFiniteScalarQuantize

OUT = os.path.dirname(os.path.abspath(__file__))
cp = hf_hub_download("fishaudio/fish-speech-1.5", "firefly-gan-vq-fsq-8x1024-21hz-generator.pth")
sd = torch.load(cp, map_location="cpu", weights_only=True)
if "state_dict" in sd: sd = sd["state_dict"]
if "generator" in sd: sd = sd["generator"]

head = HiFiGANGenerator(hop_length=512, upsample_rates=[8, 8, 2, 2, 2], upsample_kernel_sizes=[16, 16, 4, 4, 4],
    resblock_kernel_sizes=[3, 7, 11], resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    num_mels=512, upsample_initial_channel=512, pre_conv_kernel_size=13, post_conv_kernel_size=13)
quant = DownsampleFiniteScalarQuantize(input_dim=512, n_groups=8, n_codebooks=1, levels=[8, 5, 5, 5], downsample_factor=[2, 2])
head.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("head.")}, strict=False)
quant.load_state_dict({k[10:]: v for k, v in sd.items() if k.startswith("quantizer.")}, strict=False)
head.eval(); quant.eval()

ref = load_file(os.path.join(OUT, "firefly_ref_io.safetensors"))
indices = ref["indices"].to(torch.long)   # [1,8,L]
d = {}
with torch.no_grad():
    g = quant.residual_fsq.groups
    idx = rearrange(indices, "b (g r) l -> g b l r", g=g)
    z_q = quant.residual_fsq.get_output_from_indices(idx)   # [b, l, 512]
    d["fsq_out"] = z_q.mT.float().contiguous()              # [1,512,L]
    x = z_q.mT
    for i, stage in enumerate(quant.upsample):
        x = stage(x)
        d[f"q_up_{i}"] = x.float().contiguous()
    d["z_q"] = x.float().clone().contiguous()               # [1,512,4L]

    # head
    h = head.conv_pre(x)
    d["h_conv_pre"] = h.float().contiguous()
    for i in range(head.num_upsamples):
        h = F.silu(h)
        h = head.ups[i](h)
        d[f"h_up_{i}"] = h.float().contiguous()
        h = head.resblocks[i](h)
        d[f"h_res_{i}"] = h.float().contiguous()
    h = head.activation_post(h)
    h = head.conv_post(h)
    d["h_pre_tanh"] = h.float().contiguous()
    audio = torch.tanh(h)
    d["audio_out"] = audio.float().contiguous()

for k, v in d.items():
    print(f"{k:14s} {tuple(v.shape)}  mean={float(v.mean()):+.4f} std={float(v.std()):.4f}")
save_file(d, os.path.join(OUT, "firefly_stages.safetensors"))
print("wrote firefly_stages.safetensors")
