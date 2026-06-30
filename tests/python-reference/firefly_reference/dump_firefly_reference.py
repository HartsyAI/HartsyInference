"""Fish-Speech 1.5 firefly-gan-vq codec DECODE parity oracle (indices -> 44.1 kHz audio)."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for `import fsmod`
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from fsmod.firefly import HiFiGANGenerator
from fsmod.fsq import DownsampleFiniteScalarQuantize

OUT = os.path.dirname(os.path.abspath(__file__))
cp = hf_hub_download("fishaudio/fish-speech-1.5", "firefly-gan-vq-fsq-8x1024-21hz-generator.pth")
sd = torch.load(cp, map_location="cpu", weights_only=True)
if "state_dict" in sd: sd = sd["state_dict"]
if "generator" in sd: sd = sd["generator"]
print("top prefixes:", sorted({k.split('.')[0] for k in sd}))

head = HiFiGANGenerator(hop_length=512, upsample_rates=[8, 8, 2, 2, 2], upsample_kernel_sizes=[16, 16, 4, 4, 4],
    resblock_kernel_sizes=[3, 7, 11], resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    num_mels=512, upsample_initial_channel=512, pre_conv_kernel_size=13, post_conv_kernel_size=13)
quant = DownsampleFiniteScalarQuantize(input_dim=512, n_groups=8, n_codebooks=1, levels=[8, 5, 5, 5], downsample_factor=[2, 2])

head_sd = {k[len("head."):]: v for k, v in sd.items() if k.startswith("head.")}
quant_sd = {k[len("quantizer."):]: v for k, v in sd.items() if k.startswith("quantizer.")}
mh = head.load_state_dict(head_sd, strict=False); mq = quant.load_state_dict(quant_sd, strict=False)
print("head missing", len(mh.missing_keys), "unexpected", len(mh.unexpected_keys), "| quant missing", len(mq.missing_keys), "unexpected", len(mq.unexpected_keys))
head.eval(); quant.eval()

torch.manual_seed(0)
L = 16
vocab = 8 * 5 * 5 * 5   # 1000
indices = torch.randint(0, vocab, (1, 8, L))      # [b, g*r=8, l]
with torch.no_grad():
    z_q = quant.decode(indices)                   # [1, 512, L*4]
    audio = head(z_q)                             # [1, 1, S]
print("indices", tuple(indices.shape), "z_q", tuple(z_q.shape), "audio", tuple(audio.shape), "rms", float(audio.pow(2).mean().sqrt()))

# also dump the head/quantizer weights so the C# loads identical tensors
w = {f"head.{k}": v.float().contiguous() for k, v in head_sd.items()}
w.update({f"quantizer.{k}": v.float().contiguous() for k, v in quant_sd.items()})
save_file(w, os.path.join(OUT, "firefly_weights.safetensors"))
save_file({"indices": indices.to(torch.int32).contiguous(), "z_q": z_q.float().contiguous(), "audio_out": audio.float().contiguous()},
          os.path.join(OUT, "firefly_ref_io.safetensors"))
print("wrote firefly weights + ref")
