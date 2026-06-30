"""EnCodec 24 kHz (Bark codec) DECODE parity oracle via transformers EncodecModel."""
import os, torch
from transformers import EncodecModel
from safetensors.torch import save_file

OUT = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)
m = EncodecModel.from_pretrained("facebook/encodec_24khz").eval()
print("norm", m.config.normalize, "sr", m.config.sampling_rate, "codebook", m.config.codebook_size)

x = torch.randn(1, 1, 24000)
with torch.no_grad():
    enc = m.encode(x, bandwidth=6.0)            # 8 codebooks @ 6 kbps
    codes = enc.audio_codes                     # [n_chunks, B, nQ, T]
    audio = m.decode(codes, enc.audio_scales)[0]  # [B, 1, S]
ac = codes[0]                                   # [B, nQ, T]
nq = ac.shape[1]; T = ac.shape[2]
print("codes", tuple(ac.shape), "audio", tuple(audio.shape), "rms", float(audio.pow(2).mean().sqrt()))

sd = {k: v.detach().contiguous().float().cpu() for k, v in m.state_dict().items()}
save_file(sd, os.path.join(OUT, "encodec24_weights.safetensors"))
# engine wants [nQ, B, T]
save_file({"codes": ac.permute(1, 0, 2).to(torch.int32).contiguous(), "audio_out": audio.float().contiguous(),
           "nq": torch.tensor([nq], dtype=torch.int32), "t": torch.tensor([T], dtype=torch.int32)},
          os.path.join(OUT, "encodec24_ref_io.safetensors"))
print("wrote encodec24 weights + ref; nq", nq, "T", T)
