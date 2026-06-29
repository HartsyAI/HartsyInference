"""DAC 44.1 kHz decode parity oracle (descript-audio-codec)."""
import os, torch, dac
from safetensors.torch import save_file

OUT = os.path.dirname(os.path.abspath(__file__))
p = dac.utils.download(model_type="44khz")
m = dac.DAC.load(p).eval()
print("n_codebooks", m.n_codebooks, "sr", m.sample_rate, "hop", m.hop_length)

sd = {k: v.detach().contiguous().float().cpu() for k, v in m.state_dict().items()}
save_file(sd, os.path.join(OUT, "dac_44khz.safetensors"))
print("weights", len(sd))

torch.manual_seed(1)
B, nq, T = 1, m.n_codebooks, 20
codes = torch.randint(0, m.codebook_size, (B, nq, T), dtype=torch.long)
with torch.no_grad():
    z, _, _ = m.quantizer.from_codes(codes)
    audio = m.decode(z)
print("z", tuple(z.shape), "audio", tuple(audio.shape), "rms", float(audio.pow(2).mean().sqrt()))

# save codes as [nQ, B, T] to match the C# Dac.Decode convention
ref = {
    "codes": codes.permute(1, 0, 2).to(torch.int32).cpu().contiguous(),   # [nQ, B, T]
    "audio_out": audio.float().cpu().contiguous(),
    "nq": torch.tensor([nq], dtype=torch.int32),
    "tframes": torch.tensor([T], dtype=torch.int32),
}
save_file(ref, os.path.join(OUT, "dac_ref_io.safetensors"))
print("wrote ref: nq", nq, "T", T)
