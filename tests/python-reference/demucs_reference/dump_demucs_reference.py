"""HTDemucs (v4) source-separation parity oracle via the `demucs` package. Dumps a fixed input waveform, the
full 4-stem output, and the model weights (as safetensors for the C# HtDemucs). Also dumps the spectrogram
front-end (model._spec) as an isolated component for stage debugging."""
import os, torch
from demucs.pretrained import get_model
from safetensors.torch import save_file

OUT = os.path.dirname(os.path.abspath(__file__))
m = get_model("htdemucs").models[0].eval()
m.use_train_segment = False        # process the raw input length (no 7.8s segment pad/crop) for clean model parity
print("HTDemucs sources", m.sources, "nfft", m.nfft, "depth", m.depth)

torch.manual_seed(0)
C = len(m.audio_channels) if hasattr(m, "audio_channels") and not isinstance(m.audio_channels, int) else m.audio_channels
L = int(m.samplerate * 1.0)            # 1 s
wav = torch.randn(1, C, L) * 0.1
with torch.no_grad():
    # Isolate the raw model from the apply_model norm wrapper: feed a fixed input, dump the raw output.
    x = wav
    out = m(x)                          # [1, sources, C, L]
    spec = m._spec(x)                   # [1, C, Fq, T] complex (or cac depending on version)
print("out", tuple(out.shape), "spec", tuple(spec.shape), "spec dtype", spec.dtype)

d = {"wav": x.float().contiguous(), "stems": out.float().contiguous()}
# spec may be complex; store real+imag stacked on a new last axis if so.
if torch.is_complex(spec):
    d["spec_re"] = spec.real.float().contiguous()
    d["spec_im"] = spec.imag.float().contiguous()
else:
    d["spec"] = spec.float().contiguous()
save_file(d, os.path.join(OUT, "demucs_ref_io.safetensors"))

sd = {k: v.float().contiguous() for k, v in m.state_dict().items()}
save_file(sd, os.path.join(OUT, "demucs_weights.safetensors"))
print("wrote demucs ref + weights:", len(sd), "tensors")
