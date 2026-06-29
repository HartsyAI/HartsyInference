"""Pocket-TTS Mimi continuous-latent DECODE parity oracle (latent[32] -> audio).

Loads the real model (falls back to the ungated without-voice-cloning weights, no token needed),
dumps the mimi.* weights + a fixed 32-dim latent and its decoded waveform.
"""
import os, torch
from safetensors.torch import save_file
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import init_states

OUT = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)

model = TTSModel.load_model(language="english")
model.eval()
mimi = model.mimi
print("frame_rate", mimi.frame_rate, "encoder_frame_rate", mimi.encoder_frame_rate, "sr", mimi.sample_rate)

# dump full state dict (C# will pick the mimi.* subtree)
sd = {k: v.detach().contiguous().float().cpu() for k, v in model.state_dict().items()}
save_file(sd, os.path.join(OUT, "pockettts_english.safetensors"))
print("weights", len(sd), "mimi keys", sum(1 for k in sd if k.startswith("mimi.")))

T = 20
latent = torch.randn(1, 32, T)   # [B, inner_dim, T] continuous flow latent
with torch.no_grad():
    mimi_state = init_states(mimi, batch_size=1, sequence_length=10000)
    emb = mimi.quantizer(latent)                      # output_proj 32 -> 512
    audio = mimi.decode_from_latent(emb, mimi_state)
print("latent", tuple(latent.shape), "emb", tuple(emb.shape), "audio", tuple(audio.shape),
      "rms", float(audio.pow(2).mean().sqrt()))

ref = {
    "latent": latent.float().cpu().contiguous(),       # [1,32,T]
    "audio_out": audio.float().cpu().contiguous(),     # [1,1,S]
    "tframes": torch.tensor([T], dtype=torch.int32),
}
save_file(ref, os.path.join(OUT, "pockettts_ref_io.safetensors"))
print("wrote ref")
