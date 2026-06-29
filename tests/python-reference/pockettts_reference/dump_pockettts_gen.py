"""Pocket-TTS deterministic end-to-end AR loop (Phase D) parity oracle.

Replicates the reference forward math (input_linear + bos_emb + transformer + out_norm + flow_net lsd_decode)
in a full-prefix loop with INJECTED fixed noise (so it is deterministic), then mimi-decodes the latents to
audio. Dumps the inputs (text embeddings + noise) and expected outputs (latents + audio).
"""
import os, torch
from safetensors.torch import save_file
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import init_states

OUT = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)
model = TTSModel.load_model(language="english")
model.eval()
flm = model.flow_lm
mimi = model.mimi

Ttext, N = 5, 8
text_emb = torch.randn(1, Ttext, flm.dim)        # stand-in for conditioner output
noises = torch.randn(N, flm.ldim)                # fixed per-frame noise

@torch.no_grad()
def step(latents):
    seq = torch.stack([flm.bos_emb] + latents, dim=0)[None]   # [1, f+1, 32]
    input_ = flm.input_linear(seq)
    full = torch.cat([text_emb, input_], dim=1)
    tout = flm.out_norm(flm.transformer(full, None))
    return tout[:, -1]                                         # [1,1024]

latents = []
with torch.no_grad():
    for f in range(N):
        hidden = step(latents)
        noise = noises[f][None]                                # [1,32]
        flow_dir = flm.flow_net(hidden, torch.zeros(1, 1), torch.ones(1, 1), noise)
        latents.append((noise + flow_dir)[0])                  # lsd_decode, 1 step
    lat_seq = torch.stack(latents, dim=0).transpose(0, 1)[None]  # [1,32,N]
    emb = mimi.quantizer(lat_seq)
    mimi_state = init_states(mimi, batch_size=1, sequence_length=10000)
    audio = mimi.decode_from_latent(emb, mimi_state)
print("lat_seq", tuple(lat_seq.shape), "audio", tuple(audio.shape), "rms", float(audio.pow(2).mean().sqrt()))

save_file({
    "text_emb": text_emb.float().contiguous(),
    "noises": noises.float().contiguous(),
    "latents": lat_seq.float().contiguous(),
    "audio_out": audio.float().contiguous(),
}, os.path.join(OUT, "pockettts_gen_ref.safetensors"))
print("wrote gen ref; N", N)
