"""Pocket-TTS FlowLM backbone (Phase B) + flow_net (Phase C) parity oracle."""
import os, torch
from safetensors.torch import save_file
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import init_states

OUT = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)
model = TTSModel.load_model(language="english")
model.eval()
flm = model.flow_lm

ref = {}

# ── Phase B: transformer backbone (input_ [1,S,1024] -> out_norm(transformer(input_))) ──
S = 24
inp = torch.randn(1, S, flm.dim)
state = init_states(flm, batch_size=1, sequence_length=10000)
with torch.no_grad():
    tout = flm.transformer(inp, state)
    tout = flm.out_norm(tout)
ref["b_input"] = inp.float().contiguous()
ref["b_out"] = tout.float().contiguous()
print("Phase B: input", tuple(inp.shape), "out", tuple(tout.shape))

# ── Phase C: flow_net(cond[1,1024], s[1,1], t[1,1], x_t[1,32]) -> velocity[1,32] ──
cond = torch.randn(1, flm.dim)
x_t = torch.randn(1, flm.ldim)
s = torch.zeros(1, 1)
t = torch.ones(1, 1)
with torch.no_grad():
    vel = flm.flow_net(cond, s, t, x_t)
ref["c_cond"] = cond.float().contiguous()
ref["c_xt"] = x_t.float().contiguous()
ref["c_s"] = s.float().contiguous()
ref["c_t"] = t.float().contiguous()
ref["c_vel"] = vel.float().contiguous()
print("Phase C: cond", tuple(cond.shape), "x_t", tuple(x_t.shape), "vel", tuple(vel.shape))

save_file(ref, os.path.join(OUT, "pockettts_flowlm_ref.safetensors"))
print("wrote flowlm ref")
