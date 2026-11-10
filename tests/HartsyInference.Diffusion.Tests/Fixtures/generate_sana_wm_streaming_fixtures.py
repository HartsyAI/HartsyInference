"""Fixtures for the SANA-WM streaming unit tests, generated from upstream NVlabs/Sana b3bad956 itself (not a transcription):
chunk boundaries from chunk_utils.chunk_index_from_chunk_size and SelfForcingFlowEuler.create_autoregressive_segments, and
the distilled schedule and per-token Euler step from diffusers FlowMatchEulerDiscreteScheduler as SelfForcingFlowEulerCamCtrl
drives it (shift 1.0, explicit sigmas t/1000, scheduler.step(-velocity, t, x, per_token_timesteps=...)).

usage: <upstream venv python> -I generate_sana_wm_streaming_fixtures.py <upstream root> <out.json>
"""
import json
import sys
import types

root, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
import torch  # noqa: E402
from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402
from diffusion.utils.chunk_utils import chunk_index_from_chunk_size  # noqa: E402
from diffusion.scheduler.self_forcing_flow_euler_sampler import SelfForcingFlowEuler  # noqa: E402

fixture = {"chunk_index": [], "ar_segments": [], "schedule": {}, "step": {}}
for strategy in ("uniform", "first_frame", "first_plus_one", "first_chunk_plus_one"):
    for base in (3, 4):
        for total in range(1, 65):
            try:
                starts = chunk_index_from_chunk_size(total, base, strategy)
                fixture["chunk_index"].append({"strategy": strategy, "base": base, "total": total, "starts": [int(s) for s in starts]})
            except Exception as exc:  # unknown strategy name or invalid total: record the refusal
                fixture["chunk_index"].append({"strategy": strategy, "base": base, "total": total, "error": type(exc).__name__})
for base in (3, 4):
    for total in range(base + 1, 65):
        dummy = types.SimpleNamespace(base_chunk_frames=base)
        fixture["ar_segments"].append({"base": base, "total": total,
                                       "bounds": [int(b) for b in SelfForcingFlowEuler.create_autoregressive_segments(dummy, total)]})

steps = [1000, 960, 889, 727, 0]
sched = FlowMatchEulerDiscreteScheduler(shift=1.0)
sched.set_timesteps(sigmas=[t / 1000.0 for t in steps[:-1]], device="cpu")
fixture["schedule"] = {"steps": steps, "timesteps": sched.timesteps.tolist(), "sigmas": sched.sigmas.tolist()}

# One full chunk walk: B=1, C=2 channels, F=3 frames, S=4 tokens; frame 0 clean (timestep 0) as the sampler marks it.
g = torch.Generator().manual_seed(7)
B, C, F, S = 1, 2, 3, 4
x = torch.randn(B, C, F, S, generator=g)
clean = torch.zeros(B, F)
clean[:, 0] = 1.0
traj = {"x0": x.flatten().tolist(), "velocities": [], "after": []}
sched.set_timesteps(sigmas=[t / 1000.0 for t in steps[:-1]], device="cpu")
for i, t in enumerate(sched.timesteps):
    v = torch.randn(B, C, F, S, generator=g)
    frame_t = (1.0 - clean) * t
    per_token = frame_t[:, :, None].expand(B, F, S).reshape(B, -1)
    out_x = sched.step(-v.reshape(B, C, -1).transpose(1, 2), t, x.reshape(B, C, -1).transpose(1, 2),
                       per_token_timesteps=per_token, return_dict=False)[0]
    x = out_x.transpose(1, 2).reshape(B, C, F, S)
    traj["velocities"].append(v.flatten().tolist())
    traj["after"].append(x.flatten().tolist())
fixture["step"] = {"shape": [B, C, F, S], "clean_frames": [0], **traj}
# The softmax KV window: upstream's own _accumulate_softmax_kv_cache over chunks 0..6, each chunk's saved K a tensor filled
# with its chunk id (length = its latent frames), so the accumulated K reads back which chunks the window kept, in order.
from diffusion.scheduler import self_forcing_flow_euler_sampler as sfs  # noqa: E402

fixture["window"] = []
for cached in (-1, 1, 2, 3):
    for sink in (False, True):
        bounds = [0, 4, 7, 10, 13, 16, 19, 22]
        dummy = types.SimpleNamespace(num_model_blocks=1, num_cached_blocks=cached, sink_token=sink, _chunk_indices=bounds)
        cache = [[[None] * sfs._NUM_CACHE_SLOTS] for _ in range(len(bounds) - 1)]
        for c in range(len(bounds) - 1):
            cur, accumulated, sink_num, cached_frames = sfs.SelfForcingFlowEulerCamCtrl._accumulate_softmax_kv_cache(dummy, cache, c)
            k = cur[0][0]
            ids = [] if k is None or c == 0 else [int(v) for v in k[0, 0, :, 0].tolist()]
            kept = [ids[i] for i in range(len(ids)) if i == 0 or ids[i] != ids[i - 1]]
            fixture["window"].append({"cached": cached, "sink": sink, "chunk": c, "kept_frames": ids, "kept_chunks": kept,
                                      "sink_num": int(sink_num), "cached_frames": int(cached_frames)})
            n = bounds[c + 1] - bounds[c]
            saved = [None] * sfs._NUM_CACHE_SLOTS
            saved[0] = torch.full((1, 1, n, 1), float(c))
            saved[1] = torch.full((1, 1, n, 1), float(c))
            saved[sfs._SLOT_TYPE_FLAG] = torch.tensor(0.0)
            cache[c] = [saved]
with open(out, "w") as f:
    json.dump(fixture, f)
print("wrote", out)
