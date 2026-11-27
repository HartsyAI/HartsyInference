"""Regenerates the MovieRoll schedule fixture from MovieRoll's own scheduler code, not a transcription.

Imports `FlowMatchDiscreteScheduler` and `StreamingSchedulerWrapper` from the upstream checkout (Tencent-Hunyuan/MovieRoll @ 0b5f73e) and
records their sigmas, timesteps and per-frame step indices in the same JSON shape as generate_movieroll_schedule_fixtures.py, so the two can be
compared. Needs torch and diffusers, and the upstream checkout on PYTHONPATH:

  set PYTHONPATH=<MovieRoll-upstream>
  python -I generate_movieroll_schedule_fixtures_upstream.py > movieroll_schedule_upstream.json
"""
import json

from movieroll.commons.streaming_config import StreamingConfig
from movieroll.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from movieroll.schedulers.scheduling_flow_match_streaming import StreamingSchedulerWrapper

BASE_GLOBALS = list(range(0, 60, 7)) + [59, 60, 61, 90, 99, 100, 250]


def wrapper_indices(scheduler, num_inference_steps, window_size, globals_):
    wrapper = StreamingSchedulerWrapper(StreamingConfig(), scheduler, window_size, num_inference_steps)
    return wrapper.steps_per_window, [wrapper.get_step_index(g).tolist() for g in globals_]


out = {}

# Base chain: 50 steps, shift 5.0 (set_timesteps, the sd3 shift as upstream applies it).
base = FlowMatchDiscreteScheduler(num_train_timesteps=1000, shift=5.0)
base.set_timesteps(50)
spw, index = wrapper_indices(base, 50, 5, BASE_GLOBALS)
out["base"] = {
    "sigmas": base.sigmas.tolist(),
    "timesteps": base.timesteps.tolist(),
    "steps_per_window": spw,
    "index": index,
    "index_globals": BASE_GLOBALS,
}

# Distilled chain: [1000, 800, 600, 400, 200], shift 5.0 (set_custom_timesteps).
distilled = FlowMatchDiscreteScheduler(num_train_timesteps=1000, shift=5.0)
distilled.set_custom_timesteps([1000, 800, 600, 400, 200])
spw, index = wrapper_indices(distilled, 5, 5, list(range(0, 12)))
out["distilled"] = {
    "sigmas": distilled.sigmas.tolist(),
    "timesteps": distilled.timesteps.tolist(),
    "steps_per_window": spw,
    "index": index,
    "index_globals": list(range(0, 12)),
}

print(json.dumps(out))
