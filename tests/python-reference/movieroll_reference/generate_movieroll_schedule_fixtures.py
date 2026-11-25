"""Regenerates tests/HartsyInference.Diffusion.Tests/Fixtures/movieroll_schedule_reference.json: a transcription of MovieRoll's flow-match sigma ladders and the rolling step-index wrapper (no diffusers needed).

Run in the CPU reference image and redirect stdout to the fixture:
  bash .claude/run-outputs/refenv/run.sh -v "<worktree>:/wt" -- python -I /wt/tests/python-reference/movieroll_reference/generate_movieroll_schedule_fixtures.py > <worktree>/tests/HartsyInference.Diffusion.Tests/Fixtures/movieroll_schedule_reference.json
"""
import json
import math

NUM_TRAIN = 1000


def shift_sigmas(t, shift):
    return [shift * x / (1 + (shift - 1) * x) for x in t]


def base(num_inference_steps, shift):
    # set_timesteps: sigmas = linspace(1, 0, N+1), then the sd3 shift; timesteps = sigmas[:-1] * N_train
    sigmas = [1 - i / num_inference_steps for i in range(num_inference_steps + 1)]
    sigmas = shift_sigmas(sigmas, shift)
    timesteps = [s * NUM_TRAIN for s in sigmas[:-1]]
    return sigmas, timesteps


def distilled(denoising_step_list, shift):
    # set_custom_timesteps: sigmas from list/N_train, shifted, zero appended; timesteps come from the SHIFTED sigmas
    sig = shift_sigmas([t / NUM_TRAIN for t in denoising_step_list], shift) + [0.0]
    timesteps = [s * NUM_TRAIN for s in sig[:-1]]
    return sig, timesteps


def step_table(num_inference_steps, window_size):
    spw = num_inference_steps // window_size
    orig = list(range(num_inference_steps))
    rows = [[orig[(i - w * spw) % num_inference_steps] for i in range(num_inference_steps)] for w in range(window_size)]
    return spw, rows


def step_index(rows, spw, window_size, num_inference_steps, g):
    step_idx = g % num_inference_steps
    offset = math.ceil(max(g - num_inference_steps + 1, 0) / spw) % window_size
    return [rows[(offset + k) % window_size][step_idx] for k in range(window_size)]


BASE_GLOBALS = list(range(0, 60, 7)) + [59, 60, 61, 90, 99, 100, 250]
out = {}
s, t = base(50, 5.0)
spw, rows = step_table(50, 5)
out["base"] = {"sigmas": s, "timesteps": t, "steps_per_window": spw,
               "index": [step_index(rows, spw, 5, 50, g) for g in BASE_GLOBALS],
               "index_globals": BASE_GLOBALS}
d_list = [1000, 800, 600, 400, 200]
s, t = distilled(d_list, 5.0)
spw, rows = step_table(5, 5)
out["distilled"] = {"sigmas": s, "timesteps": t, "steps_per_window": spw,
                    "index": [step_index(rows, spw, 5, 5, g) for g in range(0, 12)],
                    "index_globals": list(range(0, 12))}
print(json.dumps(out))
