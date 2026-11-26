"""Tiny-config transcription of MovieRoll's rolling loop at 0b5f73e, for parity with MovieRollSampler.

The loop follows `MovieRollStreamingPipeline._generate_strategy_prompt_switch` (base chain, CFG), its `_pop_and_output_clean_chunk`,
`_kv_recompute`, `_select_recompute_chunks`, `_run_inner_denoising_loop` and `FlowMatchStreamingScheduler.step` / `StreamingSchedulerWrapper`,
and `MovieRollDmdPipeline`'s rolling loop (distilled chain: one step per round, no CFG, a custom ladder). Single event, no i2v.

Run in the CPU reference image. The MovieRoll checkout must be at 0b5f73e and mounted read-only at /src:

  bash .claude/run-outputs/refenv/run.sh -v "<worktree>:/wt" -v "<MovieRoll 0b5f73e>:/src:ro" -- \
      python -I /wt/tests/python-reference/movieroll_reference/dump_rolling_tiny.py \
      --weights /wt/tests/HartsyInference.Diffusion.Tests/Fixtures/hy15_dit_tiny.safetensors \
      --out /wt/tests/HartsyInference.Diffusion.Tests/Fixtures/rolling_tiny.safetensors

The DiT blocks, the frame-causal encode and the KV recompute are the ones dump_hy15_tiny.py already transcribes (`_window_block`,
`_recompute_cases`); this file adds only the loop. Every noise draw is saved in call order so the C# test can replay them.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from safetensors.torch import load_file, save_file

import dump_hy15_tiny as d

C, H, W = d.LATENT, d.HEIGHT, d.WIDTH
PF = H * W
NTRAIN = 1000.0
MAX_FRAMES = 40
COS, SIN = d.get_nd_rotary_pos_embed(d.ROPE_DIMS, (MAX_FRAMES, H, W), theta=256, device="cpu")


def sd3_shift(values, shift):
    # FlowMatchDiscreteScheduler.sd3_time_shift; applied only when shift != 1, as in the scheduler.
    return [shift * x / (1 + (shift - 1) * x) for x in values] if shift != 1.0 else list(values)


def base_schedule(steps, shift):
    # set_timesteps: linspace(1, 0, steps + 1), shifted; timesteps are the shifted sigmas[:-1] * 1000.
    sigmas = sd3_shift([1 - i / steps for i in range(steps + 1)], shift)
    return sigmas, [s * NTRAIN for s in sigmas[:-1]]


def ladder_schedule(ladder, shift):
    # set_custom_timesteps: ladder / 1000, shifted, then a trailing 0 appended.
    sigmas = sd3_shift([t / NTRAIN for t in ladder], shift) + [0.0]
    return sigmas, [s * NTRAIN for s in sigmas[:-1]]


class StepIndices:
    """StreamingSchedulerWrapper: row k is torch.roll(arange(steps), k * steps_per_window); the window's per-slot index is offset by
    ceil(max(g - steps + 1, 0) / steps_per_window) modulo the window size."""

    def __init__(self, steps, window):
        self.n = steps
        self.w = window
        self.spw = steps // window
        orig = list(range(steps))
        self.table = [[orig[(i - k * self.spw) % steps] for i in range(steps)] for k in range(window)]

    def get(self, global_step):
        step = global_step % self.n
        offset = math.ceil(max(global_step - self.n + 1, 0) / self.spw) % self.w
        return [self.table[(offset + k) % self.w][step] for k in range(self.w)]


class Stream:
    """The KV cache: per DiT block, a list of (keys, values, frames) entries in order (MovieRollKvCache's per-layer frame ranges)."""

    def __init__(self, model, full):
        self.m = model
        self.full = full
        self.cache = {b: [] for b in range(len(model.double_blocks))}
        self.encodes = 0
        self.recomputes = 0

    def cached_frames(self, b):
        return sum(frames for _, _, frames in self.cache[b])

    def history(self, b):
        entries = self.cache[b]
        if not entries:
            return None, 0
        keys = torch.cat([k for k, _, _ in entries], 1)
        values = torch.cat([v for _, v, _ in entries], 1)
        return (keys, values), self.cached_frames(b)

    def denoise(self, window, frame_t, qwen, byt5):
        """One denoising pass over the whole queue, each frame at its own timestep. Text is refined at the first frame's timestep."""
        m = self.m
        frames = window.shape[2]
        thw = [frames, H, W]
        img = m.img_in(torch.cat([window, torch.zeros(1, C + 1, frames, H, W)], 1))
        vec = m.time_in(frame_t).reshape(1, frames, -1)
        t = frame_t[:1]
        vec_txt = m.time_in(t)[:, None, :]
        txt = torch.cat([m.byt5_in(byt5) + m.cond_type_embedding.weight[1],
                         m.txt_in(qwen, t) + m.cond_type_embedding.weight[0]], 1)
        for b, block in enumerate(m.double_blocks):
            kv, cached = self.history(b)
            n = cached + frames
            img, txt, _ = d._window_block(block, img, txt, vec, vec_txt, thw, COS, SIN,
                                          torch.arange(cached * PF, n * PF), torch.arange(0, n * PF), kv, False)
        return m.final_layer(img, vec, thw=thw).reshape(1, frames, H, W, C).permute(0, 4, 1, 2, 3)

    def encode(self, clean):
        """A clean chunk enters the cache at timestep 0 without text. Sink-recent blocks are trimmed to the sink first."""
        m = self.m
        frames = clean.shape[2]
        for b in self.cache:
            if b >= self.full:
                self.cache[b] = self.cache[b][:1]
        vec = m.time_in(torch.zeros(frames)).reshape(1, frames, -1)
        img = m.img_in(torch.cat([clean, torch.zeros(1, C + 1, frames, H, W)], 1))
        for b, block in enumerate(m.double_blocks):
            kv, cached = self.history(b)
            n = cached + frames
            img, _, kv_out = d._window_block(block, img, None, vec, None, [frames, H, W], COS, SIN,
                                             torch.arange(cached * PF, n * PF), torch.arange(0, n * PF), kv, True)
            self.cache[b].append((kv_out[0], kv_out[1], frames))
        self.encodes += 1

    def recompute(self, joined, sizes):
        """Rebuild every cache from the selected chunks: full-history blocks keep one entry per chunk; sink-recent blocks keep the
        first chunk and, when there is more than one, the last one, both taken from this pass."""
        tensors = d._recompute_cases(self.m, joined, sizes)
        self.cache = {b: [] for b in self.cache}
        starts = [sum(sizes[:i]) * PF for i in range(len(sizes))]
        for b in range(len(self.m.double_blocks)):
            if b < self.full:
                key, value = tensors[f"rec.full{b}.k"], tensors[f"rec.full{b}.v"]
                self.cache[b] = [(key[:, s:s + size * PF], value[:, s:s + size * PF], size)
                                 for s, size in zip(starts, sizes)]
            else:
                key, value = tensors[f"rec.sink{b}.k"], tensors[f"rec.sink{b}.v"]
                sink_tokens = sizes[0] * PF
                entries = [(key[:, :sink_tokens], value[:, :sink_tokens], sizes[0])]
                if len(sizes) > 1:
                    entries.append((key[:, sink_tokens:], value[:, sink_tokens:], sizes[-1]))
                self.cache[b] = entries
        self.recomputes += 1
        return sum(sizes)


def select_recompute(total, recent_k):
    """_select_recompute_chunks with one event: the first chunk, the middle of the run, and the recent_k most recent chunks."""
    indices = {0}
    if total > 0:
        indices.add((0 + total - 1) // 2)
    indices.update(range(max(0, total - recent_k), total))
    return sorted(indices)


def run_loop(model, qwen, byt5, neg_qwen, neg_byt5, spec, draws, seed):
    gen = torch.Generator().manual_seed(seed)

    def draw(frames):
        x = torch.randn(1, C, frames, H, W, generator=gen)
        draws.append(x)
        return x

    def context(x):
        # apply_context_noise with force_equal_sigma: (1 - sigma) * x + sigma * eps, one eps draw of x's shape.
        sigma = spec["context"]
        if sigma <= 0:
            return x
        eps = draw(x.shape[2])
        return (1 - sigma) * x + sigma * eps

    sigmas, timesteps = spec["schedule"]
    steps, window = spec["steps"], spec["window"]
    spw = steps // window
    wrap = StepIndices(steps, window)
    stream = Stream(model, spec["full"])
    num_chunks, chunk = spec["num_chunks"], spec["chunk"]
    tail = chunk
    queue, outputs = [], []
    decoded = 0
    with torch.no_grad():
        for r in range(num_chunks + window):
            if r >= window:
                clean = queue.pop(0)
                outputs.append(clean.clone())
                is_global_last = decoded == num_chunks - 1
                if not is_global_last:
                    do_recompute = spec["recompute"] and len(stream.cache[0]) >= spec["threshold"]
                    if do_recompute:
                        selected = select_recompute(len(outputs), spec["recent_k"])
                        if len(selected) >= len(outputs):
                            do_recompute = False
                    if do_recompute:
                        joined = torch.cat([outputs[i] for i in selected], 2)
                        stream.recompute(context(joined), [outputs[i].shape[2] for i in selected])
                    else:
                        stream.encode(context(clean))
                decoded += 1
                if decoded >= num_chunks:
                    break
            # t2v's first chunk is 1 + chunk (movieroll_dmd_pipeline.py:740-765, movieroll_pipeline.py:991-992); spec "first" sets it.
            first = spec.get("first") or chunk
            frames = first if r == 0 else (chunk if r < num_chunks else tail)
            queue.append(draw(frames))
            for s in range(spw):
                g = r * spw + s
                index = wrap.get(g)[:len(queue)]
                frame_t = torch.cat([torch.full((x.shape[2],), timesteps[index[c]]) for c, x in enumerate(queue)])
                window_latents = torch.cat(queue, 2)
                velocity = stream.denoise(window_latents, frame_t, qwen, byt5)
                uncond = stream.denoise(window_latents, frame_t, neg_qwen, neg_byt5) if spec["cfg"] else None
                updated, off = [], 0
                for c, x in enumerate(queue):
                    f = x.shape[2]
                    v = velocity[:, :, off:off + f]
                    if spec["cfg"]:
                        lo, hi = spec["cfg"]
                        scale = lo + (hi - lo) * timesteps[index[c]] / NTRAIN
                        v = uncond[:, :, off:off + f] + scale * (v - uncond[:, :, off:off + f])
                    dt = sigmas[index[c] + 1] - sigmas[index[c]]
                    updated.append(x + dt * v)
                    off += f
                queue = updated
    return torch.cat(outputs, 2), stream


CONFIGS = {
    # Base chain: CFG with AnnealCFG (1 -> 4), four steps in two windows, KV recompute at three cached chunks.
    "base": dict(schedule=base_schedule(4, 5.0), steps=4, window=2, full=1, context=0.05, chunk=1, num_chunks=5,
                 cfg=(1.0, 4.0), recompute=True, threshold=3, recent_k=1),
    # Base chain with two-frame chunks and no recompute: frame offsets and the RoPE rows of a wider window.
    "base_chunk2": dict(schedule=base_schedule(4, 5.0), steps=4, window=2, full=1, context=0.05, chunk=2, num_chunks=3,
                        cfg=(1.0, 4.0), recompute=False, threshold=0, recent_k=0),
    # Distilled chain: the five-step ladder, one step per round, no CFG, context noise 0.15; recompute fires on several rounds.
    "distilled": dict(schedule=ladder_schedule([1000, 800, 600, 400, 200], 5.0), steps=5, window=5, full=1, context=0.15,
                      chunk=1, num_chunks=7, cfg=None, recompute=True, threshold=3, recent_k=1),
    # Text-to-video layout: a first chunk of 1 + chunk frames (2 here), then one-frame chunks, with CFG and recompute as in "base".
    # Listed last because each config's seed is its index in this dict, so the earlier fixtures keep their draws.
    "base_t2v": dict(schedule=base_schedule(4, 5.0), steps=4, window=2, full=1, context=0.05, chunk=1, first=2, num_chunks=5,
                     cfg=(1.0, 4.0), recompute=True, threshold=3, recent_k=1),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    state = load_file(args.weights)
    model = d.Model().float().eval()
    model.load_state_dict({name: state[name] for name in model.state_dict().keys()})
    qwen, byt5 = state["in.qwen"], state["in.byt5"]
    neg_gen = torch.Generator().manual_seed(77)
    neg_qwen = torch.randn(1, d.QWEN_LEN, d.TEXT_DIM, generator=neg_gen)
    neg_byt5 = torch.randn(1, d.BYT5_LEN, d.BYT5_DIM, generator=neg_gen)
    tensors = {"in.neg_qwen": neg_qwen, "in.neg_byt5": neg_byt5}
    for index, (name, spec) in enumerate(CONFIGS.items()):
        assert spec["full"] == d.STREAM_FULL_HISTORY, "the recompute fixture fixes the full-history block count"
        assert len(spec["schedule"][1]) == spec["steps"], "the ladder length is the step count"
        draws = []
        out, stream = run_loop(model, qwen, byt5, neg_qwen, neg_byt5, spec, draws, seed=4242 + index)
        if spec["recompute"]:
            assert stream.recomputes >= 1, f"{name}: the recompute branch never ran"
        tensors[f"{name}.out"] = out.contiguous()
        tensors[f"{name}.timesteps"] = torch.tensor(spec["schedule"][1], dtype=torch.float32)
        tensors[f"{name}.sigmas"] = torch.tensor(spec["schedule"][0], dtype=torch.float32)
        for i, value in enumerate(draws):
            tensors[f"{name}.noise.{i:03d}"] = value.contiguous()
        print(f"{name}: out {tuple(out.shape)}, {len(draws)} draws, {stream.encodes} encodes, {stream.recomputes} recomputes")
    save_file({k: v.float().contiguous() for k, v in tensors.items()}, args.out,
              metadata={"source": "MovieRoll 0b5f73e rolling loop transcription, tiny config"})
    print("wrote", args.out, len(tensors), "tensors")


if __name__ == "__main__":
    main()
