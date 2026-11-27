"""LongCat-Video 720p refinement: per-step reference dump on the tiny B1 checkpoint.

Runs the pinned reference's own `generate_refine` loop, its FlowMatchEulerDiscreteScheduler and its DiT with block-sparse attention
enabled, with an image condition and without one. The stubs of dump_longcat_pipeline.py apply (see its docstring).

STAND-IN: the reference's attention kernel is Triton and needs CUDA. Here `flash_attn_bsa_3d` is replaced by the reference's own
blocking, mean pooling and top-k selection (the function sources are compiled from block_sparse_attention/bsa_interface.py, without
their torch.compile decorators) followed by torch SDPA over the selected key blocks of each query block. The kernel itself is
compared with the engine's operation elsewhere (tests/python-reference/longcat_bsa).

With --triton the reference's own kernel runs instead (needs CUDA and triton; the DiT stays on the CPU and only the attention
operands move to the card). `meta.json`'s `attention` field records which ran.

Stubbed and therefore NOT verified here: the VAE (the dump queues the latents it would have produced for the upsampled clip and the
condition frames), umT5, and the pixel preparation's result (the reference computes it, and the VAE stub ignores it). The refinement
LoRA is not loaded: the tiny checkpoint has none.

Outputs (under --out): `ckpt/`, `meta.json`, and raw little-endian F32 files `<scenario>.<name>.f32`.
"""
import argparse
import ast
import importlib
import json
import math
import os
import shutil
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

import dump_longcat_pipeline as base

SPARSITY = 0.5
BLOCK = [4, 4, 4]                # the released block; the Triton kernel does not compile 2x2x2
SHIFT = 12.0                     # the released base scheduler's shift; with 50 steps and t 0.5 the refinement runs five
STEPS = 50
THRESHOLD = 0.5
HEIGHT, WIDTH = 128, 192          # pixel size the pipeline is told: latent 16 x 24, token grid 8 x 12
STAGE1_FRAMES = 5                # doubled to 10, padded to multiples of four latents
NOISY_LATENTS = 4
SELECTION = []                   # (query blocks, key blocks, selected) of every attention call


def reference_selection(repo):
    """The reference's own blocking and selection functions, compiled from its source."""
    path = os.path.join(repo, "longcat_video", "block_sparse_attention", "bsa_interface.py")
    with open(path, encoding="utf-8") as file:
        tree = ast.parse(file.read(), path)
    wanted = {"rearrange_THW_to_3d_block", "rearrange_3d_block_to_THW", "mean_pooling_compression", "cal_score",
              "get_select_indices_topk_from_score", "get_select_indices_topk", "get_select_indices"}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in body} == wanted, "the reference's selection functions changed"
    for node in body:
        node.decorator_list = []
    namespace = {"torch": torch, "F": F, "math": math}
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), namespace)
    return namespace


def stand_in(reference):
    def flash_attn_bsa_3d(q, k, v, latent_shape_q, latent_shape_k, sparsity=0.875, cdf_threshold=None,
                          chunk_3d_shape_q=(4, 4, 8), chunk_3d_shape_k=(4, 4, 8)):
        assert cdf_threshold is None, "only the top-k form is dumped"
        batch, heads, _, dim = q.shape
        counts_q = [g // b for g, b in zip(latent_shape_q, chunk_3d_shape_q)]
        counts_k = [g // b for g, b in zip(latent_shape_k, chunk_3d_shape_k)]
        size_q, size_k = int(np.prod(chunk_3d_shape_q)), int(np.prod(chunk_3d_shape_k))
        qb = reference["rearrange_THW_to_3d_block"](q, *counts_q, *chunk_3d_shape_q, dim)
        kb = reference["rearrange_THW_to_3d_block"](k, *counts_k, *chunk_3d_shape_k, dim)
        vb = reference["rearrange_THW_to_3d_block"](v, *counts_k, *chunk_3d_shape_k, dim)
        indices, _ = reference["get_select_indices"](reference["mean_pooling_compression"](qb, size_q),
                                                     reference["mean_pooling_compression"](kb, size_k), sparsity, None)
        query_blocks, key_blocks, chosen = int(np.prod(counts_q)), int(np.prod(counts_k)), indices.shape[-1]
        SELECTION.append((query_blocks, key_blocks, chosen))
        kb = kb.view(batch, heads, key_blocks, size_k, dim)
        vb = vb.view(batch, heads, key_blocks, size_k, dim)
        out = torch.zeros_like(qb)
        for b in range(batch):
            for h in range(heads):
                for i in range(query_blocks):
                    taken = indices[b, h, i].sort().values
                    keys = kb[b, h, taken].reshape(1, chosen * size_k, dim)
                    values = vb[b, h, taken].reshape(1, chosen * size_k, dim)
                    rows = qb[b, h, i * size_q:(i + 1) * size_q][None]
                    out[b, h, i * size_q:(i + 1) * size_q] = F.scaled_dot_product_attention(rows, keys, values)[0]
        return reference["rearrange_3d_block_to_THW"](out, *counts_q, *chunk_3d_shape_q, dim)

    return flash_attn_bsa_3d


def reference_kernel():
    """The reference's own `flash_attn_bsa_3d`, run on the card; the DiT around it stays on the CPU."""
    name = "longcat_video.block_sparse_attention.bsa_interface"
    sys.modules.pop(name, None)
    kernel = importlib.import_module(name).flash_attn_bsa_3d

    def flash_attn_bsa_3d(q, k, v, latent_shape_q, latent_shape_k, **params):
        counts_q = [g // b for g, b in zip(latent_shape_q, params["chunk_3d_shape_q"])]
        counts_k = [g // b for g, b in zip(latent_shape_k, params["chunk_3d_shape_k"])]
        key_blocks = int(np.prod(counts_k))
        SELECTION.append((int(np.prod(counts_q)), key_blocks, int((1 - params["sparsity"]) * key_blocks)))
        return kernel(q.cuda(), k.cuda(), v.cuda(), latent_shape_q, latent_shape_k, **params).cpu()

    return flash_attn_bsa_3d


class Frame(np.ndarray):
    """A stage-1 frame: the pipeline reads `height` and `width` and converts the list to one array."""

    height = property(lambda self: self.shape[0])
    width = property(lambda self: self.shape[1])


def run(args):
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    repo = os.path.abspath(args.repo)
    base.install_stubs(repo)
    attention = reference_kernel() if args.triton else stand_in(reference_selection(repo))
    interface = types.ModuleType("longcat_video.block_sparse_attention.bsa_interface")
    interface.flash_attn_bsa_3d = attention
    sys.modules["longcat_video.block_sparse_attention.bsa_interface"] = interface
    if not args.triton:          # the pipeline calls these whether or not a card is there
        torch.cuda.empty_cache = lambda: None
        torch.cuda.ipc_collect = lambda: None
    base.CONFIG["bsa_params"] = dict(sparsity=SPARSITY, chunk_3d_shape_q=BLOCK, chunk_3d_shape_k=BLOCK)

    ckpt_dst = os.path.join(out_dir, "ckpt")
    if not os.path.isdir(ckpt_dst):
        shutil.copytree(os.path.abspath(args.ckpt), ckpt_dst)

    meta = dict(commit=base.LONGCAT_COMMIT, torch=torch.__version__, attention="triton" if args.triton else "stand-in",
                shift=SHIFT, steps=STEPS, threshold=THRESHOLD, sparsity=SPARSITY,
                block=BLOCK, text_len=base.TEXT_LEN, valid_pos=base.VALID_POS, tensors={}, scenarios={})
    recorder = base.Recorder(out_dir, meta)
    base.SHIFT = SHIFT           # the scheduler build_pipeline constructs
    pipe = base.build_pipeline(ckpt_dst, recorder)
    assert pipe.scheduler.shift == SHIFT
    pipe.dit.model.enable_bsa()
    pipe.get_condition_shape = lambda condition, resolution, scale_factor_spatial=32: (HEIGHT, WIDTH)

    gen = torch.Generator().manual_seed(args.seed)
    pipe.pos, pipe.neg = base.text_inputs(gen)
    recorder.name = "text"
    recorder.tensor("pos_embeds", pipe.pos[0][0])

    channels = base.CONFIG["in_channels"]

    def latent(frames):
        return torch.randn(1, channels, frames, HEIGHT // 8, WIDTH // 8, generator=gen)

    original_randn_like = torch.randn_like
    for name, num_cond_frames, cond_latents in (("refine_i2v", 1, 4), ("refine_t2v", 0, 0)):
        frames = cond_latents + NOISY_LATENTS
        raw = latent(frames)
        pipe.vae.queue.append(raw)
        raw_cond = latent(cond_latents) if cond_latents else None
        if raw_cond is not None:
            pipe.vae.queue.append(raw_cond)
        recorder.begin(name, num_cond=cond_latents, frames=frames, latent_height=HEIGHT // 8, latent_width=WIDTH // 8)
        recorder.tensor("latent", pipe.normalize_latents(raw))
        if raw_cond is not None:
            recorder.tensor("cond", pipe.normalize_latents(raw_cond))

        def recorded_randn_like(like, **_):
            noise = torch.randn(like.shape, generator=gen, dtype=like.dtype)
            recorder.tensor("noise", noise)
            return noise

        torch.randn_like = recorded_randn_like
        SELECTION.clear()
        stage1 = [np.full((HEIGHT // 2, WIDTH // 2, 3), 16 * i, dtype=np.uint8).view(Frame) for i in range(STAGE1_FRAMES)]
        image = torch.zeros(1, 3, HEIGHT, WIDTH) if num_cond_frames else None
        try:
            final = pipe.generate_refine(image=image, prompt=base.PROMPT, stage1_video=stage1, num_cond_frames=num_cond_frames,
                                         num_inference_steps=STEPS, output_type="latent", max_sequence_length=base.TEXT_LEN,
                                         t_thresh=THRESHOLD)
        finally:
            torch.randn_like = original_randn_like
        assert not pipe.vae.queue, "a queued latent was not read"
        assert list(final.shape[2:]) == [frames, HEIGHT // 8, WIDTH // 8], final.shape
        assert SELECTION and all(0 < chosen < keys for _, keys, chosen in SELECTION), "a call read no or every key block"
        recorder.name = name
        recorder.tensor("final", final)
        meta["scenarios"][name]["sigmas"] = pipe.scheduler.sigmas.tolist()
        meta["scenarios"][name]["selection"] = sorted(set(SELECTION))

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    print("wrote", out_dir, "scenarios:", {k: (len(v["steps"]), v["selection"]) for k, v in meta["scenarios"].items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="LongCat-Video at " + base.LONGCAT_COMMIT)
    ap.add_argument("--ckpt", required=True, help="the B1 tiny checkpoint folder (ckpt/ from the B1 dump)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--triton", action="store_true", help="run the reference's Triton kernel on the card instead of the stand-in")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
