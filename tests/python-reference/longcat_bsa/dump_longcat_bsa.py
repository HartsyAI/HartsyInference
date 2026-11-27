"""Outputs of LongCat-Video's block-sparse attention (flash_attn_bsa_3d, its own Triton kernel) for the cases
BlockSparseAttentionReferenceTests rebuilds.

Usage:
    python dump_longcat_bsa.py --repo /path/to/LongCat-Video --out DIR [--vectors BlockSparseAttentionVectors.cs]
Needs an NVIDIA GPU, torch 2.6.0 (cu124 wheels, which bring triton 3.2.0), einops, and a checkout of meituan-longcat/LongCat-Video
at 58eb336edc5c8c63389838727e40b8c18434e54e.

Nothing of the reference is replaced. Two environment settings are forced before triton and torch load: TRITON_F32_DEFAULT=ieee,
because Triton's F32 default on this hardware is TF32, and TORCHDYNAMO_DISABLE=1, so the torch.compile helpers run eagerly.

LongCat's head dim of 128 is not among the cases: the F32 kernel at 64 keys a block needs more shared memory than an RTX 30 or 40
series card has, even at one pipeline stage. BF16, which the pipeline runs, fits.

The F32 run is what the C# reference is compared with. A BF16 run of the same inputs is reported beside it (how far the output and
the block selection move at the precision the pipeline runs); no test reads it.
"""
import argparse
import hashlib
import json
import os
import sys

os.environ["TRITON_F32_DEFAULT"] = "ieee"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import numpy as np
import torch
import triton

LONGCAT_COMMIT = "58eb336edc5c8c63389838727e40b8c18434e54e"
BSA_FILES = ("bsa_interface.py", "flash_attn_bsa_varlen_mask.py")

# Grids and blocks are (frames, height, width). q and k are scaled by 2 after generation so the softmax is not flat.
CASES = [
    # The shipped bsa_params: 4x4x4 blocks, sparsity 0.9375.
    dict(name="Shipped", batch=1, heads=2, dim=32, q_grid=(8, 16, 16), k_grid=(8, 16, 16), q_block=(4, 4, 4),
         k_block=(4, 4, 4), sparsity=0.9375, seed=11, row_step=67),
    # More key frames than query frames, as the pipeline's noisy queries over all keys.
    dict(name="CrossLength", batch=1, heads=2, dim=64, q_grid=(4, 16, 16), k_grid=(12, 16, 16), q_block=(4, 4, 4),
         k_block=(4, 4, 4), sparsity=0.9375, seed=23, row_step=61),
    dict(name="Batched", batch=2, heads=3, dim=16, q_grid=(4, 32, 16), k_grid=(4, 32, 16), q_block=(4, 4, 4),
         k_block=(4, 4, 4), sparsity=0.875, seed=37, row_step=127),
    # Different query and key blocks.
    dict(name="UnequalBlocks", batch=1, heads=2, dim=16, q_grid=(8, 16, 16), k_grid=(8, 16, 32), q_block=(4, 4, 4),
         k_block=(4, 4, 8), sparsity=0.9375, seed=41, row_step=67),
    # (1 - 0.9) * 30 is 2.9999999999999996 in double, so two blocks are read, not three.
    dict(name="Truncated", batch=1, heads=1, dim=32, q_grid=(4, 8, 8), k_grid=(20, 12, 8), q_block=(4, 4, 4),
         k_block=(4, 4, 4), sparsity=0.9, seed=53, row_step=11),
    # (1 - 0.9375) * 8 truncates to zero selected blocks.
    dict(name="NoneSelected", batch=1, heads=1, dim=16, q_grid=(8, 8, 8), k_grid=(8, 8, 8), q_block=(4, 4, 4),
         k_block=(4, 4, 4), sparsity=0.9375, seed=67, row_step=29),
]


def uniform(count, seed):
    """Values in [-1, 1) from a 32-bit LCG, exact in F32, identical in the C# test."""
    values = np.empty(count, dtype=np.float32)
    state = seed
    for i in range(count):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        values[i] = ((state >> 8) & 0xFFFF) / 32768.0 - 1.0
    return values


def operands(case):
    batch, heads, dim = case["batch"], case["heads"], case["dim"]
    sq, sk = int(np.prod(case["q_grid"])), int(np.prod(case["k_grid"]))
    q = uniform(batch * heads * sq * dim, case["seed"]).reshape(batch, heads, sq, dim) * 2
    k = uniform(batch * heads * sk * dim, case["seed"] + 1).reshape(batch, heads, sk, dim) * 2
    v = uniform(batch * heads * sk * dim, case["seed"] + 2).reshape(batch, heads, sk, dim)
    return q, k, v


def selection(bsa, case, q, k):
    """The reference's own pooling and top-k, which forward() does not return. Returns ascending indices and the smallest score
    gap between the last block taken and the first left out."""
    blocks = []
    for x, grid, box in ((q, case["q_grid"], case["q_block"]), (k, case["k_grid"], case["k_block"])):
        counts = [g // b for g, b in zip(grid, box)]
        blocked = bsa.rearrange_THW_to_3d_block(x, *counts, *box, x.shape[-1])
        blocks.append(bsa.mean_pooling_compression(blocked, int(np.prod(box))))
    indices, lengths = bsa.get_select_indices(blocks[0], blocks[1], case["sparsity"], None)
    chosen = indices.shape[-1]
    assert bool((lengths == chosen).all())
    gap = None
    if 0 < chosen < blocks[1].shape[2]:
        ranked = torch.sort(bsa.cal_score(*blocks), dim=-1, descending=True).values.double()
        gap = float((ranked[..., chosen - 1] - ranked[..., chosen]).min())
    return torch.sort(indices, dim=-1).values.to(torch.int32), gap


def relative_rms(actual, expected):
    norm = float(np.sqrt(np.mean(np.square(expected, dtype=np.float64))))
    error = float(np.sqrt(np.mean(np.square(actual.astype(np.float64) - expected))))
    return error / norm if norm > 0 else error


def source_hash(repo, name):
    with open(os.path.join(repo, "longcat_video", "block_sparse_attention", name), "rb") as file:
        return hashlib.sha256(file.read().replace(b"\r\n", b"\n")).hexdigest()


def csharp_array(kind, name, values, per_line, form):
    lines = [", ".join(form(v) for v in values[i:i + per_line]) for i in range(0, len(values), per_line)]
    body = "".join(f"        {line},\n" for line in lines)
    return f"    public static readonly {kind}[] {name} =\n    [\n{body}    ];\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vectors")
    args = parser.parse_args()
    sys.path.insert(0, args.repo)
    from longcat_video.block_sparse_attention import bsa_interface as bsa

    os.makedirs(args.out, exist_ok=True)
    hashes = {name: source_hash(args.repo, name) for name in BSA_FILES}
    environment = dict(commit=LONGCAT_COMMIT, torch=torch.__version__, triton=triton.__version__,
                       device=torch.cuda.get_device_name(0), sources=hashes)
    print(json.dumps(environment))
    report = dict(environment=environment, cases=[])
    members = []
    for case in CASES:
        q, k, v = operands(case)
        runs = {}
        for precision, dtype in (("f32", torch.float32), ("bf16", torch.bfloat16)):
            tq, tk, tv = (torch.from_numpy(x).cuda().to(dtype) for x in (q, k, v))
            indices, gap = selection(bsa, case, tq, tk)
            output = bsa.flash_attn_bsa_3d(tq, tk, tv, case["q_grid"], case["k_grid"], sparsity=case["sparsity"],
                                           cdf_threshold=None, chunk_3d_shape_q=list(case["q_block"]),
                                           chunk_3d_shape_k=list(case["k_block"]))
            torch.cuda.synchronize()
            runs[precision] = (output.float().cpu().numpy(), indices.cpu().numpy(), gap)
        output, indices, gap = runs["f32"]
        low_output, low_indices, _ = runs["bf16"]
        # A block set differs when any of its entries does: both lists are ascending.
        moved = int((low_indices != indices).any(axis=-1).sum()) if indices.size else 0
        row = dict(name=case["name"], selected=int(indices.shape[-1]), min_score_gap=gap,
                   output_rms=float(np.sqrt(np.mean(np.square(output, dtype=np.float64)))),
                   bf16_relative_rms=relative_rms(low_output, output), bf16_query_blocks_with_other_selection=moved,
                   query_block_rows=int(np.prod(indices.shape[:3])))
        report["cases"].append(row)
        print(json.dumps(row))
        output.astype("<f4").tofile(os.path.join(args.out, case["name"] + ".output.f32"))
        indices.astype("<i4").tofile(os.path.join(args.out, case["name"] + ".selected.i32"))

        sampled = output[:, :, ::case["row_step"], :].astype("<f4").ravel()
        members.append(csharp_array("uint", case["name"] + "Output", sampled.view("<u4").tolist(), 8, lambda x: f"0x{x:08X}"))
        if indices.size:
            members.append(csharp_array("int", case["name"] + "Selected", indices.ravel().tolist(), 24, str))

    with open(os.path.join(args.out, "report.json"), "w", newline="\n") as file:
        json.dump(report, file, indent=2)

    if args.vectors:
        header = (
            "// Generated by tests/python-reference/longcat_bsa/dump_longcat_bsa.py; do not edit by hand.\n"
            f"// meituan-longcat/LongCat-Video {LONGCAT_COMMIT},\n"
            + "".join(f"// {name} sha256 {digest} (line endings normalized),\n" for name, digest in hashes.items())
            + f"// torch {torch.__version__}, triton {triton.__version__}, {environment['device']}, F32 with IEEE dot products.\n"
            "namespace HartsyInference.Core.Tests.Backends;\n\n"
            "/// <summary>The reference kernel's block selection and sampled output rows for the cases BlockSparseAttentionReferenceTests\n"
            "/// rebuilds: every RowStep-th query row as F32 bits in <c>[batch, head, row, channel]</c> order, and each query block's key\n"
            "/// blocks, ascending, as <c>[batch, head, queryBlock, selected]</c>.</summary>\n"
            "internal static class BlockSparseAttentionVectors\n{\n"
        )
        with open(args.vectors, "w", newline="\n") as file:
            file.write(header + "\n".join(members) + "}\n")


if __name__ == "__main__":
    main()
