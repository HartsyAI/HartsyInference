"""Numpy reference outputs of Open Vision Banana's decoders and prompt builders, for the C# decoder tests (OvbDecoderTests).

These are TRANSCRIPTIONS, not runs of the release: the functions below were read from OpenVisionBanana/OpenVisionBanana
@ d457d3f7fe3a871d757956167ed552a1b4762ad8 (Apache-2.0) and rewritten here without importing any of its code (nothing from that
repository is executed). Each cites its source. A mistake in the transcription would be shared with the C# port, so the
real-weight run also compares against the release's own pipeline outputs.

  rgb_to_trainids_via_legend_with_dist, merge_grouped_predictions   common/semseg/diversify.py L359-L402
  cumulative_majority_vote_trainids, majority_vote_trainids, make_selective_prompt
                                                                      evaluation/semseg/inference_partial_flux2_klein_img2img.py
  decode_surface_normals_rgb (vision_banana_x_left_red, min_norm)     common/surface_normal_encoding.py L123-L147
  depth read: PIL convert("L") / 255                                   evaluation/depth/inference_partial_flux2_klein_img2img.py L262
  rgb_to_foreground (referring segmentation)                           evaluation/refseg/inference_flux2_klein_refseg.py L176-L188

Run in a CPU container with numpy, pillow and torch (for safetensors); the project's `hartsy-ref:cpu` image is local to the maintainers' machine, any equivalent works:

    python tests/python-reference/dump_ovb_decoders.py --out tests/HartsyInference.Engine.Tests/Fixtures/ovb_decoders

Everything is saved as float32 (the values are small integers or exact floats), plus meta.json with the prompt strings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.torch import save_file
import torch

H, W, K, PASSES = 20, 24, 5, 3


IGNORE = 255


def rgb_to_trainids_via_legend_with_dist(rgb, colors, class_ids=None):
    # The release's legend is one colour per label of the whole taxonomy; a grouped pass (a legend over a subset of the classes)
    # names its classes by their global id through class_ids, which this transcription adds to model that. merge_grouped_predictions
    # documents its inputs as "already mapped to FULL-taxonomy ids".
    arr = np.array(rgb, dtype=np.int32)
    h, w, _ = arr.shape
    palette = np.array(list(colors) + [[0, 0, 0]], dtype=np.int32)
    ids_for_colors = list(range(len(colors))) if class_ids is None else list(class_ids)
    mapping = np.array(ids_for_colors + [IGNORE], dtype=np.uint8)
    flat = arr.reshape(-1, 3)
    dist = ((flat[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    idx = dist.argmin(axis=1)
    ids = mapping[idx].reshape(h, w)
    min_dist = dist[np.arange(dist.shape[0]), idx].reshape(h, w).astype(np.float32)
    return ids, min_dist


def merge_grouped_predictions(ids_maps, dist_maps):
    ids = np.stack([np.asarray(m) for m in ids_maps])
    dist = np.stack([np.asarray(d, dtype=np.float64) for d in dist_maps])
    dist = np.where(ids == IGNORE, np.inf, dist)
    best = dist.argmin(axis=0)
    return np.take_along_axis(ids, best[None], axis=0)[0]


def majority_vote_trainids(masks, num_classes):
    shape = masks[0].shape
    counts = np.zeros((num_classes, *shape), dtype=np.uint16)
    voted_total = np.zeros(shape, dtype=np.uint16)
    for mask in masks:
        valid = mask < num_classes
        voted_total[valid] += 1
        for class_id in np.unique(mask[valid]):
            counts[class_id][valid & (mask == class_id)] += 1
    winner = counts.argmax(axis=0).astype(np.uint8)
    top = counts.max(axis=0)
    voted = np.where(top > 0, winner, IGNORE).astype(np.uint8)
    agreement = np.zeros(shape, dtype=np.float32)
    np.divide(top, voted_total, out=agreement, where=voted_total > 0)
    return voted, agreement


def make_selective_prompt(classes, colors):
    color_map = ", ".join(f"'{name}': {list(colors[idx])}" for idx, name in enumerate(classes))
    return (
        "[TASK: SEMANTIC_SEGMENTATION] Generate a semantic segmentation visualization of the input image, "
        f"using this color mapping for the listed classes: {{ {color_map} }}. "
        "Paint pixels belonging to a listed class with exactly its RGB color. "
        "Pixels that do not belong to one of these listed classes must be black [0, 0, 0] "
        "and must not be forced into a listed class. Output only the segmentation visualization image."
    )


def rgb_to_foreground(rgb, fg_color):
    # evaluation/refseg/inference_flux2_klein_refseg.py L176-L188: nearer to the paint colour than to black, strictly.
    arr = np.asarray(rgb, dtype=np.int64)
    fg = np.asarray(fg_color, dtype=np.int64)
    d_fg = ((arr - fg) ** 2).sum(-1)
    d_bg = (arr ** 2).sum(-1)
    return (d_fg < d_bg).astype(np.uint8)


def decode_normals(rgb, min_norm=0.05):
    array = np.asarray(rgb, dtype=np.float32)
    stored = array / 255.0 * 2.0 - 1.0
    normals = stored.copy()
    normals[..., 0] *= -1  # vision_banana_x_left_red
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    valid = np.isfinite(norm[..., 0]) & (norm[..., 0] >= float(min_norm))
    unit = np.divide(normals, norm, out=np.zeros_like(normals), where=norm >= float(min_norm))
    unit[~np.isfinite(unit)] = 0
    return unit.astype(np.float32), valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    nprng = np.random.default_rng(11)
    classes = ["road", "sidewalk", "car", "person", "sky"]
    # Ground truth: blobs of classes, label 255 = background (painted black).
    label = np.full((H, W), 255, dtype=np.int64)
    for c in range(K):
        y, x = nprng.integers(0, H - 6), nprng.integers(0, W - 8)
        label[y:y + nprng.integers(4, 7), x:x + nprng.integers(5, 9)] = c

    tensors: dict[str, torch.Tensor] = {}
    legends, ids_list, dist_list = [], [], []
    for g in range(PASSES):
        # Any distinct colours do for a decoder test (the legend sampler is tested on the C# side by its own properties).
        colors = [[int(v) for v in c] for c in nprng.choice(np.arange(8, 248, 16), size=(K, 3))]
        while len({tuple(c) for c in colors}) < K:
            colors = [[int(v) for v in c] for c in nprng.choice(np.arange(8, 248, 16), size=(K, 3))]
        legends.append(colors)
        painted = np.zeros((H, W, 3), dtype=np.int32)
        for c in range(K):
            painted[label == c] = colors[c]
        # A noisy, partly wrong "model output": jitter every channel and replace 8% of pixels with random colours.
        painted = np.clip(painted + nprng.integers(-14, 15, size=painted.shape), 0, 255)
        wrong = nprng.random((H, W)) < 0.08
        painted[wrong] = nprng.integers(0, 256, size=(int(wrong.sum()), 3))
        painted = painted.astype(np.uint8)
        ids, dist = rgb_to_trainids_via_legend_with_dist(painted, colors)
        ids_list.append(ids.astype(np.int64))
        dist_list.append(dist)
        tensors[f"legend{g}"] = torch.tensor(colors, dtype=torch.float32)
        tensors[f"rgb{g}"] = torch.tensor(painted, dtype=torch.float32)
        tensors[f"ids{g}"] = torch.tensor(ids.astype(np.float32))
        tensors[f"dist{g}"] = torch.tensor(dist, dtype=torch.float32)

    voted, agreement = majority_vote_trainids([i.astype(np.uint8) for i in ids_list], K)
    tensors["voted"] = torch.tensor(np.asarray(voted, dtype=np.float32))
    tensors["agreement"] = torch.tensor(np.asarray(agreement, dtype=np.float32))
    merged = merge_grouped_predictions([i.astype(np.uint8) for i in ids_list], dist_list)
    tensors["merged"] = torch.tensor(np.asarray(merged, dtype=np.float32))

    # Depth read and normals decode on an arbitrary RGB image (not gray: the luma weights matter).
    free = nprng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    # Some mid-grey pixels so the validity floor (norm < 0.05) is exercised.
    free[0, :6] = [128, 128, 128]
    tensors["free_rgb"] = torch.tensor(free, dtype=torch.float32)
    tensors["depth"] = torch.tensor(np.asarray(Image.fromarray(free).convert("L"), dtype=np.float32) / 255.0)
    unit, valid = decode_normals(free)
    tensors["normals"] = torch.tensor(unit, dtype=torch.float32)
    tensors["normals_valid"] = torch.tensor(valid.astype(np.float32))

    # Grouped protocol: three passes whose legends cover DISJOINT class sets (global ids g*4..g*4+3 of 12 classes), each painting
    # only its own classes and black elsewhere, merged by snap distance.
    GROUPS, PER = 3, 4
    truth12 = np.full((H, W), 255, dtype=np.int64)
    for c in range(GROUPS * PER):
        y, x = nprng.integers(0, H - 5), nprng.integers(0, W - 7)
        truth12[y:y + nprng.integers(3, 6), x:x + nprng.integers(4, 8)] = c
    group_ids, group_dists = [], []
    for g in range(GROUPS):
        gcolors = [[int(v) for v in c] for c in nprng.choice(np.arange(8, 248, 16), size=(PER, 3))]
        while len({tuple(c) for c in gcolors}) < PER:
            gcolors = [[int(v) for v in c] for c in nprng.choice(np.arange(8, 248, 16), size=(PER, 3))]
        gpaint = np.zeros((H, W, 3), dtype=np.int32)
        for k in range(PER):
            gpaint[truth12 == g * PER + k] = gcolors[k]
        gpaint = np.clip(gpaint + nprng.integers(-14, 15, size=gpaint.shape), 0, 255).astype(np.uint8)
        gids, gdist = rgb_to_trainids_via_legend_with_dist(gpaint, gcolors, class_ids=list(range(g * PER, g * PER + PER)))
        group_ids.append(gids)
        group_dists.append(gdist)
        tensors[f"grp_legend{g}"] = torch.tensor(gcolors, dtype=torch.float32)
        tensors[f"grp_rgb{g}"] = torch.tensor(gpaint, dtype=torch.float32)
        tensors[f"grp_ids{g}"] = torch.tensor(gids.astype(np.float32))
        tensors[f"grp_dist{g}"] = torch.tensor(gdist, dtype=torch.float32)
    tensors["grp_merged"] = torch.tensor(np.asarray(merge_grouped_predictions(group_ids, group_dists), dtype=np.float32))

    # Referring segmentation: a picture of red, black and in-between pixels, decoded against the paint colour.
    refer = np.zeros((H, W, 3), dtype=np.uint8)
    refer[:, : W // 2] = [220, 30, 30]
    refer = np.clip(refer.astype(np.int32) + nprng.integers(-60, 61, size=refer.shape), 0, 255).astype(np.uint8)
    refer[0, 0] = [110, 15, 15]  # exactly on the midpoint between red and black: a tie, which is background
    tensors["refer_rgb"] = torch.tensor(refer, dtype=torch.float32)
    tensors["refer_fg"] = torch.tensor(rgb_to_foreground(refer, [220, 30, 30]).astype(np.float32))

    quickstart_colors = legends[0]
    quickstart = (
        "[TASK: SEMANTIC_SEGMENTATION] Generate a semantic segmentation visualization of "
        "the input image, using this color mapping: {" + ", ".join(
            f'"{n}": [{c[0]},{c[1]},{c[2]}]' for n, c in zip(classes, quickstart_colors)) + "} "
        "Color each pixel of a listed class with its color and leave everything else black. "
        "Output only the segmentation visualization image."
    )
    save_file(tensors, str(out / "reference.safetensors"))
    (out / "meta.json").write_text(json.dumps({
        "height": H, "width": W, "classes": classes, "passes": PASSES, "classCount": K,
        "ovbCommit": "d457d3f7fe3a871d757956167ed552a1b4762ad8",
        "legends": legends,
        "selectivePrompt": make_selective_prompt(classes, legends[0]),
        "quickstartPrompt": quickstart,
        "note": "label 255 = ignore; quickstartPrompt follows the README quickstart literal with the same legend",
    }, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
