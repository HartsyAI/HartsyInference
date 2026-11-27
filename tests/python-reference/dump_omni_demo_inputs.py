"""Reference values for how Hunyuan3D-Omni turns its demo control files into encoder input rows: upstream's own `inference.py` steps, run on the CPU.

Reproduces, for each control kind, what `inference.py` of a checkout of Tencent-Hunyuan/Hunyuan3D-Omni (pinned in docs/Research/HUNYUAN3D_OMNI_ARCHITECTURE.md) does to a demo
file before the pipeline hands it to the encoder, with the script's own `normalize_mesh` (copied from the source text) and trimesh:

* point: `trimesh.load(<ply>)`, `normalize_mesh(mesh, scale=0.98)`, `mesh.vertices`;
* voxel: `trimesh.load(<mesh>)`, a rotation of -90 degrees about x, `normalize_mesh(mesh)` (scale 0.9999), `mesh.sample(81920)` (trimesh's `sample_surface` with seed 0 here), then the
  occupied voxel centres of the encoder's `generate_voxel` at resolution 16 (the transcription of `dump_omni_control.py`, which is the one copy of it);
* pose: `np.loadtxt(<bone .txt>)`, one bone per row of six values;
* bbox: the first three numbers of `demos/bbox/data.json`.

The encoder then repeats three values to six (`rows.repeat(1, 1, 2)`), which `dump_omni_control.py` covers, so this fixture stops at the values the encoder receives.

    python dump_omni_demo_inputs.py --omni <Omni checkout> --out <fixture.npz>

Fixture keys: `point.raw` (the file's vertices), `point.normalised` (what upstream hands the encoder); `voxel.vertices`, `voxel.faces` (the loaded mesh) and `voxel.centres`
(the occupied voxel centres upstream's sample gives, sorted by linear index); `pose.raw` (the bone file); `bbox.values`.
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import torch
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))


def normalize_mesh(mesh, scale=0.9999):
    """inference.py, copied: centre the bounding box on the origin and scale its longest side to 2 * scale."""
    bbox = mesh.bounds
    center = (bbox[1] + bbox[0]) / 2
    scale_ = (bbox[1] - bbox[0]).max()
    mesh.apply_translation(-center)
    mesh.apply_scale(1 / scale_ * 2 * scale)
    return mesh


def voxel_centres(points, resolution=16):
    """The occupied cells' centres in ascending linear-index order, by `generate_voxel` of dump_omni_control.py (a transcription of OmniEncoder.generate_voxel)."""
    spec = importlib.util.spec_from_file_location("dump_omni_control", os.path.join(HERE, "dump_omni_control.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_voxel(torch.from_numpy(points).unsqueeze(0), resolution)[0].numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--omni", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    demos = os.path.join(args.omni, "demos")
    out = {}

    cloud = trimesh.load(os.path.join(demos, "point", "plys", "008.ply"))
    out["point.raw"] = np.asarray(cloud.vertices, dtype=np.float32)
    out["point.normalised"] = np.asarray(normalize_mesh(cloud, scale=0.98).vertices, dtype=np.float32)

    mesh = trimesh.load(os.path.join(demos, "voxel", "plys", "1c1ff58afbf4455ca80228d280f86aef.ply"))
    out["voxel.vertices"] = np.asarray(mesh.vertices, dtype=np.float32)
    out["voxel.faces"] = np.asarray(mesh.faces, dtype=np.int32)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(angle=np.radians(-90), direction=[1, 0, 0]))
    mesh = normalize_mesh(mesh)
    surface, _ = trimesh.sample.sample_surface(mesh, 81920, seed=0)
    out["voxel.centres"] = voxel_centres(surface.astype(np.float32))

    out["pose.raw"] = np.loadtxt(os.path.join(demos, "pose", "a_pose_bone.txt")).astype(np.float32)
    out["bbox.values"] = np.array(json.load(open(os.path.join(demos, "bbox", "data.json")))["bbox"][0], dtype=np.float32)
    np.savez(args.out, **out)
    print("wrote", args.out, {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
