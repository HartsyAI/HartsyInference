"""Reference values for how Hunyuan3D-Omni turns its demo control files into encoder input rows: upstream's own `inference.py` steps, run on the CPU.

Reproduces, for each control kind, what `inference.py` of a checkout of Tencent-Hunyuan/Hunyuan3D-Omni (pinned in docs/Research/HUNYUAN3D_OMNI_ARCHITECTURE.md) does to a demo
file before the pipeline hands it to the encoder, with the script's own `normalize_mesh` (copied from the source text) and trimesh:

* point: `trimesh.load(<ply>)`, `normalize_mesh(mesh, scale=0.98)`, `mesh.vertices`;
* voxel: `trimesh.load(<mesh>)`, a rotation of -90 degrees about x, `normalize_mesh(mesh)` (scale 0.9999), `mesh.sample(81920)` (numpy's global RNG, seeded 0 here), then the
  occupied voxel centres of the encoder's `generate_voxel` at resolution 16;
* pose: `np.loadtxt(<bone .txt>)`, one bone per row of six values;
* bbox: the three numbers of `demos/bbox/data.json`.

The encoder then repeats three values to six (`rows.repeat(1, 1, 2)`), which `dump_omni_control.py` covers, so this fixture stops at the values the encoder receives.

    python dump_omni_demo_inputs.py --omni <Omni checkout> --out <fixture.npz>

Fixture keys: `point.raw` (the file's vertices), `point.normalised` (what upstream hands the encoder); `voxel.vertices`, `voxel.faces` (the loaded mesh) and `voxel.centres`
(the occupied voxel centres upstream's sample gives, sorted by linear index); `pose.raw` (the bone file); `bbox.values`.
"""
import argparse
import os

import numpy as np
import trimesh


def normalize_mesh(mesh, scale=0.9999):
    """inference.py, copied: centre the bounding box on the origin and scale its longest side to 2 * scale."""
    bbox = mesh.bounds
    center = (bbox[1] + bbox[0]) / 2
    scale_ = (bbox[1] - bbox[0]).max()
    mesh.apply_translation(-center)
    mesh.apply_scale(1 / scale_ * 2 * scale)
    return mesh


def voxel_centres(points, resolution=16):
    """OmniEncoder.generate_voxel: the occupied cells' centres in ascending linear-index order."""
    norm = (points + 1) / 2
    cells = np.clip(np.floor(norm * resolution).astype(np.int64), 0, resolution - 1)
    linear = cells[:, 0] + cells[:, 1] * resolution + cells[:, 2] * resolution * resolution
    unique = np.unique(linear)
    z = unique // (resolution * resolution)
    y = (unique % (resolution * resolution)) // resolution
    x = unique % resolution
    size = 2.0 / resolution
    return (np.stack([x, y, z], axis=1) * size + size / 2 - 1).astype(np.float32)


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
    np.random.seed(0)
    surface = mesh.sample(81920)
    out["voxel.centres"] = voxel_centres(surface.astype(np.float32))

    out["pose.raw"] = np.loadtxt(os.path.join(demos, "pose", "a_pose_bone.txt")).astype(np.float32)
    out["bbox.values"] = np.array([0.8, 0.64, 1.0], dtype=np.float32)
    np.savez(args.out, **out)
    print("wrote", args.out, {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
