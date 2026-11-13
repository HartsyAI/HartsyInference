"""Reference values for P3-SAM's automatic-segmentation geometry: upstream's own `mesh_sam` of `P3-SAM/demo/auto_mask.py` on the CPU, with a fake network.

Executes `auto_mask.py` of a checkout of Tencent-Hunyuan/Hunyuan3D-Part (pinned in docs/Research/HUNYUAN3D_PART_ARCHITECTURE.md) and runs `mesh_sam` end to end on a mesh. The network is the only thing
replaced: `get_feat` returns nothing and `get_mask` returns three deterministic ball masks per prompt point (radii 0.2, 0.45 and 0.9 in the normalised cube) with a predicted IoU that favours the ball
holding about a tenth of the points, so the ranking, the NMS, the merging, the face voting, `fix_label`, the connected regions and the post-processing all run on real, varied masks. Upstream's other
imports are stand-ins: `fpsample.fps_sampling` is exact farthest-point sampling from point 0 (the real library is not installed and its sampling is not what is compared), `numba.njit` is the
identity, `sklearn` and `tqdm` are only touched by options that are off, and the package `model` is a stub. Surface sampling (`trimesh.sample.sample_surface`) and mesh cleaning are upstream's own (but for trimesh's winding fix, which needs scipy).

    python -I dump_p3sam_geometry.py --upstream <Hunyuan3D-Part checkout>/P3-SAM/demo --mesh <mesh.obj> --out <fixture.npz>

The fixture holds the inputs of the geometry stage as upstream computed them (`vertices` and `faces` of the cleaned mesh, the normalised sampled `points`, the `face_idx` each point came from, the
per-prompt selected `masks` and their `iou`) and its results, with and without `post_process` (`face_ids_0`, `face_ids_1`, `aabb_0`, `aabb_1`; boxes as [parts, 2, 3]).
"""
import argparse
import os
import sys
import time
import types

import numpy as np
import trimesh


def stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module


def exact_fps(points, count):
    """Farthest-point sampling from point 0 (what fpsample.fps_sampling stands in for)."""
    chosen = np.zeros(count, dtype=np.int64)
    distance = np.full(len(points), np.inf)
    for k in range(1, count):
        distance = np.minimum(distance, np.sum((points - points[chosen[k - 1]]) ** 2, axis=1))
        chosen[k] = int(np.argmax(distance))
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, help="the P3-SAM/demo folder")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--radii", default="0.2,0.45,0.9", help="the three ball radii of the fake masks")
    args = parser.parse_args()

    stub("fpsample", fps_sampling=exact_fps)
    stub("numba", njit=lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f)), jit=lambda *a, **k: (lambda f: f))
    stub("sklearn"), stub("sklearn.decomposition", PCA=None)
    class tqdm:  # a silent progress bar
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable if self.iterable is not None else [])

        def update(self, *args, **kwargs):
            pass

        def set_postfix(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    stub("tqdm", tqdm=tqdm)
    stub("model", build_P3SAM=None, load_state_dict=None)

    source = open(os.path.join(args.upstream, "auto_mask.py"), encoding="utf-8").read()
    source = source.replace("sys.path.append('..')", "")
    namespace = {"__name__": "auto_mask_upstream"}
    exec(compile(source, "auto_mask.py", "exec"), namespace)

    class Timer:  # upstream's Timer prints its (Chinese) stage names, which a Windows console cannot encode
        def __init__(self, name=None, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    namespace["Timer"] = Timer

    # trimesh's fix_normals (called by Trimesh.process inside upstream's clean_mesh) needs scipy, which is not installed here; the winding of the input mesh is already consistent.
    trimesh.Trimesh.fix_normals = lambda self, *args, **kwargs: None

    captured = {"masks": [], "iou": []}

    real_sample = trimesh.sample.sample_surface

    def sample_surface(mesh, count, **kwargs):
        points, face_idx = real_sample(mesh, count, **kwargs)
        captured["face_idx"] = np.asarray(face_idx)
        return points, face_idx

    trimesh.sample.sample_surface = sample_surface

    real_normalize = namespace["normalize_pc"]

    def normalize_pc(pc):
        out = real_normalize(pc)
        captured["points"] = np.asarray(out)
        return out

    namespace["normalize_pc"] = normalize_pc
    namespace["get_feat"] = lambda model, points, normals: None

    radii = tuple(float(r) for r in args.radii.split(","))

    def get_mask(model, feats, points, prompts, iter=1):
        d = np.linalg.norm(points[:, None, :] - prompts[None, :, :], axis=-1)  # [N, K]
        masks = [d < r for r in radii]
        fractions = np.stack([m.mean(axis=0) for m in masks], axis=-1)  # [K, 3]
        jitter = 1e-4 * np.sin(prompts.sum(axis=-1) * 37.0)[:, None]
        iou = 0.9 - np.abs(fractions - 0.1) + jitter
        chosen = np.argmax(iou, axis=-1)
        for k in range(len(prompts)):
            captured["masks"].append(np.stack(masks, axis=-1)[:, k, chosen[k]].copy())
            captured["iou"].append(float(iou[k, chosen[k]]))
        return masks[0], masks[1], masks[2], iou

    namespace["get_mask"] = get_mask

    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    out = {}
    for post in (0, 1):
        captured["masks"].clear()
        captured["iou"].clear()
        start = time.time()
        aabb, face_ids, used = namespace["mesh_sam"]((None, None), mesh.copy(), None, post_process=bool(post), threshold=0.95, seed=42)
        print(f"post_process={bool(post)}: {len(used.faces)} faces, {len(np.unique(face_ids[face_ids >= 0]))} parts, {time.time() - start:.0f} s", flush=True)
        out[f"face_ids_{post}"] = np.asarray(face_ids, dtype=np.int32)
        out[f"aabb_{post}"] = np.asarray(aabb, dtype=np.float32)
        if post == 0:
            out["vertices"] = np.asarray(used.vertices, dtype=np.float32)
            out["faces"] = np.asarray(used.faces, dtype=np.int32)
            out["points"] = captured["points"].astype(np.float32)
            out["face_idx"] = captured["face_idx"].astype(np.int32)
            out["masks"] = np.stack(captured["masks"]).astype(np.uint8)
            out["iou"] = np.asarray(captured["iou"], dtype=np.float32)
    np.savez(args.out, **out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
