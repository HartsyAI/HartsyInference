"""Reference values for the Hunyuan3D-Paint 2.1 bake: upstream's own `MeshRender.back_project` and `fast_bake_texture` on the CPU.

Runs `hy3dpaint/DifferentiableRenderer/MeshRender.py` and `camera_utils.py` of a checkout of Tencent-Hunyuan/Hunyuan3D-2.1 (pinned in docs/Research/HUNYUAN3D_PAINT_ARCHITECTURE.md) on a UV-mapped mesh
and six view images written by the C# side (`PaintBakeParityTests.WriteInputsForTheOracle`: set `HARTSY_PAINT_BAKE_INPUT_DIR`, `HARTSY_PAINT_BAKE_MESH` and `HARTSY_PAINT_BAKE_REF`), with the default configuration of the 2.1 pipeline (orthographic camera at distance 1.45, `back_sample` bake, `face` shader).
Things that differ from a normal run, all to keep the dump CPU-only and dependency-light:

* `custom_rasterizer` (a CUDA extension) is replaced by a transcription of its kernel (`rasterizer.cpp` / `rasterizer_gpu.cu` of the same commit, whose CPU and GPU versions run the same algorithm,
  and `render.py`): the edge-function barycentrics of the pixel centre, depth quantised to 2^18 steps, one `min` token per pixel, then perspective-corrected barycentrics. The float32 arithmetic and
  the integer truncations are kept.
* `trimesh.geometry.mean_vertex_normals` (used for the texel normals, which `back_sample` does not read) is replaced by its definition: the normalised sum of the unit face normals of the vertex.
* `bpy` and the C++ inpainting extension are not loaded (upstream wraps both imports in try/except); `cv2` is the real OpenCV when it can be imported (needed for `Canny`, which makes the edge mask), else
  a numpy stand-in whose result is NOT comparable and is recorded in the fixture as `canny = 0`.

    python -I dump_paint_bake.py --upstream <checkout>/hy3dpaint --in <the input directory> --out <fixture.npz>

The input directory (`tex` is the texture size, `res` the view resolution) holds `verts.f32` (N x 3), `uvs.f32` (N x 2), `idx.i32` (F x 3), `view<k>.f32` (R x R x 3, row 0 first), `meta.json` (texture size, resolution, the views' elevation, azimuth and weight).
The fixture holds the inputs (`verts`, `uvs`, `idx`, `views` as elevation, azimuth and weight, `image<k>`, `texture_size` as texture size and render resolution) and, per view, the texture-space `color<k>`, `cos<k>` and `boundary<k>` (kept for a later comparison of the edge mask) of `back_project`, the merged `merged` texture and `painted` mask of `fast_bake_texture` (cosines raised to the
power 4 and scaled by the view weight, as `bake_texture` does), the texel positions `tex_position` and `tex_grid`, and `canny` (1 when real OpenCV was used).
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F


def stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module


# --- custom_rasterizer ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

MAXINT = 2147483647
F32 = np.float32


def _area2(a, b, c):
    return (c[0] - a[0]) * (b[1] - a[1]) - (b[0] - a[0]) * (c[1] - a[1])


def rasterize_image(V, Fc, width, height):
    """The kernel of rasterizer.cpp: V is [N, 4] clip positions (float32), Fc [F, 3] int32. Returns findices [H, W] (face + 1, 0 = background) and barycentric [H, W, 3]."""
    V = V.astype(F32)
    w = V[:, 3]
    sx = ((V[:, 0] / w) * F32(0.5) + F32(0.5)) * F32(width - 1) + F32(0.5)
    sy = (F32(0.5) + F32(0.5) * (V[:, 1] / w)) * F32(height - 1) + F32(0.5)
    sz = (V[:, 2] / w) * F32(0.49999) + F32(0.5)
    maxint = np.int64(MAXINT) * np.int64(MAXINT) + np.int64(MAXINT - 1)
    zbuf = np.full(height * width, maxint, dtype=np.int64)
    for f in range(Fc.shape[0]):
        i0, i1, i2 = Fc[f]
        x0, y0, z0, x1, y1, z1, x2, y2, z2 = sx[i0], sy[i0], sz[i0], sx[i1], sy[i1], sz[i1], sx[i2], sy[i2], sz[i2]
        x_min, x_max = min(x0, x1, x2), max(x0, x1, x2)
        y_min, y_max = min(y0, y1, y2), max(y0, y1, y2)
        px_lo, px_hi = max(int(x_min), 0), min(int(math.ceil(float(x_max + F32(1)))), width)  # for (int px = x_min; px < x_max + 1; ++px), px in [0, width)
        py_lo, py_hi = max(int(y_min), 0), min(int(math.ceil(float(y_max + F32(1)))), height)
        if px_lo >= px_hi or py_lo >= py_hi:
            continue
        px, py = np.meshgrid(np.arange(px_lo, px_hi, dtype=np.int32), np.arange(py_lo, py_hi, dtype=np.int32), indexing="ij")
        px, py = px.reshape(-1), py.reshape(-1)
        ptx, pty = px.astype(F32) + F32(0.5), py.astype(F32) + F32(0.5)
        area = _area2((x0, y0), (x1, y1), (x2, y2))
        if area == 0:
            continue
        beta_tri = (F32(x2) - F32(x0)) * (pty - F32(y0)) - (ptx - F32(x0)) * (F32(y2) - F32(y0))  # calculateSignedArea2(a, p, c)
        gamma_tri = (ptx - F32(x0)) * (F32(y1) - F32(y0)) - (F32(x1) - F32(x0)) * (pty - F32(y0))  # calculateSignedArea2(a, b, p)
        inv = F32(1.0 / float(area))
        beta, gamma = beta_tri * inv, gamma_tri * inv
        alpha = F32(1.0) - beta - gamma
        inside = (alpha >= 0) & (alpha <= 1) & (beta >= 0) & (beta <= 1) & (gamma >= 0) & (gamma <= 1)
        if not inside.any():
            continue
        depth = alpha[inside] * F32(z0) + beta[inside] * F32(z1) + gamma[inside] * F32(z2)
        keep = depth >= 0  # depth_thres is 0 without a depth prior
        if not keep.any():
            continue
        z_quant = (depth[keep] * F32(2 << 17)).astype(np.int64)  # int z_quantize = depth * (2<<17), truncation toward zero (depth >= 0)
        token = z_quant * np.int64(MAXINT) + np.int64(f + 1)
        pix = (py[inside][keep] * width + px[inside][keep]).astype(np.int64)
        np.minimum.at(zbuf, pix, token)
    face = zbuf % MAXINT
    background = face == (MAXINT - 1)
    findices = np.where(background, 0, face).astype(np.int32)
    bary = np.zeros((height * width, 3), dtype=F32)
    ids = np.nonzero(~background)[0]
    f = findices[ids].astype(np.int64) - 1
    i0, i1, i2 = Fc[f, 0], Fc[f, 1], Fc[f, 2]
    pt = np.stack([(ids % width).astype(F32) + F32(0.5), (ids // width).astype(F32) + F32(0.5)], 1)
    a = np.stack([sx[i0], sy[i0]], 1)
    b = np.stack([sx[i1], sy[i1]], 1)
    c = np.stack([sx[i2], sy[i2]], 1)

    def area2(p, q, r):
        return (r[:, 0] - p[:, 0]) * (q[:, 1] - p[:, 1]) - (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1])

    beta_tri, gamma_tri, area = area2(a, pt, c), area2(a, b, pt), area2(a, b, c)
    degenerate = area == 0
    inv = (F32(1.0) / np.where(degenerate, F32(1), area).astype(np.float64)).astype(F32)
    beta, gamma = beta_tri * inv, gamma_tri * inv
    alpha = F32(1.0) - beta - gamma
    b0, b1, b2 = alpha / w[i0], beta / w[i1], gamma / w[i2]
    scale = F32(1.0) / (b0 + b1 + b2)
    out = np.stack([b0 * scale, b1 * scale, b2 * scale], 1)
    out[degenerate] = 0  # a zero-area face never wins a pixel (its barycentrics are out of bounds), so this is not reached
    bary[ids] = out
    return findices.reshape(height, width), bary.reshape(height, width, 3)


def make_custom_rasterizer():
    def rasterize(pos, tri, resolution, clamp_depth=None, use_depth_prior=0):
        findices, barycentric = rasterize_image(pos[0].numpy(), tri.numpy().astype(np.int32), resolution[1], resolution[0])
        return torch.from_numpy(findices), torch.from_numpy(barycentric)

    def interpolate(col, findices, barycentric, tri):
        f = findices - 1 + (findices == 0)
        vcol = col[0, tri.long()[f.long()]]
        result = barycentric.view(*barycentric.shape, 1) * vcol
        result = torch.sum(result, axis=-2)
        return result.view(1, *result.shape)

    return stub("custom_rasterizer", rasterize=rasterize, interpolate=interpolate)


# --- trimesh and cv2 --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def mean_vertex_normals(vertex_count, faces, face_normals):
    faces, face_normals = np.asarray(faces).astype(np.int64), np.asarray(face_normals)
    out = np.zeros((vertex_count, 3), dtype=np.float64)
    for k in range(3):
        np.add.at(out, faces[:, k], face_normals)
    length = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(length == 0, 1, length)


def numpy_canny(image, low, high):
    """A stand-in for cv2.Canny: 3x3 Sobel, L1 magnitude, double threshold with hysteresis, and no non-maximum suppression, so its edges are thicker than OpenCV's; results that depend on it are not comparable."""
    img = image.reshape(image.shape[0], image.shape[1]).astype(np.float32)
    p = np.pad(img, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    mag = np.abs(gx) + np.abs(gy)
    strong = mag > high
    weak = mag > low
    out = strong.copy()
    for _ in range(64):
        grown = out | (weak & (np.pad(out, 1)[:-2, 1:-1] | np.pad(out, 1)[2:, 1:-1] | np.pad(out, 1)[1:-1, :-2] | np.pad(out, 1)[1:-1, 2:]))
        if (grown == out).all():
            break
        out = grown
    return (out * 255).astype(np.uint8)


def load_module(path, name, package):
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, help="the hy3dpaint folder of the checkout")
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    real_canny = True
    try:
        import cv2  # noqa: F401
    except ImportError:
        real_canny = False
        stub("cv2", Canny=numpy_canny, inpaint=None, resize=None, INPAINT_NS=0)
        print("WARNING: OpenCV is not installed; Canny is a numpy stand-in and the edge-dependent values are not comparable", flush=True)
    stub("trimesh", geometry=types.SimpleNamespace(mean_vertex_normals=mean_vertex_normals))
    make_custom_rasterizer()

    renderer_dir = os.path.join(args.upstream, "DifferentiableRenderer")
    package = types.ModuleType("DifferentiableRenderer")
    package.__path__ = [renderer_dir]
    sys.modules["DifferentiableRenderer"] = package
    load_module(os.path.join(renderer_dir, "camera_utils.py"), "DifferentiableRenderer.camera_utils", "DifferentiableRenderer")
    # mesh_utils imports bpy; MeshRender wraps that import in try/except, so it simply fails to load here.
    mesh_render = load_module(os.path.join(renderer_dir, "MeshRender.py"), "DifferentiableRenderer.MeshRender", "DifferentiableRenderer")

    meta = json.load(open(os.path.join(args.inp, "meta.json")))
    tex, res = meta["tex"], meta["res"]
    verts = np.fromfile(os.path.join(args.inp, "verts.f32"), dtype=np.float32).reshape(-1, 3)
    uvs = np.fromfile(os.path.join(args.inp, "uvs.f32"), dtype=np.float32).reshape(-1, 2)
    idx = np.fromfile(os.path.join(args.inp, "idx.i32"), dtype=np.int32).reshape(-1, 3)

    start = time.time()
    render = mesh_render.MeshRender(camera_distance=1.45, camera_type="orth", default_resolution=res, texture_size=tex, bake_mode="back_sample", shader_type="face", device="cpu")
    render.set_mesh(verts.copy(), idx.copy(), uvs.copy(), idx.copy())
    print(f"mesh set and texel positions extracted in {time.time() - start:.0f} s; {render.tex_position.shape[0]} covered texels", flush=True)

    out = {"canny": np.array([1 if real_canny else 0], dtype=np.int32), "tex_grid": render.tex_grid.numpy().astype(np.int32), "tex_position": render.tex_position.numpy(),
           "verts": verts, "uvs": uvs, "idx": idx, "views": np.array([[v["elev"], v["azim"], v["weight"]] for v in meta["views"]], dtype=np.float32), "texture_size": np.array([tex, res], dtype=np.int32)}
    textures, cos_maps = [], []
    for k, view in enumerate(meta["views"]):
        image = np.fromfile(os.path.join(args.inp, f"view{k}.f32"), dtype=np.float32).reshape(res, res, 3)
        t0 = time.time()
        with torch.no_grad():
            texture, cos_map, boundary = render.back_project(torch.from_numpy(image), view["elev"], view["azim"])
        out[f"image{k}"] = image
        out[f"color{k}"], out[f"cos{k}"], out[f"boundary{k}"] = texture.numpy(), cos_map.numpy(), boundary.numpy()
        weighted = view["weight"] * (cos_map ** 4)
        textures.append(texture)
        cos_maps.append(weighted)
        print(f"view {k} (elev {view['elev']}, azim {view['azim']}): {(cos_map > 0).sum().item()} painted texels, {time.time() - t0:.0f} s", flush=True)
    merged, trust = render.fast_bake_texture(textures, cos_maps)
    out["merged"], out["painted"] = merged.numpy(), trust.numpy().astype(np.float32)
    np.savez(args.out, **out)
    print("wrote", args.out, f"in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
