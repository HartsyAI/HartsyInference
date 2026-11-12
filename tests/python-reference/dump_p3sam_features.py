"""Reference values for P3-SAM (Hunyuan3D-Part), produced by upstream's own Sonata modules on the CPU.

Runs `XPart/partgen/models/sonata` (the Point Transformer V3 of `facebook/sonata`, pinned in docs/Research/HUNYUAN3D_PART_ARCHITECTURE.md) with the
Sonata tensors of `tencent/Hunyuan3D-Part` `p3sam/p3sam.safetensors`, then transcribes `P3-SAM/demo/auto_mask.py::get_feat` and `P3-SAM/model.py` (the
`mlp` and the mask and IoU heads) on fixed inputs. What differs from a normal run, to keep it CPU-only and dependency-free:

* `spconv` is replaced by a brute-force `SubMConv3d` (a cross-correlation over the 27 neighbours present, kernel index k meaning offset k - 1 on each of the
  three index columns, weights `[out, 3, 3, 3, in]`), `torch_scatter.segment_csr` by a loop over segments, `timm.layers.DropPath` by the identity (inference)
  and `addict.Dict` by a dict with attribute access. The `spconv` convention is an assumption (the real library is not installed): a mismatch with the C#
  side would show up as a difference in the first stage, not later.
* `enable_flash` is false (flash-attn is not installed; the dense path computes the same attention within patches) and `shuffle_orders` is false, including on
  the pooling layers, which upstream builds without that argument (so they shuffle by default, at inference too, and upstream's features from stage 1 on
  are random from run to run). The four serialisation orders are used in their fixed sequence. Hunyuan3D-Part loads Sonata with the config stored in
  `sonata.pth`, a pickle that was not read.
* The input transform (`CenterShift`, `GridSample` at 0.005 with FNV hash, `NormalizeColor`) is transcribed with the deterministic first-point pick in
  place of the random pick, so the dump is reproducible; the selected indices, the inverse map and the grid coordinates are dumped too.

    python dump_p3sam_features.py --part <Hunyuan3D-Part checkout> --weights <p3sam.safetensors> --out <fixture.safetensors>

The fixture holds the inputs and these outputs: `selected`, `inverse`, `grid_coord`, `stage0..stage4` (each stage's features after its blocks, finest rows are
the voxels), `gathered` (1232 channels), `features` (512 per point), and for two prompts `mask_{k}`, `iou_{k}`. Times are printed.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import importlib.util
import json
import sys
import time
import types

import numpy as np
import torch
import torch.nn as nn


class AttrDict(dict):
    """The part of addict.Dict that Sonata's Point uses."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


class SparseConvTensor:
    def __init__(self, features, indices, spatial_shape, batch_size):
        self.features = features
        self.indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size

    def replace_feature(self, feature):
        other = SparseConvTensor(feature, self.indices, self.spatial_shape, self.batch_size)
        return other


class SubMConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, indice_key=None, algo=None):
        super().__init__()
        assert kernel_size == 3 and stride == 1 and dilation == 1 and groups == 1
        self.weight = nn.Parameter(torch.empty(out_channels, 3, 3, 3, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        indices = x.indices.long().cpu().numpy()
        lookup = {tuple(row): i for i, row in enumerate(indices)}
        feats = x.features
        out = torch.zeros(feats.shape[0], self.weight.shape[0], dtype=feats.dtype)
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    rows, neighbours = [], []
                    for i, (n, i0, i1, i2) in enumerate(indices):
                        j = lookup.get((n, i0 + a - 1, i1 + b - 1, i2 + c - 1))
                        if j is not None:
                            rows.append(i)
                            neighbours.append(j)
                    if rows:
                        out[rows] += feats[neighbours] @ self.weight[:, a, b, c, :].T
        if self.bias is not None:
            out = out + self.bias
        return x.replace_feature(out)


def segment_csr(src, indptr, reduce="sum"):
    out = []
    for s in range(len(indptr) - 1):
        chunk = src[indptr[s]:indptr[s + 1]]
        if reduce == "max":
            out.append(chunk.max(dim=0).values)
        elif reduce == "min":
            out.append(chunk.min(dim=0).values)
        elif reduce == "mean":
            out.append(chunk.mean(dim=0))
        else:
            out.append(chunk.sum(dim=0))
    return torch.stack(out)


def install_stubs():
    spconv = types.ModuleType("spconv")
    pytorch = types.ModuleType("spconv.pytorch")
    pytorch.SparseConvTensor = SparseConvTensor
    pytorch.SubMConv3d = SubMConv3d
    modules = types.ModuleType("spconv.pytorch.modules")
    modules.is_spconv_module = lambda module: isinstance(module, SubMConv3d)
    pytorch.modules = modules
    spconv.pytorch = pytorch
    sys.modules.update({"spconv": spconv, "spconv.pytorch": pytorch, "spconv.pytorch.modules": modules})
    scatter = types.ModuleType("torch_scatter")
    scatter.segment_csr = segment_csr
    sys.modules["torch_scatter"] = scatter

    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()

        def forward(self, x):
            return x

    timm = types.ModuleType("timm")
    layers = types.ModuleType("timm.layers")
    layers.DropPath = DropPath
    timm.layers = layers
    sys.modules.update({"timm": timm, "timm.layers": layers})
    addict = types.ModuleType("addict")
    addict.Dict = AttrDict
    sys.modules["addict"] = addict


def load_sonata(part_root):
    """Imports partgen.models.sonata.{model,structure,module,utils} without the package __init__ (which pulls in the transforms and the data pipeline)."""
    base = os.path.join(part_root, "XPart", "partgen", "models", "sonata")
    for name in ("partgen", "partgen.models", "partgen.models.sonata"):
        package = types.ModuleType(name)
        package.__path__ = [os.path.join(part_root, "XPart", *name.split("."))] if name != "partgen.models.sonata" else [base]
        sys.modules[name] = package
    loaded = {}
    for name in ("utils", "serialization", "structure", "module", "model"):
        full = f"partgen.models.sonata.{name}"
        path = os.path.join(base, name, "__init__.py") if name == "serialization" else os.path.join(base, f"{name}.py")
        spec = importlib.util.spec_from_file_location(full, path, submodule_search_locations=[os.path.join(base, name)] if name == "serialization" else None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded["model"]


FNV_OFFSET, FNV_PRIME = 14695981039346656037, 1099511628211


def fnv(grid):
    h = np.full(grid.shape[0], FNV_OFFSET, dtype=np.uint64)
    g = grid.astype(np.uint64)
    for j in range(3):
        h = h * np.uint64(FNV_PRIME)
        h = np.bitwise_xor(h, g[:, j])
    return h


def transform(coord, normal):
    """CenterShift(apply_z), GridSample(0.005, fnv) with the first point of each voxel, NormalizeColor; returns the model input and the maps."""
    x_min, y_min, z_min = coord.min(axis=0)
    x_max, y_max, _ = coord.max(axis=0)
    coord = coord - np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min])
    scaled = coord / 0.005
    grid = np.floor(scaled).astype(np.int64)
    grid -= grid.min(axis=0)
    key = fnv(grid)
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    unique, inverse_sorted, counts = np.unique(sorted_key, return_inverse=True, return_counts=True)
    starts = np.cumsum(np.insert(counts, 0, 0)[:-1])
    selected = order[starts]
    inverse = np.zeros(len(order), dtype=np.int64)
    inverse[order] = inverse_sorted
    feat = np.concatenate([coord[selected], np.ones((len(selected), 3)) / 255.0, normal[selected]], axis=1)
    return coord[selected], grid[selected], feat, selected, inverse


def mlp(state, prefix):
    layers = []
    for index in (0, 2, 4):
        w, b = state[f"{prefix}.{index}.weight"], state[f"{prefix}.{index}.bias"]
        layer = nn.Linear(w.shape[1], w.shape[0])
        layer.weight.data, layer.bias.data = w.clone(), b.clone()
        layers.append(layer)
        if index != 4:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


def heads(state, feats, points, prompt):
    """P3-SAM/model.py forward (feats [N, 512], points [N, 3], prompt [3]) with iter = 1; returns the three masks per point and the three IoUs."""
    n = points.shape[0]
    seg = torch.cat([feats, points, prompt.expand(n, 3)], dim=-1)
    seg_mlp = [mlp(state, f"seg_mlp_{i}") for i in (1, 2, 3)]
    stage1 = torch.stack([m(seg).squeeze(-1) for m in seg_mlp], dim=-1)
    seg2 = torch.cat([seg, stage1], dim=-1)
    glob = torch.max(mlp(state, "seg_s2_mlp_g")(seg2), dim=0).values.expand(n, -1)
    seg3 = torch.cat([glob, seg2], dim=-1)
    stage2 = torch.stack([mlp(state, f"seg_s2_mlp_{i}")(seg3).squeeze(-1) for i in (1, 2, 3)], dim=-1)
    iou_in = torch.cat([glob, seg, stage2], dim=-1)
    pooled = torch.max(mlp(state, "iou_mlp")(iou_in), dim=0).values
    iou = torch.sigmoid(mlp(state, "iou_mlp_out")(pooled))
    return torch.sigmoid(stage2), iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--points", type=int, default=3000)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    from safetensors.torch import load_file, save_file

    install_stubs()
    sonata = load_sonata(args.part)
    config = json.load(open(os.path.join(args.part, "XPart", "partgen", "config", "sonata.json")))
    config.update(enable_flash=False, shuffle_orders=False, drop_path=0.0)
    state = load_file(args.weights)
    model = sonata.PointTransformerV3(**config)
    sonata_state = {k[len("sonata."):]: v for k, v in state.items() if k.startswith("sonata.")}
    missing, unexpected = model.load_state_dict(sonata_state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"Sonata weights do not match: missing {missing[:5]}, unexpected {unexpected[:5]}")
    model.eval()
    # PointTransformerV3.__init__ builds its GridPooling layers without passing shuffle_orders, so they keep the default True and permute the serialisation orders
    # at random on every pooling, even at inference: upstream's features from the second stage on differ between runs. Pin it off to get a reproducible oracle.
    for module in model.modules():
        if isinstance(module, sonata.GridPooling):
            module.shuffle_orders = False

    rng = np.random.default_rng(1)
    count = args.points
    theta, phi = np.arccos(1 - 2 * rng.random(count)), 2 * np.pi * rng.random(count)
    x, y, z = np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)
    points = np.stack([0.6 * x, 1.0 * y, 0.8 * z], axis=1)
    normals = np.stack([x / 0.6, y / 1.0, z / 0.8], axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    coord, grid, feat, selected, inverse = transform(points, normals)
    data = {
        "coord": torch.tensor(coord, dtype=torch.float32),
        "grid_coord": torch.tensor(grid, dtype=torch.int32),
        "feat": torch.tensor(feat, dtype=torch.float32),
        "offset": torch.tensor([coord.shape[0]]),
    }
    start = time.time()
    point = model(data)
    print(f"sonata forward over {coord.shape[0]} voxels: {time.time() - start:.1f} s")
    out = {"points": torch.tensor(points, dtype=torch.float32), "normals": torch.tensor(normals, dtype=torch.float32),
           "selected": torch.tensor(selected, dtype=torch.float32), "inverse": torch.tensor(inverse, dtype=torch.float32),
           "grid_coord": torch.tensor(grid, dtype=torch.float32)}
    stages = []
    grids = []
    walker = point
    while True:
        stages.append(walker.feat.clone())
        grids.append(walker.grid_coord.clone().float())
        if "pooling_parent" not in walker.keys():
            break
        walker = walker["pooling_parent"]
    stages.reverse()
    grids.reverse()
    for s, f in enumerate(stages):
        out[f"stage{s}"] = f
        out[f"grid{s}"] = grids[s]
    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inv = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inv]], dim=-1)
        point = parent
    gathered = point.feat
    out["gathered"] = gathered
    voxel_features = mlp(state, "mlp")(gathered)
    features = voxel_features[torch.tensor(inverse)]
    out["features"] = features
    unit = torch.tensor(points, dtype=torch.float32)
    for k, index in enumerate((0, count // 2)):
        start = time.time()
        masks, iou = heads(state, features, unit, unit[index])
        print(f"prompt {k} (point {index}): {time.time() - start:.1f} s, iou {iou.tolist()}")
        out[f"mask_{k}"] = masks
        out[f"iou_{k}"] = iou
        out[f"prompt_{k}"] = unit[index]
    save_file({k: v.detach().float().contiguous() for k, v in out.items()}, args.out)
    print("wrote", args.out, {k: tuple(v.shape) for k, v in out.items()})


if __name__ == "__main__":
    main()
