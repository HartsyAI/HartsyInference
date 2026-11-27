"""Reference values for the NAF upsampler Pixal3D uses, produced by upstream's own modules.

Runs `src/model/naf.py`, `src/layers/{attentions,convolutions,rope}.py` of a checkout of https://github.com/valeoai/NAF (pinned in
docs/Research/PIXAL3D_ARCHITECTURE.md) with the released weights. Two things differ from a normal run, both to keep the dump safe and
dependency-free:

* `naf_release.pth` is read by a restricted unpickler that allows only `OrderedDict` and torch's tensor rebuild call (nothing else in the
  file can be executed), not by `torch.load`.
* `natten` and `einops` are not installed here. `natten.na2d` is replaced by a brute-force transcription of NATTEN's definition (dilated
  neighbourhood attention whose window slides inward at the borders, scale `head_dim ** -0.5`), and `einops.rearrange` by the handful of
  reshapes upstream's code uses.

    python dump_pixal3d_naf.py --naf <NAF checkout> --weights <naf_release.pth> --out <fixture.json>
"""
import argparse
import collections
import json
import math
import pickle
import sys
import types
import zipfile

import torch


PROBE_CHANNELS = [0, 1, 31, 32, 63, 64, 128, 255]


class SafeUnpickler(pickle.Unpickler):
    """Allows only what a flat torch state_dict needs; any other global raises instead of running."""

    def __init__(self, handle, storages):
        super().__init__(handle)
        self._storages = storages

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        if module == "torch" and name == "FloatStorage":
            return "float32"
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")

    def persistent_load(self, pid):
        kind, dtype, key, _location, numel = pid
        if kind != "storage" or dtype != "float32":
            raise pickle.UnpicklingError(f"unsupported storage {pid[:2]}")
        return self._storages(key, numel)


def _rebuild_tensor(storage, offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
    return torch.as_strided(storage, tuple(size), tuple(stride), offset).clone()


def read_state_dict(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        root = names[0].split("/")[0]

        def storage(key, numel):
            raw = archive.read(f"{root}/data/{key}")
            return torch.frombuffer(bytearray(raw), dtype=torch.float32, count=numel)

        with archive.open(f"{root}/data.pkl") as handle:
            return SafeUnpickler(handle, storage).load()


def install_stubs(naf_root):
    """Provides `einops.rearrange` for the patterns NAF uses and a brute-force `natten.na2d`."""

    def rearrange(x, pattern, **axes):
        if pattern == "b (n d) h w -> b h w n d":
            b, c, h, w = x.shape
            n = axes["n"]
            return x.reshape(b, n, c // n, h, w).permute(0, 3, 4, 1, 2)
        if pattern == "b h w n d -> b (n d) h w":
            b, h, w, n, d = x.shape
            return x.permute(0, 3, 4, 1, 2).reshape(b, n * d, h, w)
        if pattern == "b (n d) h w -> b n (h w) d":
            b, c, h, w = x.shape
            n = axes["n"]
            return x.reshape(b, n, c // n, h * w).permute(0, 1, 3, 2)
        if pattern == "b n (h w) d -> b (n d) h w":
            b, n, hw, d = x.shape
            h, w = axes["h"], axes["w"]
            return x.permute(0, 1, 3, 2).reshape(b, n * d, h, w)
        raise NotImplementedError(pattern)

    einops = types.ModuleType("einops")
    einops.rearrange = rearrange
    sys.modules["einops"] = einops

    def window_start(index, size, kernel):
        half = kernel // 2
        return max(index - half, 0) + (index + half >= size) * (size - index - half - 1)

    def na2d(q, k, v, kernel_size, dilation, stride=1, backend=None):
        # q, k: [B, H, W, heads, D]; v: [B, H, W, heads, Dv]. Brute force from the definition.
        b, h, w, heads, d = q.shape
        kh, kw = kernel_size
        dh, dw = dilation
        out = torch.zeros(b, h, w, heads, v.shape[-1], dtype=q.dtype)
        scale = d ** -0.5
        for i in range(h):
            for j in range(w):
                # The dilated grid this pixel lives on has size ceil((h - i % dh) / dh); the window slides inward inside it.
                sub_h = (h - i % dh + dh - 1) // dh
                sub_w = (w - j % dw + dw - 1) // dw
                si = window_start(i // dh, sub_h, kh)
                sj = window_start(j // dw, sub_w, kw)
                rows = [(si + a) * dh + i % dh for a in range(kh)]
                cols = [(sj + c) * dw + j % dw for c in range(kw)]
                keys = k[:, rows][:, :, cols]  # [B, kh, kw, heads, D]
                values = v[:, rows][:, :, cols]
                logits = torch.einsum("bnd,bxynd->bnxy", q[:, i, j], keys) * scale
                weights = logits.flatten(2).softmax(-1).reshape_as(logits)
                out[:, i, j] = torch.einsum("bnxy,bxynd->bnd", weights, values)
        return out

    natten = types.ModuleType("natten")
    natten.na2d = na2d
    sys.modules["natten"] = natten  # no natten.functional, so upstream takes its "recent natten" branch
    sys.path.insert(0, naf_root)


def flat(tensor):
    """Row-major values at float32's round-trip precision (9 significant digits), which keeps the fixtures small."""
    return [float(f"{v:.9g}") for v in tensor.reshape(-1).tolist()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--naf", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    install_stubs(args.naf)
    from src.model.naf import NAF  # noqa: E402  (needs the stubs and the checkout on sys.path)

    model = NAF()
    model.load_state_dict(read_state_dict(args.weights))
    model.eval()

    generator = torch.Generator().manual_seed(11)
    cases = []
    for image_size, output_size, feature_size, channels in [(64, 32, 16, 8), (48, 48, 12, 4)]:
        image = torch.rand(1, 3, image_size, image_size, generator=generator)
        features = torch.randn(1, channels, feature_size, feature_size, generator=generator)
        with torch.no_grad():
            query = model.image_encoder(image, output_size=(output_size, output_size))
            out = model(image, features, (output_size, output_size))
        cases.append({
            "image_size": image_size,
            "output_size": output_size,
            "feature_size": feature_size,
            "channels": channels,
            "image": flat(image[0]),
            # Feature cells as rows [fh * fw, channels], the layout of DINOv3 patch tokens.
            "feature_rows": flat(features[0].permute(1, 2, 0)),
            # A few channels of the rope'd query map (head starts and ends) to localise a mismatch without shipping 256 of them.
            "query_channels": PROBE_CHANNELS,
            "query_probe": flat(query[0, PROBE_CHANNELS]),
            "expected": flat(out[0]),
        })

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"torch": torch.__version__, "cases": cases}, handle, separators=(",", ":"))
    print(f"wrote {args.out}: {len(cases)} cases")


if __name__ == "__main__":
    main()
