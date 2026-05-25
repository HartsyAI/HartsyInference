"""
Converts Ultralytics YOLOv8 .pt checkpoints to safetensors for SharpInference.Vision.

Folding: BatchNorm running stats + affine are folded into the preceding Conv2D weight
and bias so inference becomes a plain Conv2D op (no BN kernel needed). The math:

    w_folded[c_out, ...] = w_conv[c_out, ...] * (gamma[c_out] / sqrt(var[c_out] + eps))
    b_folded[c_out]      = beta[c_out] - mean[c_out] * (gamma[c_out] / sqrt(var[c_out] + eps))

Key renaming: nothing — Ultralytics' state dict keys already match SharpInference's
expected naming (`model.{layer}.cv1.conv.weight` etc.) once BN params are dropped.
The detect head's three projection convs (`model.22.cv2.{0,1,2}.2.weight`) keep their
plain `Conv2d` form (no BN to fold).

Usage:
    pip install torch safetensors ultralytics
    python convert_yolov8_pt_to_safetensors.py <input.pt> <output.safetensors>

The script can also be imported and called as `convert(pt_path, out_path)`.
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def fold_bn_into_conv(conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, eps):
    """Fold BatchNorm parameters into the preceding Conv2d weight + bias.

    Args:
        conv_weight: [c_out, c_in, kH, kW] — original conv kernel.
        conv_bias:   [c_out] or None — original conv bias.
        bn_weight:   [c_out] — gamma.
        bn_bias:     [c_out] — beta.
        bn_mean:     [c_out] — running mean.
        bn_var:      [c_out] — running variance.
        eps:         scalar — BatchNorm eps.

    Returns:
        (folded_weight [c_out, c_in, kH, kW], folded_bias [c_out])
    """
    inv_std = bn_weight / torch.sqrt(bn_var + eps)
    # Broadcast inv_std along [c_in, kH, kW]:
    folded_weight = conv_weight * inv_std.view(-1, 1, 1, 1)
    if conv_bias is None:
        conv_bias = torch.zeros_like(bn_bias)
    folded_bias = (conv_bias - bn_mean) * inv_std + bn_bias
    return folded_weight, folded_bias


def convert(pt_path: str, out_path: str) -> None:
    """Convert one Ultralytics .pt checkpoint to a folded-BN safetensors file."""
    print(f"Loading {pt_path}")
    # weights_only=False is required for Ultralytics checkpoints since they pickle
    # model class instances. Trust the source — only run this on .pt files you control.
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)

    # Ultralytics wraps the model object under different keys depending on the
    # checkpoint format. Try the common ones in order.
    if isinstance(ckpt, dict) and "model" in ckpt:
        model = ckpt["model"]
    elif hasattr(ckpt, "state_dict"):
        model = ckpt
    else:
        raise RuntimeError(f"Unrecognized checkpoint format: top-level keys = {list(ckpt) if isinstance(ckpt, dict) else type(ckpt)}")

    # Ultralytics ships some models in fp16. Cast to fp32 for safe folding.
    model = model.float()
    state = model.state_dict()
    print(f"  state_dict has {len(state)} tensors")

    folded = {}
    bn_keys_consumed: set[str] = set()

    # Walk all keys. For each `*.conv.weight` followed by matching `*.bn.{weight,bias,running_mean,running_var}`,
    # fold them into a single `*.conv.weight` + `*.conv.bias`.
    for key, tensor in state.items():
        if not key.endswith(".conv.weight"):
            continue

        prefix = key[: -len(".conv.weight")]  # e.g. "model.0", "model.2.cv1", "model.22.cv2.0.0"
        bn_prefix = f"{prefix}.bn"
        bn_w = state.get(f"{bn_prefix}.weight")
        bn_b = state.get(f"{bn_prefix}.bias")
        bn_m = state.get(f"{bn_prefix}.running_mean")
        bn_v = state.get(f"{bn_prefix}.running_var")

        if bn_w is None or bn_b is None or bn_m is None or bn_v is None:
            # No BN here — this is a plain Conv (e.g. detect head's final 1×1 projection).
            # Copy as-is. Also pick up a matching conv bias if any.
            folded[f"{prefix}.conv.weight"] = tensor.contiguous().clone()
            conv_b = state.get(f"{prefix}.conv.bias")
            if conv_b is not None:
                folded[f"{prefix}.conv.bias"] = conv_b.contiguous().clone()
            continue

        conv_b_orig = state.get(f"{prefix}.conv.bias")
        eps = 1e-3  # Ultralytics' default BatchNorm eps. Confirm by introspecting the module if your fork differs.
        fw, fb = fold_bn_into_conv(tensor, conv_b_orig, bn_w, bn_b, bn_m, bn_v, eps)
        folded[f"{prefix}.conv.weight"] = fw.contiguous().clone()
        folded[f"{prefix}.conv.bias"] = fb.contiguous().clone()

        bn_keys_consumed.update([
            f"{bn_prefix}.weight",
            f"{bn_prefix}.bias",
            f"{bn_prefix}.running_mean",
            f"{bn_prefix}.running_var",
            f"{bn_prefix}.num_batches_tracked",
        ])

    # The detect head's three final 1×1 projection convs live at `model.22.cv2.{0,1,2}.2.weight`
    # (no `.conv.` suffix because they're plain `Conv2d` modules, not the wrapped `Conv` block).
    # Same for `model.22.cv3.{0,1,2}.2.weight`. Copy these directly.
    for key, tensor in state.items():
        if "model.22." in key and (".2.weight" in key or ".2.bias" in key):
            # Skip if already processed above.
            if key not in folded and not any(key.startswith(b.rsplit(".", 1)[0]) and key in bn_keys_consumed for b in folded):
                if key not in folded:
                    folded[key] = tensor.contiguous().clone()

    # Sanity check: every folded tensor should have a sensible shape.
    print(f"  folded into {len(folded)} tensors")
    detect_keys = sorted(k for k in folded if k.startswith("model.22"))
    print(f"  detect head tensors ({len(detect_keys)}):")
    for k in detect_keys[:8]:
        print(f"    {k}: {tuple(folded[k].shape)}")
    if len(detect_keys) > 8:
        print(f"    ... ({len(detect_keys) - 8} more)")

    print(f"Saving to {out_path}")
    save_file(folded, out_path, metadata={"format": "sharpinference-yolo-folded-v1"})
    out_size = Path(out_path).stat().st_size
    print(f"  wrote {out_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.pt> <output.safetensors>", file=sys.stderr)
        sys.exit(2)
    convert(sys.argv[1], sys.argv[2])
