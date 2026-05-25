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

    folded: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    # Pass 1: walk all keys. For each `*.conv.weight` followed by matching `*.bn.{weight,bias,running_mean,running_var}`,
    # fold them into a single `*.conv.weight` + `*.conv.bias`. Mark all participating keys as consumed.
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
            # No BN here — Conv-only with no normalization. Copy weights as-is plus any bias.
            folded[f"{prefix}.conv.weight"] = tensor.contiguous().clone()
            conv_b = state.get(f"{prefix}.conv.bias")
            if conv_b is not None:
                folded[f"{prefix}.conv.bias"] = conv_b.contiguous().clone()
                consumed.add(f"{prefix}.conv.bias")
            consumed.add(key)
            continue

        conv_b_orig = state.get(f"{prefix}.conv.bias")
        eps = 1e-3  # Ultralytics' default BatchNorm eps. Confirm by introspecting the module if your fork differs.
        fw, fb = fold_bn_into_conv(tensor, conv_b_orig, bn_w, bn_b, bn_m, bn_v, eps)
        folded[f"{prefix}.conv.weight"] = fw.contiguous().clone()
        folded[f"{prefix}.conv.bias"] = fb.contiguous().clone()

        consumed.add(key)
        if conv_b_orig is not None:
            consumed.add(f"{prefix}.conv.bias")
        consumed.update([
            f"{bn_prefix}.weight",
            f"{bn_prefix}.bias",
            f"{bn_prefix}.running_mean",
            f"{bn_prefix}.running_var",
            f"{bn_prefix}.num_batches_tracked",
        ])

    # Pass 2: pick up any remaining weight/bias tensors that don't belong to a BN-folded Conv.
    # These are the plain `nn.Conv2d` layers in the detect head — both YOLOv8's three final
    # projection convs (cv2.{s}.2, cv3.{s}.2) and YOLO11's depthwise-separable cv3 modules use
    # the same pattern. Filtering by suffix rather than hard-coding the detect-head layer index
    # lets the script work for both v8 and v11 (and future variants) without modification.
    for key, tensor in state.items():
        if key in consumed:
            continue
        if not (key.endswith(".weight") or key.endswith(".bias")):
            continue
        # Skip the BN scalar metadata and any other non-tensor parameter we don't recognize.
        if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            continue
        # If the parent key path looks like a BN parameter we should have already folded, skip —
        # something is wrong upstream and copying would double-count.
        if key.endswith(".bn.weight") or key.endswith(".bn.bias"):
            continue
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
