"""
Dumps Flux RoPE (Rotary Position Embedding) reference data: cos/sin tables and
rotated Q/K vectors for known positions.

Validates the axial RoPE implementation with axes_dim=[16,56,56], theta=10000.
Saves raw float32 binary for cross-runtime comparison against SharpInference C#.

Usage: python dump_flux_rope_reference.py [output_dir]
Requires: pip install torch numpy
"""
import sys
import os
import json
import torch
import numpy as np
from typing import List


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """
    Compute RoPE rotation matrices for a single axis.
    BFL reference: src/flux/math.py

    Args:
        pos: position values [seq_len] or [B, seq_len]
        dim: dimension for this axis (must be even)
        theta: base frequency (10000)

    Returns:
        Rotation matrices [seq_len, dim//2, 2, 2] (or with batch)
    """
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)  # [dim//2]
    out = torch.einsum("...n,d->...nd", pos.float(), omega.float())  # [..., seq_len, dim//2]
    out = torch.stack([torch.cos(out), -torch.sin(out),
                       torch.sin(out), torch.cos(out)], dim=-1)
    out = out.view(*out.shape[:-1], 2, 2)  # [..., seq_len, dim//2, 2, 2]
    return out.float()


def embed_nd(pos_ids: torch.Tensor, axes_dim: List[int], theta: int) -> torch.Tensor:
    """
    Compute multi-axis RoPE embeddings.
    BFL reference: EmbedND class in src/flux/modules/layers.py

    Args:
        pos_ids: [seq_len, num_axes] position IDs
        axes_dim: dimension per axis [16, 56, 56]
        theta: base frequency (10000)

    Returns:
        RoPE embeddings [1, seq_len, total_dim//2, 2, 2]
    """
    embs = []
    for i, dim in enumerate(axes_dim):
        emb = rope(pos_ids[:, i], dim, theta)  # [seq_len, dim//2, 2, 2]
        embs.append(emb)
    # Concatenate along the frequency dimension
    result = torch.cat(embs, dim=-3)  # [seq_len, 64, 2, 2]
    return result.unsqueeze(0)  # [1, seq_len, 64, 2, 2]


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Apply RoPE rotation to Q and K.
    BFL reference: src/flux/math.py apply_rope function.

    Args:
        xq: query [B, H, L, D] where D=head_dim=128
        xk: key [B, H, L, D]
        freqs_cis: [1, L, D//2, 2, 2] rotation matrices

    Returns:
        rotated (xq, xk) same shapes
    """
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)  # [B, H, L, D//2, 1, 2]
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)  # [B, H, L, D//2, 1, 2]
    # freqs_cis: [1, L, D//2, 2, 2] -> broadcast over B, H
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def save_tensor(t: torch.Tensor, path: str):
    t.float().cpu().numpy().tofile(path)


def tensor_stats(name: str, t: torch.Tensor) -> dict:
    with torch.no_grad():
        flat = t.float().flatten()
        return {
            "name": name,
            "shape": list(t.shape),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "first_8": [float(x) for x in flat[:8]],
        }


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "flux_reference_tensors/rope"
    os.makedirs(output_dir, exist_ok=True)

    axes_dim = [16, 56, 56]
    theta = 10000
    head_dim = 128
    num_heads = 24

    stats = []
    stats.append({
        "name": "config",
        "axes_dim": axes_dim,
        "theta": theta,
        "head_dim": head_dim,
        "num_heads": num_heads,
    })

    # --- Test 1: Small grid (4x4 image, 8 text tokens) ---
    print("=== Test 1: Small grid (4x4 packed, 8 text tokens) ===")
    h_packed, w_packed = 4, 4
    num_text = 8
    num_image = h_packed * w_packed  # 16
    total_seq = num_text + num_image  # 24

    # Build position IDs
    txt_ids = torch.zeros(num_text, 3)  # text: all zeros
    img_ids = torch.zeros(num_image, 3)
    for row in range(h_packed):
        for col in range(w_packed):
            idx = row * w_packed + col
            img_ids[idx, 1] = row   # y-coordinate
            img_ids[idx, 2] = col   # x-coordinate

    pos_ids = torch.cat([txt_ids, img_ids], dim=0)  # [24, 3]

    # Compute RoPE embeddings
    pe = embed_nd(pos_ids, axes_dim, theta)  # [1, 24, 64, 2, 2]

    print(f"  pos_ids shape: {pos_ids.shape}")
    print(f"  pe shape: {pe.shape}")
    print(f"  pe[0, 0, 0]: {pe[0, 0, 0]}")  # Text token 0, first freq pair
    print(f"  pe[0, 8, 0]: {pe[0, 8, 0]}")  # First image token, first freq pair

    save_tensor(pos_ids, os.path.join(output_dir, "small_pos_ids.bin"))
    save_tensor(pe, os.path.join(output_dir, "small_pe.bin"))
    stats.append(tensor_stats("small_pos_ids", pos_ids))
    stats.append(tensor_stats("small_pe", pe))

    # Also save the individual cos/sin components
    cos_vals = pe[0, :, :, 0, 0]  # [seq_len, 64] — cos values
    sin_vals = pe[0, :, :, 1, 0]  # [seq_len, 64] — sin values
    save_tensor(cos_vals, os.path.join(output_dir, "small_cos.bin"))
    save_tensor(sin_vals, os.path.join(output_dir, "small_sin.bin"))
    stats.append(tensor_stats("small_cos", cos_vals))
    stats.append(tensor_stats("small_sin", sin_vals))

    # --- Test 2: Apply RoPE to random Q/K ---
    print("\n=== Test 2: Apply RoPE to random Q/K ===")
    torch.manual_seed(42)
    B = 1
    q = torch.randn(B, num_heads, total_seq, head_dim)
    k = torch.randn(B, num_heads, total_seq, head_dim)

    save_tensor(q, os.path.join(output_dir, "small_q_before.bin"))
    save_tensor(k, os.path.join(output_dir, "small_k_before.bin"))

    q_rotated, k_rotated = apply_rope(q, k, pe)

    save_tensor(q_rotated, os.path.join(output_dir, "small_q_after.bin"))
    save_tensor(k_rotated, os.path.join(output_dir, "small_k_after.bin"))

    stats.append(tensor_stats("small_q_before", q))
    stats.append(tensor_stats("small_k_before", k))
    stats.append(tensor_stats("small_q_after", q_rotated))
    stats.append(tensor_stats("small_k_after", k_rotated))

    # Verify RoPE is a rotation (norm should be preserved)
    q_norm_before = q.float().norm(dim=-1).mean()
    q_norm_after = q_rotated.float().norm(dim=-1).mean()
    print(f"  Q norm before: {q_norm_before:.6f}, after: {q_norm_after:.6f}, "
          f"diff: {abs(q_norm_before - q_norm_after):.8f}")

    # --- Test 3: 1024x1024 resolution (64x64 packed) ---
    print("\n=== Test 3: 1024x1024 resolution ===")
    h_packed_large, w_packed_large = 64, 64
    num_text_large = 512
    num_image_large = h_packed_large * w_packed_large  # 4096
    total_seq_large = num_text_large + num_image_large  # 4608

    txt_ids_large = torch.zeros(num_text_large, 3)
    img_ids_large = torch.zeros(num_image_large, 3)
    for row in range(h_packed_large):
        for col in range(w_packed_large):
            idx = row * w_packed_large + col
            img_ids_large[idx, 1] = row
            img_ids_large[idx, 2] = col

    pos_ids_large = torch.cat([txt_ids_large, img_ids_large], dim=0)
    pe_large = embed_nd(pos_ids_large, axes_dim, theta)

    print(f"  pos_ids shape: {pos_ids_large.shape}")
    print(f"  pe shape: {pe_large.shape}")

    # Save just the first and last few tokens for size
    save_tensor(pos_ids_large[:20], os.path.join(output_dir, "large_pos_ids_first20.bin"))
    save_tensor(pe_large[:, :20], os.path.join(output_dir, "large_pe_first20.bin"))
    save_tensor(pe_large[:, -20:], os.path.join(output_dir, "large_pe_last20.bin"))

    # Save a specific image position for validation: row=32, col=48
    target_idx = num_text_large + 32 * w_packed_large + 48
    save_tensor(pe_large[:, target_idx:target_idx+1], os.path.join(output_dir, "large_pe_row32_col48.bin"))
    stats.append({
        "name": "large_pe_row32_col48",
        "target_idx": target_idx,
        "row": 32,
        "col": 48,
        "first_8": [float(x) for x in pe_large[0, target_idx].flatten()[:8]],
    })

    stats.append(tensor_stats("large_pe", pe_large))

    # --- Test 4: Verify per-axis frequency patterns ---
    print("\n=== Test 4: Per-axis frequency verification ===")
    # Single position for axis 0 (dim=16): pos=0 (always zero for standard flux)
    pos_axis0 = torch.tensor([0.0])
    rope_axis0 = rope(pos_axis0, 16, theta)  # [1, 8, 2, 2]
    print(f"  Axis 0 (pos=0): all cos=1, sin=0? {torch.allclose(rope_axis0[0, :, 0, 0], torch.ones(8))}")

    # Single position for axis 1 (dim=56): pos=5
    pos_axis1 = torch.tensor([5.0])
    rope_axis1 = rope(pos_axis1, 56, theta)  # [1, 28, 2, 2]
    save_tensor(rope_axis1, os.path.join(output_dir, "axis1_pos5.bin"))
    stats.append(tensor_stats("axis1_pos5", rope_axis1))

    # Single position for axis 2 (dim=56): pos=10
    pos_axis2 = torch.tensor([10.0])
    rope_axis2 = rope(pos_axis2, 56, theta)  # [1, 28, 2, 2]
    save_tensor(rope_axis2, os.path.join(output_dir, "axis2_pos10.bin"))
    stats.append(tensor_stats("axis2_pos10", rope_axis2))

    # Frequency values (omega) for each axis
    for i, dim in enumerate(axes_dim):
        scale = torch.arange(0, dim, 2, dtype=torch.float64) / dim
        omega = 1.0 / (theta ** scale)
        stats.append({
            "name": f"omega_axis{i}_dim{dim}",
            "values": [float(o) for o in omega],
            "num_pairs": len(omega),
        })
        print(f"  Axis {i} (dim={dim}): {len(omega)} freq pairs, "
              f"omega range [{float(omega[0]):.6f}, {float(omega[-1]):.10f}]")

    # --- Save all stats ---
    with open(os.path.join(output_dir, "flux_rope_reference.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nAll RoPE reference data saved to {output_dir}")


if __name__ == "__main__":
    main()
