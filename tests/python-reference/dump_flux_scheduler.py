"""
Dumps Flux flow-match Euler scheduler reference data: sigma schedules with
dynamic exponential shift for various resolutions and step counts.

Validates the exp(mu) shift formula and saves schedules as raw float32 binary
for cross-runtime comparison against HartsyInference C#.

Usage: python dump_flux_scheduler.py [output_dir]
Requires: pip install torch numpy diffusers
"""
import sys
import os
import json
import math
import numpy as np


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                    base_shift=0.5, max_shift=1.15):
    """Compute the mu parameter for dynamic shift."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def apply_exponential_shift(sigmas, mu):
    """Apply exponential time shift: sigma = exp(mu) * t / (1 + (exp(mu) - 1) * t)."""
    shift = math.exp(mu)
    shifted = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    return shifted


def apply_linear_shift(sigmas, mu):
    """Apply linear time shift: sigma = mu * t / (1 + (mu - 1) * t)."""
    shifted = mu * sigmas / (1.0 + (mu - 1.0) * sigmas)
    return shifted


def compute_schedule(num_steps, image_seq_len, shift_type="exponential"):
    """Compute the full sigma schedule for a given resolution and step count."""
    mu = calculate_shift(image_seq_len)

    # Base sigmas: linearly spaced from 1.0 to 0.0 (N+1 points)
    sigmas = np.linspace(1.0, 0.0, num_steps + 1).astype(np.float64)

    if shift_type == "exponential":
        sigmas = apply_exponential_shift(sigmas, mu)
    elif shift_type == "linear":
        sigmas = apply_linear_shift(sigmas, mu)

    timesteps = sigmas[:-1] * 1000.0  # Scale to 0-1000 for model conditioning
    return mu, sigmas, timesteps


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "flux_reference_tensors/scheduler"
    os.makedirs(output_dir, exist_ok=True)

    stats = []

    # Test configurations: (num_steps, resolution_name, H_latent, W_latent)
    configs = [
        (4, "512x512", 64, 64),        # Schnell typical
        (4, "1024x1024", 128, 128),     # Schnell high-res
        (20, "512x512", 64, 64),        # Dev typical
        (20, "1024x1024", 128, 128),    # Dev high-res
        (20, "768x1024", 96, 128),      # Dev non-square
        (28, "1024x1024", 128, 128),    # Dev many steps
        (50, "1024x1024", 128, 128),    # Dev max steps
    ]

    for num_steps, res_name, h_lat, w_lat in configs:
        # Image sequence length = (H_lat / patch_size) * (W_lat / patch_size)
        image_seq_len = (h_lat // 2) * (w_lat // 2)

        mu, sigmas, timesteps = compute_schedule(num_steps, image_seq_len, "exponential")
        shift = math.exp(mu)

        entry = {
            "name": f"schedule_{res_name}_{num_steps}steps",
            "resolution": res_name,
            "num_steps": num_steps,
            "h_latent": h_lat,
            "w_latent": w_lat,
            "image_seq_len": image_seq_len,
            "mu": float(mu),
            "shift_exp_mu": float(shift),
            "sigmas": [float(s) for s in sigmas],
            "timesteps": [float(t) for t in timesteps],
            "sigma_first": float(sigmas[0]),
            "sigma_last": float(sigmas[-1]),
        }
        stats.append(entry)

        # Save binary files for selected configs
        tag = f"{res_name}_{num_steps}steps"
        np.array(sigmas, dtype=np.float32).tofile(os.path.join(output_dir, f"sigmas_{tag}.bin"))
        np.array(timesteps, dtype=np.float32).tofile(os.path.join(output_dir, f"timesteps_{tag}.bin"))

        print(f"  {tag}: mu={mu:.6f}, exp(mu)={shift:.6f}, "
              f"sigma[0]={sigmas[0]:.6f}, sigma[-1]={sigmas[-1]:.6f}, "
              f"seq_len={image_seq_len}")

    # Also dump the linear shift for comparison (used by SD3, not Flux)
    print("\n--- Linear shift comparison (SD3-style) ---")
    for num_steps, res_name, h_lat, w_lat in [(20, "1024x1024", 128, 128)]:
        image_seq_len = (h_lat // 2) * (w_lat // 2)
        mu, sigmas_linear, _ = compute_schedule(num_steps, image_seq_len, "linear")
        _, sigmas_exp, _ = compute_schedule(num_steps, image_seq_len, "exponential")

        stats.append({
            "name": f"linear_vs_exp_{res_name}_{num_steps}steps",
            "mu": float(mu),
            "sigmas_linear": [float(s) for s in sigmas_linear],
            "sigmas_exponential": [float(s) for s in sigmas_exp],
            "max_diff": float(np.max(np.abs(sigmas_linear - sigmas_exp))),
        })
        print(f"  Linear vs Exp max diff: {np.max(np.abs(sigmas_linear - sigmas_exp)):.6f}")

    # Also validate specific mu values for known resolutions
    print("\n--- mu validation ---")
    mu_checks = [
        (256, "base_seq_len"),
        (1024, "512x512"),
        (3072, "768x1024"),
        (4096, "1024x1024_max"),
    ]
    for seq_len, label in mu_checks:
        mu = calculate_shift(seq_len)
        print(f"  seq_len={seq_len} ({label}): mu={mu:.6f}, exp(mu)={math.exp(mu):.6f}")
        stats.append({
            "name": f"mu_check_{label}",
            "seq_len": seq_len,
            "mu": float(mu),
            "exp_mu": float(math.exp(mu)),
        })

    # Save stats
    with open(os.path.join(output_dir, "flux_scheduler_reference.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nAll scheduler reference data saved to {output_dir}")


if __name__ == "__main__":
    main()
