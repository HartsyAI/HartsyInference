"""
Dumps intermediate tensor statistics from a diffusers SD3 pipeline run.
Saves text encoder outputs, per-step latent stats, and final latent for
cross-runtime validation against SharpInference C# output.

Usage: python dump_sd3_reference_stats.py <model_dir> [output_dir]
  model_dir: path to SD3 Medium diffusers model (e.g., stabilityai/stable-diffusion-3-medium-diffusers)
Requires: pip install diffusers transformers torch safetensors sentencepiece accelerate
"""
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path


def tensor_stats(name: str, t: torch.Tensor) -> dict:
    """Compute stats for a tensor."""
    with torch.no_grad():
        flat = t.float().flatten()
        return {
            "name": name,
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "abs_mean": float(flat.abs().mean()),
            "first_8": [float(x) for x in flat[:8]],
        }


def save_tensor(t: torch.Tensor, path: str):
    """Save tensor as raw float32 binary."""
    t.float().cpu().numpy().tofile(path)


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "sd3_reference_tensors"

    if model_dir is None:
        print("Usage: python dump_sd3_reference_stats.py <model_dir> [output_dir]")
        print("  model_dir: path to SD3 diffusers model directory or HF model ID")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
    from transformers import CLIPTextModelWithProjection, T5EncoderModel, CLIPTokenizer, T5Tokenizer

    device = "cpu"
    dtype = torch.float32

    # --- Settings ---
    prompt = "A photograph of an astronaut riding a horse"
    negative_prompt = ""
    width, height = 1024, 1024
    num_steps = 28
    cfg_scale = 7.0
    seed = 42

    stats = []
    stats.append({
        "name": "settings",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_steps": num_steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
    })

    # --- Load pipeline ---
    print(f"Loading SD3 pipeline from {model_dir}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_dir,
        torch_dtype=dtype,
    ).to(device)

    # Print config for verification
    transformer_config = pipe.transformer.config
    print(f"  Transformer: depth={transformer_config.num_layers}, "
          f"hidden={transformer_config.joint_attention_dim if hasattr(transformer_config, 'joint_attention_dim') else 'N/A'}, "
          f"heads={transformer_config.num_attention_heads}")

    # --- Step 1: Tokenize ---
    print(f"\nTokenizing: '{prompt}'")

    # CLIP-L tokens
    clip_l_tokens = pipe.tokenizer(prompt, max_length=77, padding="max_length",
                                    truncation=True, return_tensors="pt")
    clip_l_neg_tokens = pipe.tokenizer(negative_prompt, max_length=77, padding="max_length",
                                        truncation=True, return_tensors="pt")
    clip_l_ids = clip_l_tokens["input_ids"][0].tolist()

    # CLIP-G tokens
    clip_g_tokens = pipe.tokenizer_2(prompt, max_length=77, padding="max_length",
                                      truncation=True, return_tensors="pt")
    clip_g_neg_tokens = pipe.tokenizer_2(negative_prompt, max_length=77, padding="max_length",
                                          truncation=True, return_tensors="pt")
    clip_g_ids = clip_g_tokens["input_ids"][0].tolist()

    # T5 tokens
    t5_tokens = pipe.tokenizer_3(prompt, max_length=77, padding="max_length",
                                  truncation=True, return_tensors="pt")
    t5_neg_tokens = pipe.tokenizer_3(negative_prompt, max_length=77, padding="max_length",
                                      truncation=True, return_tensors="pt")
    t5_ids = t5_tokens["input_ids"][0].tolist()

    print(f"  CLIP-L IDs (first 15): {clip_l_ids[:15]}")
    print(f"  CLIP-G IDs (first 15): {clip_g_ids[:15]}")
    print(f"  T5 IDs (first 15): {t5_ids[:15]}")

    stats.append({"name": "clip_l_token_ids", "values": clip_l_ids})
    stats.append({"name": "clip_g_token_ids", "values": clip_g_ids})
    stats.append({"name": "t5_token_ids", "values": t5_ids})

    # Save token IDs
    np.array(clip_l_ids, dtype=np.int32).tofile(os.path.join(output_dir, "clip_l_token_ids.bin"))
    np.array(clip_g_ids, dtype=np.int32).tofile(os.path.join(output_dir, "clip_g_token_ids.bin"))
    np.array(t5_ids, dtype=np.int32).tofile(os.path.join(output_dir, "t5_token_ids.bin"))

    # --- Step 2: Text encoding ---
    print("\nEncoding text...")

    with torch.no_grad():
        # CLIP-L: penultimate hidden state + pooled
        clip_l_out = pipe.text_encoder(
            clip_l_tokens["input_ids"].to(device),
            output_hidden_states=True,
        )
        clip_l_hidden = clip_l_out.hidden_states[-2]  # penultimate [1, 77, 768]
        clip_l_pooled = clip_l_out.text_embeds  # [1, 768]

        # CLIP-G: penultimate hidden state + pooled
        clip_g_out = pipe.text_encoder_2(
            clip_g_tokens["input_ids"].to(device),
            output_hidden_states=True,
        )
        clip_g_hidden = clip_g_out.hidden_states[-2]  # penultimate [1, 77, 1280]
        clip_g_pooled = clip_g_out.text_embeds  # [1, 1280]

        # T5-XXL
        t5_out = pipe.text_encoder_3(
            t5_tokens["input_ids"].to(device),
        )
        t5_hidden = t5_out.last_hidden_state  # [1, 77, 4096]

    print(f"  CLIP-L hidden: {clip_l_hidden.shape}, pooled: {clip_l_pooled.shape}")
    print(f"  CLIP-G hidden: {clip_g_hidden.shape}, pooled: {clip_g_pooled.shape}")
    print(f"  T5 hidden: {t5_hidden.shape}")

    save_tensor(clip_l_hidden, os.path.join(output_dir, "clip_l_hidden.bin"))
    save_tensor(clip_l_pooled, os.path.join(output_dir, "clip_l_pooled.bin"))
    save_tensor(clip_g_hidden, os.path.join(output_dir, "clip_g_hidden.bin"))
    save_tensor(clip_g_pooled, os.path.join(output_dir, "clip_g_pooled.bin"))
    save_tensor(t5_hidden, os.path.join(output_dir, "t5_hidden.bin"))

    stats.append(tensor_stats("clip_l_hidden", clip_l_hidden))
    stats.append(tensor_stats("clip_l_pooled", clip_l_pooled))
    stats.append(tensor_stats("clip_g_hidden", clip_g_hidden))
    stats.append(tensor_stats("clip_g_pooled", clip_g_pooled))
    stats.append(tensor_stats("t5_hidden", t5_hidden))

    # --- Step 3: Combine text conditioning ---
    print("\nCombining text conditioning...")

    # Concatenate CLIP-L + CLIP-G hidden along feature dim
    lg_hidden = torch.cat([clip_l_hidden, clip_g_hidden], dim=-1)  # [1, 77, 2048]
    print(f"  lg_hidden (CLIP-L+G concat): {lg_hidden.shape}")

    # Zero-pad to 4096
    lg_padded = torch.nn.functional.pad(lg_hidden, (0, 4096 - 2048))  # [1, 77, 4096]
    print(f"  lg_padded: {lg_padded.shape}")

    # Concatenate with T5 along sequence dim
    combined_context = torch.cat([lg_padded, t5_hidden], dim=1)  # [1, 154, 4096]
    print(f"  combined_context (before projection): {combined_context.shape}")

    save_tensor(lg_hidden, os.path.join(output_dir, "lg_hidden_concat.bin"))
    save_tensor(lg_padded, os.path.join(output_dir, "lg_padded.bin"))
    save_tensor(combined_context, os.path.join(output_dir, "combined_context_pre_proj.bin"))

    stats.append(tensor_stats("lg_hidden_concat", lg_hidden))
    stats.append(tensor_stats("combined_context_pre_proj", combined_context))

    # Pooled projection
    pooled = torch.cat([clip_l_pooled, clip_g_pooled], dim=-1)  # [1, 2048]
    print(f"  pooled (CLIP-L+G): {pooled.shape}")

    save_tensor(pooled, os.path.join(output_dir, "pooled_projection.bin"))
    stats.append(tensor_stats("pooled_projection", pooled))

    # --- Step 4: Context projection through context_embedder ---
    print("\nProjecting context through context_embedder...")
    with torch.no_grad():
        projected_context = pipe.transformer.context_embedder(combined_context)  # [1, 154, hidden_size]

    print(f"  projected_context: {projected_context.shape}")
    save_tensor(projected_context, os.path.join(output_dir, "context_projected.bin"))
    stats.append(tensor_stats("context_projected", projected_context))

    # --- Step 5: Scheduler setup ---
    print("\nSetting up flow-match scheduler...")
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps)

    timesteps = scheduler.timesteps.tolist()
    sigmas = scheduler.sigmas.tolist()
    print(f"  Timesteps (first 5): {timesteps[:5]}")
    print(f"  Sigmas (first 5): {sigmas[:5]}")
    print(f"  Sigma range: [{sigmas[0]:.6f}, {sigmas[-1]:.6f}]")

    stats.append({"name": "timesteps", "values": timesteps})
    stats.append({"name": "sigmas", "values": sigmas})

    # --- Step 6: Initial noise ---
    print("\nGenerating initial noise...")
    generator = torch.Generator(device=device).manual_seed(seed)
    latent_h = height // 8  # 128
    latent_w = width // 8   # 128
    latent_shape = (1, 16, latent_h, latent_w)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)

    # Scale by initial sigma
    latents = latents * scheduler.init_noise_sigma

    print(f"  Latent shape: {latents.shape}")
    print(f"  Init noise sigma: {float(scheduler.init_noise_sigma):.6f}")

    save_tensor(latents, os.path.join(output_dir, "initial_latents.bin"))
    stats.append(tensor_stats("initial_latents", latents))
    stats.append({"name": "init_noise_sigma", "value": float(scheduler.init_noise_sigma)})

    # --- Step 7: Denoising loop ---
    print("\nRunning denoising loop...")

    # Negative text encoding for CFG
    with torch.no_grad():
        clip_l_neg_out = pipe.text_encoder(
            clip_l_neg_tokens["input_ids"].to(device),
            output_hidden_states=True,
        )
        clip_l_neg_hidden = clip_l_neg_out.hidden_states[-2]
        clip_l_neg_pooled = clip_l_neg_out.text_embeds

        clip_g_neg_out = pipe.text_encoder_2(
            clip_g_neg_tokens["input_ids"].to(device),
            output_hidden_states=True,
        )
        clip_g_neg_hidden = clip_g_neg_out.hidden_states[-2]
        clip_g_neg_pooled = clip_g_neg_out.text_embeds

        t5_neg_out = pipe.text_encoder_3(
            t5_neg_tokens["input_ids"].to(device),
        )
        t5_neg_hidden = t5_neg_out.last_hidden_state

    # Build negative conditioning
    lg_neg = torch.cat([clip_l_neg_hidden, clip_g_neg_hidden], dim=-1)
    lg_neg_padded = torch.nn.functional.pad(lg_neg, (0, 4096 - 2048))
    neg_context = torch.cat([lg_neg_padded, t5_neg_hidden], dim=1)
    neg_pooled = torch.cat([clip_l_neg_pooled, clip_g_neg_pooled], dim=-1)

    with torch.no_grad():
        neg_projected = pipe.transformer.context_embedder(neg_context)

    save_tensor(neg_projected, os.path.join(output_dir, "neg_context_projected.bin"))
    save_tensor(neg_pooled, os.path.join(output_dir, "neg_pooled_projection.bin"))

    # CFG: run two forward passes per step
    scheduler_timesteps = scheduler.timesteps

    for i, t in enumerate(scheduler_timesteps):
        sigma = scheduler.sigmas[i]

        # Scale model input
        latent_model_input = scheduler.scale_model_input(latents, t)

        with torch.no_grad():
            # Unconditional pass
            noise_pred_uncond = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=t.unsqueeze(0),
                encoder_hidden_states=neg_projected,
                pooled_projections=neg_pooled,
            ).sample

            # Conditional pass
            noise_pred_cond = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=t.unsqueeze(0),
                encoder_hidden_states=projected_context,
                pooled_projections=pooled,
            ).sample

        # CFG combination
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

        # Scheduler step
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        step_stats = tensor_stats(f"step_{i}_latents", latents)
        stats.append(step_stats)

        if i < 3 or i == len(scheduler_timesteps) - 1:
            print(f"  Step {i}/{num_steps}: t={float(t):.1f}, sigma={float(sigma):.6f}, "
                  f"latent mean={step_stats['mean']:.6f}, std={step_stats['std']:.6f}")

        # Save detailed tensors for first step
        if i == 0:
            save_tensor(latent_model_input, os.path.join(output_dir, "step0_model_input.bin"))
            save_tensor(noise_pred_uncond, os.path.join(output_dir, "step0_noise_pred_uncond.bin"))
            save_tensor(noise_pred_cond, os.path.join(output_dir, "step0_noise_pred_cond.bin"))
            save_tensor(noise_pred, os.path.join(output_dir, "step0_noise_pred_cfg.bin"))
            save_tensor(latents, os.path.join(output_dir, "step0_latents_after.bin"))
            stats.append(tensor_stats("step0_noise_pred_uncond", noise_pred_uncond))
            stats.append(tensor_stats("step0_noise_pred_cond", noise_pred_cond))
            stats.append(tensor_stats("step0_noise_pred_cfg", noise_pred))

    # Save final latents
    save_tensor(latents, os.path.join(output_dir, "final_latents.bin"))
    stats.append(tensor_stats("final_latents", latents))

    # --- Step 8: VAE decode ---
    print("\nDecoding with VAE...")
    with torch.no_grad():
        # SD3 VAE scaling: latent = (latent / scale_factor) + shift_factor
        # scale_factor=1.5305, shift_factor=0.0609
        latents_scaled = (latents / 1.5305) + 0.0609
        image = pipe.vae.decode(latents_scaled).sample

    save_tensor(latents_scaled, os.path.join(output_dir, "latents_for_vae.bin"))
    save_tensor(image, os.path.join(output_dir, "vae_output.bin"))
    stats.append(tensor_stats("latents_for_vae", latents_scaled))
    stats.append(tensor_stats("vae_output", image))

    # Clamp to [0, 1] for image
    image_clamped = ((image + 1.0) / 2.0).clamp(0, 1)
    stats.append(tensor_stats("image_clamped_01", image_clamped))

    # --- Save stats ---
    stats_path = os.path.join(output_dir, "sd3_reference_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nAll stats saved to {stats_path}")
    print(f"All tensors saved to {output_dir}")
    print(f"Total stats entries: {len(stats)}")


if __name__ == "__main__":
    main()
