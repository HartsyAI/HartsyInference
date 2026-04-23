"""
Dumps intermediate tensor statistics from a diffusers SDXL pipeline run.
Uses the same JuggernautXL checkpoint and parameters as C# SdxlGenerationTests.

Matching C# test params:
  - Prompt: "a majestic lion in a field of sunflowers, golden hour, photorealistic"
  - NegPrompt: "blurry, low quality, deformed"
  - Size: 256x256
  - Steps: 10
  - CFG: 7.0
  - Seed: 42
  - Scheduler: EulerDiscrete, leading spacing, beta_start=0.00085, beta_end=0.012
  - VAE scaling: 0.13025

Usage: python dump_sdxl_reference_stats.py [checkpoint_path] [output_dir]
Requires: pip install diffusers transformers torch safetensors accelerate
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

def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Users\kaleb\Desktop\Projects\SwarmUI\Models\Stable-Diffusion\SDXL\Juggernaut_XL_-_Ragnarok_by_RunDiffusion.safetensors"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(os.path.dirname(__file__), "sdxl_reference_tensors")

    os.makedirs(output_dir, exist_ok=True)

    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    from transformers import CLIPTokenizer

    device = "cpu"
    dtype = torch.float32

    # =========================================================================
    # Load pipeline from single-file checkpoint
    # =========================================================================
    print(f"Loading SDXL pipeline from: {checkpoint_path}")
    pipe = StableDiffusionXLPipeline.from_single_file(
        checkpoint_path,
        torch_dtype=dtype,
    ).to(device)

    # Replace scheduler to match C# defaults exactly
    pipe.scheduler = EulerDiscreteScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        timestep_spacing="leading",
        prediction_type="epsilon",
    )

    print(f"  text_encoder: {type(pipe.text_encoder).__name__}")
    print(f"  text_encoder_2: {type(pipe.text_encoder_2).__name__}")
    print(f"  unet: {type(pipe.unet).__name__}")
    print(f"  vae: {type(pipe.vae).__name__}")
    print(f"  scheduler: {type(pipe.scheduler).__name__}")

    # Verify attention heads in UNet
    for name, module in pipe.unet.named_modules():
        if hasattr(module, 'heads') and 'attn' in name:
            print(f"  {name}: heads={module.heads}")
            break

    # =========================================================================
    # Settings matching C# test
    # =========================================================================
    prompt = "a majestic lion in a field of sunflowers, golden hour, photorealistic"
    neg_prompt = "blurry, low quality, deformed"
    width, height = 256, 256
    num_steps = 10
    cfg_scale = 7.0
    seed = 42

    stats = []

    # =========================================================================
    # 1. Tokenize with both tokenizers (CLIP-L and CLIP-G use same vocab)
    # =========================================================================
    print("\n=== Tokenizing ===")
    tokenizer = pipe.tokenizer  # CLIP-L tokenizer
    tokenizer_2 = pipe.tokenizer_2  # CLIP-G tokenizer

    # Tokenize for CLIP-L
    prompt_tokens_L = tokenizer(prompt, padding="max_length", max_length=77,
                                 truncation=True, return_tensors="pt")
    neg_tokens_L = tokenizer(neg_prompt, padding="max_length", max_length=77,
                              truncation=True, return_tensors="pt")

    # Tokenize for CLIP-G
    prompt_tokens_G = tokenizer_2(prompt, padding="max_length", max_length=77,
                                   truncation=True, return_tensors="pt")
    neg_tokens_G = tokenizer_2(neg_prompt, padding="max_length", max_length=77,
                                truncation=True, return_tensors="pt")

    prompt_ids_L = prompt_tokens_L.input_ids[0].tolist()
    neg_ids_L = neg_tokens_L.input_ids[0].tolist()
    prompt_ids_G = prompt_tokens_G.input_ids[0].tolist()
    neg_ids_G = neg_tokens_G.input_ids[0].tolist()

    print(f"  CLIP-L prompt IDs: {prompt_ids_L[:15]}...")
    print(f"  CLIP-L neg IDs:    {neg_ids_L[:10]}...")
    print(f"  CLIP-G prompt IDs: {prompt_ids_G[:15]}...")
    print(f"  CLIP-G neg IDs:    {neg_ids_G[:10]}...")

    # Find EOS positions for CLIP-G (needed for pooled output)
    eos_token_id = tokenizer_2.eos_token_id
    prompt_eos_G = prompt_ids_G.index(eos_token_id) if eos_token_id in prompt_ids_G else 76
    neg_eos_G = neg_ids_G.index(eos_token_id) if eos_token_id in neg_ids_G else 76
    print(f"  CLIP-G EOS positions: prompt={prompt_eos_G}, neg={neg_eos_G}")

    stats.append({"name": "clip_l_prompt_token_ids", "values": prompt_ids_L})
    stats.append({"name": "clip_l_neg_token_ids", "values": neg_ids_L})
    stats.append({"name": "clip_g_prompt_token_ids", "values": prompt_ids_G})
    stats.append({"name": "clip_g_neg_token_ids", "values": neg_ids_G})
    stats.append({"name": "clip_g_eos_positions", "prompt": prompt_eos_G, "neg": neg_eos_G})

    # =========================================================================
    # 2. Encode text with both CLIP encoders
    # =========================================================================
    print("\n=== Text Encoding ===")

    text_encoder = pipe.text_encoder  # CLIP-L
    text_encoder_2 = pipe.text_encoder_2  # CLIP-G

    with torch.no_grad():
        # CLIP-L: batch [neg, pos] - penultimate layer hidden states
        clip_l_input = torch.cat([neg_tokens_L.input_ids, prompt_tokens_L.input_ids], dim=0)
        clip_l_out = text_encoder(clip_l_input, output_hidden_states=True)
        clip_l_penultimate = clip_l_out.hidden_states[-2]  # [2, 77, 768]
        clip_l_last = clip_l_out.last_hidden_state  # [2, 77, 768]

        print(f"  CLIP-L penultimate: shape={list(clip_l_penultimate.shape)}")
        stats.append(tensor_stats("clip_l_penultimate", clip_l_penultimate))
        stats.append(tensor_stats("clip_l_last_hidden", clip_l_last))
        clip_l_penultimate.numpy().tofile(os.path.join(output_dir, "clip_l_penultimate.bin"))

        # CLIP-G: batch [neg, pos] - penultimate layer hidden states + pooled
        clip_g_input = torch.cat([neg_tokens_G.input_ids, prompt_tokens_G.input_ids], dim=0)
        clip_g_out = text_encoder_2(clip_g_input, output_hidden_states=True)
        clip_g_penultimate = clip_g_out.hidden_states[-2]  # [2, 77, 1280]
        clip_g_last = clip_g_out.last_hidden_state  # [2, 77, 1280]

        # Pooled output: text_projection applied to EOS token hidden state
        # diffusers extracts this from the final layer output at EOS position
        clip_g_pooled = clip_g_out.text_embeds  # [2, 1280] - pooled from text_encoder_2

        print(f"  CLIP-G penultimate: shape={list(clip_g_penultimate.shape)}")
        print(f"  CLIP-G pooled: shape={list(clip_g_pooled.shape)}")
        stats.append(tensor_stats("clip_g_penultimate", clip_g_penultimate))
        stats.append(tensor_stats("clip_g_last_hidden", clip_g_last))
        stats.append(tensor_stats("clip_g_pooled", clip_g_pooled))
        clip_g_penultimate.numpy().tofile(os.path.join(output_dir, "clip_g_penultimate.bin"))
        clip_g_pooled.numpy().tofile(os.path.join(output_dir, "clip_g_pooled.bin"))

    # Concatenate CLIP-L + CLIP-G penultimate along last dim: [2, 77, 768+1280] = [2, 77, 2048]
    text_embeddings = torch.cat([clip_l_penultimate, clip_g_penultimate], dim=-1)
    print(f"  Concatenated text embeddings: shape={list(text_embeddings.shape)}")
    stats.append(tensor_stats("text_embeddings_concat", text_embeddings))
    text_embeddings.numpy().tofile(os.path.join(output_dir, "text_embeddings.bin"))

    # =========================================================================
    # 3. ADM conditioning (size/crop scalars)
    # =========================================================================
    # C# uses: [origH, origW, cropTop, cropLeft, targetH, targetW]
    size_condition = [float(height), float(width), 0.0, 0.0, float(height), float(width)]
    print(f"\n=== ADM Conditioning ===")
    print(f"  size_condition: {size_condition}")
    stats.append({"name": "size_condition", "values": size_condition})

    # Build the added conditioning kwargs as diffusers expects
    # For SDXL, these go through add_time_ids
    add_time_ids = torch.tensor([size_condition], dtype=dtype)  # [1, 6]
    # For CFG, we need [neg_time_ids, pos_time_ids]
    add_time_ids_cfg = torch.cat([add_time_ids, add_time_ids], dim=0)  # [2, 6]
    stats.append(tensor_stats("add_time_ids", add_time_ids_cfg))

    # =========================================================================
    # 4. Initial noise
    # =========================================================================
    print(f"\n=== Initial Noise ===")
    generator = torch.Generator(device=device).manual_seed(seed)
    latent_shape = (1, 4, height // 8, width // 8)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
    stats.append(tensor_stats("initial_noise", latents))
    latents.numpy().tofile(os.path.join(output_dir, "initial_noise.bin"))
    print(f"  noise shape: {list(latents.shape)}, mean={float(latents.mean()):.6f}, std={float(latents.std()):.6f}")

    # =========================================================================
    # 5. Scheduler setup
    # =========================================================================
    print(f"\n=== Scheduler ===")
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps
    sigmas = scheduler.sigmas

    print(f"  timesteps: {[float(t) for t in timesteps]}")
    print(f"  sigmas: {[float(s) for s in sigmas[:num_steps+1]]}")
    print(f"  init_noise_sigma: {float(scheduler.init_noise_sigma)}")

    stats.append({"name": "timesteps", "values": [float(t) for t in timesteps]})
    stats.append({"name": "sigmas", "values": [float(s) for s in sigmas]})
    stats.append({"name": "init_noise_sigma", "value": float(scheduler.init_noise_sigma)})

    # Scale initial noise
    latents = latents * scheduler.init_noise_sigma
    stats.append(tensor_stats("scaled_noise", latents))
    latents.numpy().tofile(os.path.join(output_dir, "scaled_noise.bin"))
    print(f"  scaled noise mean={float(latents.mean()):.6f}, std={float(latents.std()):.6f}")

    # =========================================================================
    # 6. Denoising loop
    # =========================================================================
    print(f"\n=== Denoising ({num_steps} steps, cfg={cfg_scale}) ===")

    unet = pipe.unet
    unet.eval()

    for i, t in enumerate(timesteps):
        # Scale model input (Euler scheduler divides by sqrt(sigma^2 + 1))
        latent_model_input = scheduler.scale_model_input(latents, t)

        if i < 3:
            stats.append(tensor_stats(f"step{i}_scaled_input", latent_model_input))
            latent_model_input.numpy().tofile(os.path.join(output_dir, f"step{i}_scaled_input.bin"))

        # Duplicate for CFG: [uncond, cond]
        latent_model_input_cfg = torch.cat([latent_model_input] * 2)

        # Prepare cross-attention kwargs
        # text_embeddings is [2, 77, 2048] — [neg, pos] already in CFG order
        # pooled is [2, 1280]
        added_cond_kwargs = {
            "text_embeds": clip_g_pooled,  # [2, 1280]
            "time_ids": add_time_ids_cfg,  # [2, 6]
        }

        with torch.no_grad():
            noise_pred = unet(
                latent_model_input_cfg,
                t,
                encoder_hidden_states=text_embeddings,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

        # Split predictions
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

        if i < 3:
            stats.append(tensor_stats(f"step{i}_noise_pred_uncond", noise_pred_uncond))
            stats.append(tensor_stats(f"step{i}_noise_pred_cond", noise_pred_cond))
            noise_pred_uncond.numpy().tofile(os.path.join(output_dir, f"step{i}_noise_pred_uncond.bin"))
            noise_pred_cond.numpy().tofile(os.path.join(output_dir, f"step{i}_noise_pred_cond.bin"))

        # CFG blend
        noise_pred_cfg = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

        if i < 3:
            stats.append(tensor_stats(f"step{i}_noise_pred_cfg", noise_pred_cfg))

        # Scheduler step
        latents = scheduler.step(noise_pred_cfg, t, latents).prev_sample

        step_stats = tensor_stats(f"step{i}_latents_after", latents)
        stats.append(step_stats)
        print(f"  Step {i}: t={float(t):.1f}, latent mean={step_stats['mean']:.6f}, std={step_stats['std']:.6f}")

        if i < 3:
            latents.numpy().tofile(os.path.join(output_dir, f"step{i}_latents_after.bin"))

    # Save final latents
    latents.numpy().tofile(os.path.join(output_dir, "final_latents.bin"))
    stats.append(tensor_stats("final_latents", latents))

    # =========================================================================
    # 7. VAE Decode
    # =========================================================================
    print(f"\n=== VAE Decode ===")
    vae = pipe.vae
    vae_scaling = 0.13025

    with torch.no_grad():
        latents_scaled = latents / vae_scaling
        stats.append(tensor_stats("vae_input_scaled", latents_scaled))
        image = vae.decode(latents_scaled).sample

    stats.append(tensor_stats("vae_output", image))

    image_clamped = ((image + 1.0) / 2.0).clamp(0, 1)
    stats.append(tensor_stats("image_clamped_01", image_clamped))

    # Save as image for visual comparison
    image_np = (image_clamped[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    try:
        from PIL import Image
        img = Image.fromarray(image_np)
        img.save(os.path.join(output_dir, "reference_output.png"))
        print(f"  Reference image saved to {output_dir}/reference_output.png")
    except ImportError:
        print("  PIL not available, skipping image save")

    print(f"  VAE output: mean={float(image.mean()):.6f}, std={float(image.std()):.6f}")

    # =========================================================================
    # Save stats
    # =========================================================================
    stats_path = os.path.join(output_dir, "sdxl_reference_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== Done ===")
    print(f"Stats saved to: {stats_path}")
    print(f"Binary tensors saved to: {output_dir}/")
    print(f"Total stats entries: {len(stats)}")

if __name__ == "__main__":
    main()
