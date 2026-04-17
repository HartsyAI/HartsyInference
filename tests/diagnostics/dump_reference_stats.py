"""
Dumps intermediate tensor statistics from a diffusers SD1.5 pipeline run.
Use these as reference values to compare against SharpInference C# output.

Usage: python dump_reference_stats.py <model_dir> [output_file]
Requires: pip install diffusers transformers torch safetensors
"""
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file

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
    model_dir = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\AI Overlord\Desktop\Projects\SharpInference\tests\test-models\sd15"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "reference_stats.json"

    from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler
    from transformers import CLIPTextModel, CLIPTextConfig, CLIPTokenizer

    device = "cpu"
    dtype = torch.float32

    print("Loading models...")

    # Load tokenizer (has vocab.json and merges.txt)
    tokenizer = CLIPTokenizer(
        vocab_file=os.path.join(model_dir, "tokenizer", "vocab.json"),
        merges_file=os.path.join(model_dir, "tokenizer", "merges.txt"),
    )

    # Build text encoder from config + load fp16 weights as fp32
    clip_config = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=77,
    )
    text_encoder = CLIPTextModel(clip_config).to(dtype).to(device)
    te_weights = load_file(os.path.join(model_dir, "text_encoder", "model.fp16.safetensors"))
    # Cast fp16 to fp32
    te_weights = {k: v.float() for k, v in te_weights.items()}
    text_encoder.load_state_dict(te_weights, strict=False)
    text_encoder.eval()

    # Build UNet from config + load fp16 weights as fp32
    unet = UNet2DConditionModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(320, 640, 1280, 1280),
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        cross_attention_dim=768,
        attention_head_dim=8,
        norm_num_groups=32,
    ).to(dtype).to(device)
    unet_weights = load_file(os.path.join(model_dir, "unet", "diffusion_pytorch_model.fp16.safetensors"))
    unet_weights = {k: v.float() for k, v in unet_weights.items()}
    unet.load_state_dict(unet_weights, strict=False)
    unet.eval()

    # Build VAE from config + load fp16 weights as fp32
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        norm_num_groups=32,
    ).to(dtype).to(device)
    vae_weights = load_file(os.path.join(model_dir, "vae", "diffusion_pytorch_model.fp16.safetensors"))
    vae_weights = {k: v.float() for k, v in vae_weights.items()}
    vae.load_state_dict(vae_weights, strict=False)
    vae.eval()

    # Scheduler (hardcoded SD1.5 defaults)
    scheduler = EulerDiscreteScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
    )

    # Settings matching our C# test
    prompt = "a painting of a cat sitting on a windowsill"
    negative_prompt = "blurry, bad quality"
    width, height = 256, 256
    num_steps = 20
    cfg_scale = 7.5
    seed = 42

    stats = []

    # 1. Tokenize
    print("Tokenizing...")
    prompt_inputs = tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    neg_inputs = tokenizer(negative_prompt, padding="max_length", max_length=77, return_tensors="pt")

    prompt_ids = prompt_inputs.input_ids
    neg_ids = neg_inputs.input_ids

    stats.append({"name": "prompt_token_ids", "values": prompt_ids[0].tolist()})
    stats.append({"name": "negative_token_ids", "values": neg_ids[0].tolist()})

    # 2. Text encoding
    print("Encoding text...")
    with torch.no_grad():
        prompt_embeds = text_encoder(prompt_ids)[0]  # last_hidden_state
        neg_embeds = text_encoder(neg_ids)[0]

    stats.append(tensor_stats("prompt_embeddings", prompt_embeds))
    stats.append(tensor_stats("negative_embeddings", neg_embeds))

    # Concatenate: [neg, pos] for CFG
    text_embeddings = torch.cat([neg_embeds, prompt_embeds], dim=0)  # [2, 77, 768]
    stats.append(tensor_stats("text_embeddings_concat", text_embeddings))

    # 3. Initial noise
    print("Generating noise...")
    generator = torch.Generator(device=device).manual_seed(seed)
    latent_shape = (1, 4, height // 8, width // 8)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
    stats.append(tensor_stats("initial_noise", latents))

    # 4. Scheduler setup
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps
    stats.append({"name": "timesteps", "values": [float(t) for t in timesteps]})
    stats.append({"name": "sigmas", "values": [float(s) for s in scheduler.sigmas]})
    stats.append({"name": "initial_noise_sigma", "value": float(scheduler.init_noise_sigma)})

    # Scale initial noise
    latents = latents * scheduler.init_noise_sigma
    stats.append(tensor_stats("scaled_noise", latents))

    # 5. Denoising loop (first 3 steps only for diagnostics)
    print("Running denoising steps...")
    diag_steps = min(3, num_steps)

    for i, t in enumerate(timesteps[:diag_steps]):
        # Scale model input
        latent_model_input = scheduler.scale_model_input(latents, t)
        stats.append(tensor_stats(f"step{i}_scaled_input", latent_model_input))

        # Expand for CFG: [2, 4, H, W]
        latent_model_input_cfg = torch.cat([latent_model_input] * 2)

        with torch.no_grad():
            noise_pred = unet(latent_model_input_cfg, t, encoder_hidden_states=text_embeddings).sample

        # Split predictions
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        stats.append(tensor_stats(f"step{i}_noise_pred_uncond", noise_pred_uncond))
        stats.append(tensor_stats(f"step{i}_noise_pred_cond", noise_pred_cond))

        # CFG
        noise_pred_cfg = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        stats.append(tensor_stats(f"step{i}_noise_pred_cfg", noise_pred_cfg))

        # Scheduler step
        latents = scheduler.step(noise_pred_cfg, t, latents).prev_sample
        stats.append(tensor_stats(f"step{i}_latents_after", latents))

        print(f"  Step {i}: t={float(t):.1f}, latent mean={float(latents.mean()):.6f}, std={float(latents.std()):.6f}")

    # 6. VAE decode (on final latents after all steps for full run)
    # Run remaining steps silently
    for i, t in enumerate(timesteps[diag_steps:], start=diag_steps):
        latent_model_input = scheduler.scale_model_input(latents, t)
        latent_model_input_cfg = torch.cat([latent_model_input] * 2)
        with torch.no_grad():
            noise_pred = unet(latent_model_input_cfg, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred_cfg = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        latents = scheduler.step(noise_pred_cfg, t, latents).prev_sample
        if i % 5 == 0:
            print(f"  Step {i}: t={float(t):.1f}, latent mean={float(latents.mean()):.6f}")

    stats.append(tensor_stats("final_latents", latents))

    # VAE decode
    print("Decoding with VAE...")
    with torch.no_grad():
        latents_scaled = latents / 0.18215
        image = vae.decode(latents_scaled).sample

    stats.append(tensor_stats("vae_output", image))

    # Clamp to [0, 1] for image
    image_clamped = ((image + 1.0) / 2.0).clamp(0, 1)
    stats.append(tensor_stats("image_clamped_01", image_clamped))

    # Save stats
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats saved to {output_file}")
    print(f"Total stats entries: {len(stats)}")

if __name__ == "__main__":
    main()
