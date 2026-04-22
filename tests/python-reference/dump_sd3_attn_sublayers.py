"""
Dumps sub-operation intermediate tensors from the first SD3 JointTransformerBlock.
Saves each sub-step: AdaLN modulation, Q/K/V projections, QK-norm, joint attention,
gated residuals, and SwiGLU MLP — for pinpointing the exact divergent operation
when C# layer outputs don't match.

Usage: python dump_sd3_attn_sublayers.py <model_dir> [output_dir]
Requires: pip install diffusers transformers torch safetensors sentencepiece accelerate
"""
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path


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
            "abs_mean": float(flat.abs().mean()),
            "first_8": [float(x) for x in flat[:8]],
        }


def save_tensor(t: torch.Tensor, path: str):
    t.float().cpu().numpy().tofile(path)


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "sd3_reference_tensors/attn_sublayers"

    if model_dir is None:
        print("Usage: python dump_sd3_attn_sublayers.py <model_dir> [output_dir]")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    from diffusers import StableDiffusion3Pipeline

    device = "cpu"
    dtype = torch.float32

    # Load pipeline
    print(f"Loading SD3 pipeline from {model_dir}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(model_dir, torch_dtype=dtype).to(device)
    transformer = pipe.transformer

    # --- Prepare same inputs as other scripts ---
    prompt = "A photograph of an astronaut riding a horse"
    seed = 42
    width, height = 1024, 1024
    num_steps = 28

    # Encode text
    print("Encoding text...")
    clip_l_tokens = pipe.tokenizer(prompt, max_length=77, padding="max_length",
                                    truncation=True, return_tensors="pt")
    clip_g_tokens = pipe.tokenizer_2(prompt, max_length=77, padding="max_length",
                                      truncation=True, return_tensors="pt")
    t5_tokens = pipe.tokenizer_3(prompt, max_length=77, padding="max_length",
                                  truncation=True, return_tensors="pt")

    with torch.no_grad():
        clip_l_out = pipe.text_encoder(clip_l_tokens["input_ids"].to(device), output_hidden_states=True)
        clip_l_hidden = clip_l_out.hidden_states[-2]
        clip_l_pooled = clip_l_out.text_embeds

        clip_g_out = pipe.text_encoder_2(clip_g_tokens["input_ids"].to(device), output_hidden_states=True)
        clip_g_hidden = clip_g_out.hidden_states[-2]
        clip_g_pooled = clip_g_out.text_embeds

        t5_out = pipe.text_encoder_3(t5_tokens["input_ids"].to(device))
        t5_hidden = t5_out.last_hidden_state

    # Combine conditioning
    lg_hidden = torch.cat([clip_l_hidden, clip_g_hidden], dim=-1)
    lg_padded = torch.nn.functional.pad(lg_hidden, (0, 4096 - 2048))
    combined_context = torch.cat([lg_padded, t5_hidden], dim=1)
    pooled = torch.cat([clip_l_pooled, clip_g_pooled], dim=-1)

    with torch.no_grad():
        encoder_hidden_states = transformer.context_embedder(combined_context)

    # Create noise and get first timestep
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn((1, 16, height // 8, width // 8), generator=generator, device=device, dtype=dtype)
    pipe.scheduler.set_timesteps(num_steps)
    latents = latents * pipe.scheduler.init_noise_sigma
    timestep = pipe.scheduler.timesteps[0]

    print(f"Timestep: {float(timestep):.1f}")

    # --- Manually step through the transformer to capture sub-operations ---
    print("\nManually stepping through transformer...")
    stats = []

    with torch.no_grad():
        # 1. Patch embedding
        # The transformer's forward starts by patch-embedding the latent
        # We need to replicate the transformer's preprocessing
        hidden_states = latents

        # Patch embed: Conv2d(16, hidden_size, kernel_size=patch_size, stride=patch_size)
        # Then flatten and transpose
        height_tokens = hidden_states.shape[2] // transformer.config.patch_size
        width_tokens = hidden_states.shape[3] // transformer.config.patch_size

        hidden_states = transformer.pos_embed(hidden_states)
        save_tensor(hidden_states, os.path.join(output_dir, "01_patch_embed.bin"))
        stats.append(tensor_stats("01_patch_embed", hidden_states))
        print(f"  01_patch_embed: {hidden_states.shape}, mean={float(hidden_states.mean()):.6f}")

        # 2. Timestep + pooled embedding
        timestep_expanded = timestep.unsqueeze(0).to(device)
        temb = transformer.time_text_embed(timestep_expanded, pooled)
        save_tensor(temb, os.path.join(output_dir, "02_timestep_embed.bin"))
        stats.append(tensor_stats("02_timestep_embed", temb))
        print(f"  02_timestep_embed: {temb.shape}, mean={float(temb.mean()):.6f}")

        # 3. Step through block 0 manually
        block = transformer.transformer_blocks[0]
        print(f"\n--- Block 0 sub-operations ---")

        # Save block inputs
        save_tensor(hidden_states, os.path.join(output_dir, "03_block0_img_input.bin"))
        save_tensor(encoder_hidden_states, os.path.join(output_dir, "03_block0_ctx_input.bin"))
        stats.append(tensor_stats("03_block0_img_input", hidden_states))
        stats.append(tensor_stats("03_block0_ctx_input", encoder_hidden_states))

        # AdaLN modulation for image stream (norm1)
        if hasattr(block, 'norm1'):
            norm1_out = block.norm1(hidden_states, emb=temb)
            if isinstance(norm1_out, tuple):
                img_normed = norm1_out[0]
                img_gate = norm1_out[1] if len(norm1_out) > 1 else None
            else:
                img_normed = norm1_out
                img_gate = None
            save_tensor(img_normed, os.path.join(output_dir, "04_img_adaln_normed.bin"))
            stats.append(tensor_stats("04_img_adaln_normed", img_normed))
            print(f"  04_img_adaln_normed: {img_normed.shape}, mean={float(img_normed.mean()):.6f}")
            if img_gate is not None:
                save_tensor(img_gate, os.path.join(output_dir, "04b_img_gate.bin"))
                stats.append(tensor_stats("04b_img_gate", img_gate))

        # AdaLN modulation for context stream (norm1_context)
        if hasattr(block, 'norm1_context'):
            ctx_norm_out = block.norm1_context(encoder_hidden_states, emb=temb)
            if isinstance(ctx_norm_out, tuple):
                ctx_normed = ctx_norm_out[0]
                ctx_gate = ctx_norm_out[1] if len(ctx_norm_out) > 1 else None
            else:
                ctx_normed = ctx_norm_out
                ctx_gate = None
            save_tensor(ctx_normed, os.path.join(output_dir, "05_ctx_adaln_normed.bin"))
            stats.append(tensor_stats("05_ctx_adaln_normed", ctx_normed))
            print(f"  05_ctx_adaln_normed: {ctx_normed.shape}, mean={float(ctx_normed.mean()):.6f}")
            if ctx_gate is not None:
                save_tensor(ctx_gate, os.path.join(output_dir, "05b_ctx_gate.bin"))
                stats.append(tensor_stats("05b_ctx_gate", ctx_gate))

        # Q/K/V projections (image stream)
        attn = block.attn
        inner_dim = attn.to_q.out_features
        head_dim = inner_dim // attn.heads

        img_q = attn.to_q(img_normed)
        img_k = attn.to_k(img_normed)
        img_v = attn.to_v(img_normed)

        save_tensor(img_q, os.path.join(output_dir, "06_img_q_proj.bin"))
        save_tensor(img_k, os.path.join(output_dir, "06_img_k_proj.bin"))
        save_tensor(img_v, os.path.join(output_dir, "06_img_v_proj.bin"))
        stats.append(tensor_stats("06_img_q_proj", img_q))
        stats.append(tensor_stats("06_img_k_proj", img_k))
        stats.append(tensor_stats("06_img_v_proj", img_v))
        print(f"  06_img_q: {img_q.shape}, heads={attn.heads}, head_dim={head_dim}")

        # Q/K/V projections (context stream)
        if hasattr(attn, 'add_q_proj'):
            ctx_q = attn.add_q_proj(ctx_normed)
            ctx_k = attn.add_k_proj(ctx_normed)
            ctx_v = attn.add_v_proj(ctx_normed)
        else:
            ctx_q = attn.to_q(ctx_normed)
            ctx_k = attn.to_k(ctx_normed)
            ctx_v = attn.to_v(ctx_normed)

        save_tensor(ctx_q, os.path.join(output_dir, "07_ctx_q_proj.bin"))
        save_tensor(ctx_k, os.path.join(output_dir, "07_ctx_k_proj.bin"))
        save_tensor(ctx_v, os.path.join(output_dir, "07_ctx_v_proj.bin"))
        stats.append(tensor_stats("07_ctx_q_proj", ctx_q))
        stats.append(tensor_stats("07_ctx_k_proj", ctx_k))
        stats.append(tensor_stats("07_ctx_v_proj", ctx_v))
        print(f"  07_ctx_q: {ctx_q.shape}")

        # QK-norm (if present)
        if hasattr(attn, 'norm_q') and attn.norm_q is not None:
            # Reshape to [B, seq, heads, head_dim] for per-head norm
            batch_size = img_q.shape[0]
            img_q_mh = img_q.view(batch_size, -1, attn.heads, head_dim)
            img_k_mh = img_k.view(batch_size, -1, attn.heads, head_dim)
            ctx_q_mh = ctx_q.view(batch_size, -1, attn.heads, head_dim)
            ctx_k_mh = ctx_k.view(batch_size, -1, attn.heads, head_dim)

            img_q_normed = attn.norm_q(img_q_mh)
            img_k_normed = attn.norm_k(img_k_mh)
            ctx_q_normed = attn.norm_q(ctx_q_mh) if not hasattr(attn, 'norm_added_q') else attn.norm_added_q(ctx_q_mh)
            ctx_k_normed = attn.norm_k(ctx_k_mh) if not hasattr(attn, 'norm_added_k') else attn.norm_added_k(ctx_k_mh)

            save_tensor(img_q_normed, os.path.join(output_dir, "08_img_q_qknorm.bin"))
            save_tensor(img_k_normed, os.path.join(output_dir, "08_img_k_qknorm.bin"))
            save_tensor(ctx_q_normed, os.path.join(output_dir, "08_ctx_q_qknorm.bin"))
            save_tensor(ctx_k_normed, os.path.join(output_dir, "08_ctx_k_qknorm.bin"))
            stats.append(tensor_stats("08_img_q_qknorm", img_q_normed))
            stats.append(tensor_stats("08_img_k_qknorm", img_k_normed))
            stats.append(tensor_stats("08_ctx_q_qknorm", ctx_q_normed))
            stats.append(tensor_stats("08_ctx_k_qknorm", ctx_k_normed))
            print(f"  08_qk_norm applied: img_q_normed={img_q_normed.shape}")
        else:
            print(f"  08_qk_norm: NOT present in this model")

        # Now run the full block to get actual output
        block_output = block(hidden_states, encoder_hidden_states, temb=temb)

        if isinstance(block_output, tuple):
            img_output, ctx_output = block_output[0], block_output[1] if len(block_output) > 1 else None
        else:
            img_output = block_output
            ctx_output = None

        save_tensor(img_output, os.path.join(output_dir, "10_block0_img_output.bin"))
        stats.append(tensor_stats("10_block0_img_output", img_output))
        print(f"  10_block0_img_output: {img_output.shape}, mean={float(img_output.mean()):.6f}")

        if ctx_output is not None:
            save_tensor(ctx_output, os.path.join(output_dir, "10_block0_ctx_output.bin"))
            stats.append(tensor_stats("10_block0_ctx_output", ctx_output))
            print(f"  10_block0_ctx_output: {ctx_output.shape}, mean={float(ctx_output.mean()):.6f}")

    # --- Dump attention config for verification ---
    attn_info = {
        "num_heads": attn.heads,
        "head_dim": head_dim,
        "inner_dim": inner_dim,
        "has_qk_norm": hasattr(attn, 'norm_q') and attn.norm_q is not None,
        "has_separate_ctx_proj": hasattr(attn, 'add_q_proj'),
        "scale": 1.0 / (head_dim ** 0.5),
    }
    stats.append({"name": "attention_config", **attn_info})
    print(f"\nAttention config: {json.dumps(attn_info, indent=2)}")

    # Save stats
    stats_path = os.path.join(output_dir, "attn_sublayer_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nAll sub-layer tensors saved to {output_dir}")
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
