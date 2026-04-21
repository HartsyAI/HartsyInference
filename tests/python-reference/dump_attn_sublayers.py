"""
Dumps sub-layer outputs from down_blocks.0.attentions.0 (first CrossAttentionBlock)
to isolate exactly where C# diverges from Python.
"""
import os
import json
import torch
import numpy as np
from safetensors.torch import load_file


def main():
    model_dir = r"C:\Users\AI Overlord\Desktop\Projects\SharpInference\tests\test-models\sd15"
    ref_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_tensors")
    out_dir = os.path.join(ref_dir, "attn0_sublayers")
    os.makedirs(out_dir, exist_ok=True)

    from diffusers import UNet2DConditionModel

    # Load UNet
    unet = UNet2DConditionModel(
        sample_size=32, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(320, 640, 1280, 1280),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=768, attention_head_dim=8, norm_num_groups=32,
    ).float().cpu()
    weights = load_file(os.path.join(model_dir, "unet", "diffusion_pytorch_model.fp16.safetensors"))
    weights = {k: v.float() for k, v in weights.items()}
    unet.load_state_dict(weights, strict=False)
    unet.eval()

    # Load the exact input that enters down_blocks.0.attentions.0
    # This is the output of down_blocks.0.resnets.0
    resnets0_out_path = os.path.join(ref_dir, "layers", "down_blocks_0_resnets_0.bin")
    resnets0_out = torch.from_numpy(np.fromfile(resnets0_out_path, dtype=np.float32).reshape(1, 320, 32, 32))

    # Load text embeddings
    text_emb_path = os.path.join(ref_dir, "unet_step0_text_emb.bin")
    text_emb = torch.from_numpy(np.fromfile(text_emb_path, dtype=np.float32).reshape(1, 77, 768))

    print(f"Attn input: mean={resnets0_out.mean():.6f}, std={resnets0_out.std():.6f}")
    print(f"Text emb:   mean={text_emb.mean():.6f}, std={text_emb.std():.6f}")

    # Access the first attention block directly
    attn_block = unet.down_blocks[0].attentions[0]

    # Step through manually
    with torch.no_grad():
        hidden = resnets0_out
        residual = hidden

        # 1. GroupNorm
        normed = attn_block.norm(hidden)
        save(out_dir, "01_groupnorm", normed)

        # 2. proj_in (1x1 Conv2d for SD1.5, operates on spatial [B,C,H,W])
        B, C, H, W = normed.shape
        is_conv = isinstance(attn_block.proj_in, torch.nn.Conv2d)
        print(f"  proj_in type: {type(attn_block.proj_in).__name__}, is_conv={is_conv}")

        if is_conv:
            projected_spatial = attn_block.proj_in(normed)
            save(out_dir, "02_proj_in_spatial", projected_spatial)
            inner_dim = projected_spatial.shape[1]
            projected = projected_spatial.permute(0, 2, 3, 1).reshape(B, H*W, inner_dim)
        else:
            normed_seq = normed.permute(0, 2, 3, 1).reshape(B, H*W, C)
            projected = attn_block.proj_in(normed_seq)
            inner_dim = projected.shape[-1]
        save(out_dir, "03_proj_in", projected)

        # 4. Self-attention (transformer_blocks.0.attn1)
        tb = attn_block.transformer_blocks[0]
        # norm1
        norm1_out = tb.norm1(projected)
        save(out_dir, "04_norm1_out", norm1_out)

        # attn1 (self-attention)
        attn1 = tb.attn1
        q = attn1.to_q(norm1_out)
        k = attn1.to_k(norm1_out)
        v = attn1.to_v(norm1_out)
        save(out_dir, "05_self_q", q)
        save(out_dir, "05_self_k", k)
        save(out_dir, "05_self_v", v)

        # Reshape Q/K/V to multi-head
        num_heads = attn1.heads
        inner_dim_attn = q.shape[-1]
        head_dim = inner_dim_attn // num_heads
        print(f"\nSelf-attn: num_heads={num_heads}, head_dim={head_dim}")

        q_mh = q.view(B, -1, num_heads, head_dim).transpose(1, 2)  # [B, H, S, D]
        k_mh = k.view(B, -1, num_heads, head_dim).transpose(1, 2)
        v_mh = v.view(B, -1, num_heads, head_dim).transpose(1, 2)
        save(out_dir, "06_self_q_mh", q_mh)
        save(out_dir, "06_self_k_mh", k_mh)

        # Compute attention
        scale = head_dim ** -0.5
        attn_weights = torch.matmul(q_mh, k_mh.transpose(-2, -1)) * scale
        save(out_dir, "07_self_attn_logits", attn_weights)

        attn_probs = torch.softmax(attn_weights, dim=-1)
        save(out_dir, "08_self_attn_probs", attn_probs)

        attn_output = torch.matmul(attn_probs, v_mh)
        save(out_dir, "09_self_attn_output_mh", attn_output)

        # Reshape back
        attn_output_merged = attn_output.transpose(1, 2).reshape(B, -1, C)
        save(out_dir, "10_self_attn_merged", attn_output_merged)

        # to_out
        attn_out_proj = attn1.to_out[0](attn_output_merged)  # Linear
        # dropout is the second (attn1.to_out[1])
        save(out_dir, "11_self_to_out", attn_out_proj)

        # Residual
        self_attn_result = attn_out_proj + projected
        save(out_dir, "12_self_attn_residual", self_attn_result)

        # 5. Cross-attention (norm2 + attn2)
        norm2_out = tb.norm2(self_attn_result)
        save(out_dir, "13_norm2_out", norm2_out)

        attn2 = tb.attn2
        q2 = attn2.to_q(norm2_out)
        k2 = attn2.to_k(text_emb)
        v2 = attn2.to_v(text_emb)
        save(out_dir, "14_cross_q", q2)
        save(out_dir, "14_cross_k", k2)
        save(out_dir, "14_cross_v", v2)

        # Cross-attn SDPA
        q2_mh = q2.view(B, -1, num_heads, head_dim).transpose(1, 2)
        k2_mh = k2.view(B, -1, num_heads, head_dim).transpose(1, 2)
        v2_mh = v2.view(B, -1, num_heads, head_dim).transpose(1, 2)

        cross_logits = torch.matmul(q2_mh, k2_mh.transpose(-2, -1)) * scale
        cross_probs = torch.softmax(cross_logits, dim=-1)
        cross_out = torch.matmul(cross_probs, v2_mh)
        cross_merged = cross_out.transpose(1, 2).reshape(B, -1, C)
        cross_proj = attn2.to_out[0](cross_merged)
        cross_result = cross_proj + self_attn_result
        save(out_dir, "15_cross_attn_residual", cross_result)

        # 6. FFN (norm3 + ff)
        norm3_out = tb.norm3(cross_result)
        save(out_dir, "16_norm3_out", norm3_out)

        ff_out = tb.ff(norm3_out)
        save(out_dir, "17_ff_out", ff_out)

        ff_result = ff_out + cross_result
        save(out_dir, "18_ff_residual", ff_result)

        # 7. proj_out
        is_conv_out = isinstance(attn_block.proj_out, torch.nn.Conv2d)
        if is_conv_out:
            # Reshape to spatial first, then apply conv
            spatial_before_proj = ff_result.reshape(B, H, W, inner_dim).permute(0, 3, 1, 2)
            proj_out = attn_block.proj_out(spatial_before_proj)
            save(out_dir, "19_proj_out", proj_out)
            final = proj_out + residual
        else:
            proj_out = attn_block.proj_out(ff_result)
            save(out_dir, "19_proj_out", proj_out)
            spatial_out = proj_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
            final = spatial_out + residual
        save(out_dir, "20_final_output", final)

    # Print summary
    print(f"\nSaved sub-layer outputs to {out_dir}")


def save(out_dir, name, tensor):
    fpath = os.path.join(out_dir, f"{name}.bin")
    arr = tensor.detach().numpy()
    arr.tofile(fpath)
    shape_str = str(list(tensor.shape))
    print(f"  {name:<35} shape={shape_str:<30} mean={arr.mean():>10.6f} std={arr.std():>10.6f}")


if __name__ == "__main__":
    main()
