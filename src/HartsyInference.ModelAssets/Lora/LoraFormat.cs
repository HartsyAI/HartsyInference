namespace HartsyInference.ModelAssets.Lora;

/// <summary>LoRA file naming format detected from key prefixes. The detection rules live in <see cref="LoraFormatDetector.Detect"/>.</summary>
public enum LoraFormat
{
    /// <summary>Format could not be detected — file rejected at load.</summary>
    Unknown,

    /// <summary>Kohya/sd-scripts SD1.5 format: lora_unet_*, lora_te_* keys with 4 down-block levels.</summary>
    KohyaSd15,

    /// <summary>Kohya/sd-scripts SDXL format: lora_unet_*, lora_te1_*, lora_te2_* keys with 3 down-block levels.</summary>
    KohyaSdxl,

    /// <summary>Kohya/sd-scripts Flux format: lora_unet_double_blocks_*, lora_unet_single_blocks_* with fused QKV.</summary>
    KohyaFlux,

    /// <summary>AI Toolkit (ostris/ai-toolkit) Flux format: lora_transformer_* prefix with PEFT-style .lora_A.weight / .lora_B.weight suffixes, no .alpha entries.</summary>
    AiToolkitFlux,

    /// <summary>HuggingFace PEFT Flux format: transformer.transformer_blocks.* keys with .lora_A.weight / .lora_B.weight suffixes, dotted naming throughout.</summary>
    DiffusersFlux,

    /// <summary>Kohya/musubi-tuner Wan format: lora_unet_blocks_{i}_self_attn_* / _cross_attn_* / _ffn_* underscored keys in original Wan module naming.</summary>
    KohyaWan,

    /// <summary>ComfyUI-style Wan format: diffusion_model.blocks.* dotted keys in original Wan module naming, with either PEFT (.lora_A/.lora_B) or kohya (.lora_down/.lora_up) suffixes.</summary>
    DiffusersWan,

    /// <summary>ComfyUI-style BFL format: dotted original module names under a <c>diffusion_model.</c> prefix (<c>diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight</c>) — Chroma/Flux LoRAs trained against ComfyUI checkpoints. Same translation table as KohyaFlux, different root spelling.</summary>
    ComfyBflDit,

    /// <summary>Comfy-style Z-Image format: <c>diffusion_model.{layers|context_refiner|noise_refiner}.{i}.*</c> dotted keys in Tongyi Lumina2/NextDiT module naming with split Q/K/V (Comfy-Org's Turbo distill patch LoRA).</summary>
    ComfyZImageDit,

    /// <summary>PEFT suffixes on BARE checkpoint keys — no transformer./diffusion_model. wrapper at all, so the root is already the canonical weight name (MiniMax-H3's Turbo LoRA: blocks.0.attn.qkv_proj.lora_A.weight). Detected last, so a file carrying any recognized prefix never lands here.</summary>
    DiffusersBareDit,

    /// <summary>Official LightX MiniMax-H3 Diffusers roots with split Q/K/V and <c>.lora_A.default</c> roles.</summary>
    DiffusersMiniMaxH3,

    /// <summary>FLUX.3 in any of its spellings — diffusers/PEFT under <c>transformer.</c> (<c>transformer.mode_blocks.image.0.attn.to_q.lora_A.weight</c>), BFL names under <c>diffusion_model.</c> or bare, or kohya <c>lora_unet_</c> — recognized by the modules only FLUX.3 has.</summary>
    Flux3,

    /// <summary>MOSS-Image in any of its spellings — diffusers/PEFT under <c>transformer.</c>, ComfyUI under <c>diffusion_model.</c>, bare, or kohya <c>lora_unet_</c> — recognized by the modules only MOSS-Image has (<c>layers.{i}.proj.ffn_up_gate</c>, <c>shared_block_modulation</c>, …).</summary>
    MossImage,

    /// <summary>Fibo 2.0 in its ComfyUI-native spelling (<c>diffusion_model.layers.N.attention.qkv.lora_down.weight</c> on the fused modules, with <c>.diff</c>/<c>.diff_b</c> full-weight deltas) or its diffusers/PEFT one (<c>transformer.layers.N.attention.to_q…</c>), recognized by the modules only Fibo 2.0 has (the Perceiver, the injection blocks' cross-attention, the fused <c>feed_forward.w13</c>).</summary>
    Fibo2,

    /// <summary>Kohya/musubi-tuner <c>lora_unet_&lt;module path with '.' written as '_'&gt;</c> keys no specific kohya layout claims (Krea 2: <c>lora_unet_blocks_0_attn_wq</c>). The module names are resolved against the target model's own weight names when the LoRA is applied, as ComfyUI does, so any DiT works without a per-family table. Detected after every other format.</summary>
    KohyaGeneric,
}
