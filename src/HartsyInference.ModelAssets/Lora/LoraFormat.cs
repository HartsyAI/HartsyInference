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

    /// <summary>PEFT suffixes on BARE checkpoint keys — no transformer./diffusion_model. wrapper at all, so the root is already the canonical weight name (MiniMax-H3's Turbo LoRA: blocks.0.attn.qkv_proj.lora_A.weight). Detected last, so a file carrying any recognized prefix never lands here.</summary>
    DiffusersBareDit,
}
