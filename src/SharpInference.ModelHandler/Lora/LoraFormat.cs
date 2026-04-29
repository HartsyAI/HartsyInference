namespace SharpInference.ModelHandler.Lora;

/// <summary>LoRA file naming format detected from key prefixes. See docs/Design/LORA_KEY_MAPPING.md for the detection rules.</summary>
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
}
