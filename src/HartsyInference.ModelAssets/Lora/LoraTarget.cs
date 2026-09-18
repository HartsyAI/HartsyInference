namespace HartsyInference.ModelAssets.Lora;

/// <summary>Which model component a LoRA layer targets. Determines which weight dictionary the apply step looks up.</summary>
public enum LoraTarget
{
    /// <summary>Targets the SD1.5 / SDXL UNet weight dictionary.</summary>
    UNet,

    /// <summary>Targets the Flux / Flux.2 / Z-Image / etc. transformer weight dictionary.</summary>
    Transformer,

    /// <summary>Targets the CLIP-L (OpenAI/HuggingFace base) text encoder weight dictionary.</summary>
    ClipL,

    /// <summary>Targets the CLIP-G (OpenCLIP bigG) text encoder weight dictionary — SDXL only.</summary>
    ClipG,

    /// <summary>Targets the second text encoder: the T5/umT5/Llama/Qwen-class encoder a modern pipeline pairs with CLIP, or its only encoder when it has no CLIP arm at all.</summary>
    TextEncoder2,
}
