namespace SharpInference.ModelHandler.Lora;

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
}
