namespace HartsyInference.Diffusion.Quality;

/// <summary>Top-level VRAM/quality tradeoff selector for diffusion pipelines. Maps to a <see cref="QualityProfile"/>.</summary>
public enum QualityPreset
{
    /// <summary>FP16 backbone + FP16 text encoders + FP16 VAE. Highest quality; needs 24+ GB VRAM for large DiT.</summary>
    Maximum,

    /// <summary>FP8 backbone + FP16 text encoders + FP16 VAE. Default for 12 GB consumer GPUs running Flux/SD3.5.</summary>
    High,

    /// <summary>Q8_0 backbone + FP8 T5 + FP16 CLIP/VAE. Tighter fit; depends on GGUF K-quant reader.</summary>
    Medium,

    /// <summary>Q4_K backbone + Q4_K T5 + FP16 CLIP/VAE. Last-resort fit on 8 GB cards. Visible quality drop.</summary>
    Low,

    /// <summary>Caller supplies a <see cref="QualityProfile"/> directly.</summary>
    Custom,
}
