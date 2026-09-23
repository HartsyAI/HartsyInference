namespace HartsyInference.Engine.Planning.Memory;

/// <summary>The model component a phase of a generation keeps on the device. Placement can move the text encoder and
/// the VAE to other GPUs, so the estimate keeps them apart instead of folding them into one number.</summary>
public enum MemoryComponent
{
    /// <summary>The prompt encoder(s): CLIP, T5, umT5, Qwen-VL and friends.</summary>
    TextEncoder,

    /// <summary>The diffusion transformer or UNet that runs every step.</summary>
    Denoiser,

    /// <summary>The latent encoder/decoder.</summary>
    Vae,
}
