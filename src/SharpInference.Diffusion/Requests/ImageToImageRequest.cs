using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Requests;

/// <summary>Request parameters for image-to-image generation. The pipeline encodes <see cref="SourceImage"/> through the VAE encoder, injects noise at the timestep selected by <see cref="Strength"/>, and runs the denoising loop from there.</summary>
public record ImageToImageRequest : TextToImageRequest
{
    /// <summary>Source image to transform. Shape <c>[1, 3, Height, Width]</c>, F32, values normalized to <c>[-1, 1]</c>. Caller retains ownership and is responsible for disposal.</summary>
    public required Tensor SourceImage { get; init; }

    /// <summary>How much of the source to overwrite. <c>0.0</c> returns the source unchanged; <c>1.0</c> is effectively text-to-image with the source acting only as the noise prior. Default <c>0.75</c>.</summary>
    /// <remarks>Maps to a starting step <c>t_start = steps - round(steps * strength)</c>. Diffusers applies the same formula.</remarks>
    public float Strength { get; init; } = 0.75f;
}
