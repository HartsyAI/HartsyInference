using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Requests;

/// <summary>Request parameters for image-to-image generation. The pipeline encodes <see cref="SourceImage"/> through the VAE encoder, injects noise at the timestep selected by <see cref="Strength"/>, and runs the denoising loop from there.
/// <para>If <see cref="Mask"/> is supplied, the pipeline runs in <b>blend-on-vanilla inpaint</b> mode: any standard checkpoint can inpaint by re-noising the source-derived latent each step and blending it into the unmasked region. This is the same "soft inpainting" path diffusers' <c>StableDiffusionInpaintPipelineLegacy</c> implements; it works without a dedicated 9-channel inpaint UNet.</para></summary>
public record ImageToImageRequest : TextToImageRequest
{
    /// <summary>Source image to transform. Shape <c>[1, 3, Height, Width]</c>, F32, values normalized to <c>[-1, 1]</c>. Caller retains ownership and is responsible for disposal.</summary>
    public required Tensor SourceImage { get; init; }

    /// <summary>How much of the source to overwrite. <c>0.0</c> returns the source unchanged; <c>1.0</c> is effectively text-to-image with the source acting only as the noise prior. Default <c>0.75</c>.</summary>
    /// <remarks>Maps to a starting step <c>t_start = steps - round(steps * strength)</c>. Diffusers applies the same formula.</remarks>
    public float Strength { get; init; } = 0.75f;

    /// <summary>Optional inpainting mask in pixel space, shape <c>[1, 1, Height, Width]</c>, F32, values in <c>[0, 1]</c>. Convention: <b>1 = inpaint (let the model regenerate this region), 0 = preserve (keep the source pixel)</b>. Gray values blend proportionally. Caller retains ownership and is responsible for disposal.
    /// <para>When non-null the pipeline downsamples this mask to latent resolution and uses it to blend the denoised latent with a freshly re-noised copy of the source latent at every step, keeping the unmasked region temporally consistent with the source's noise trajectory. The pipeline also applies a final pixel-space recomposite (controlled by <see cref="RecompositeAtEnd"/>) to suppress VAE round-trip artifacts in unmasked regions.</para></summary>
    public Tensor? Mask { get; init; }

    /// <summary>When <c>true</c> (default) and a <see cref="Mask"/> is set, the pipeline composites the decoded image back over the source pixels using the mask: <c>output = decoded * mask + source * (1 - mask)</c>. This eliminates VAE encode/decode drift in unmasked regions, which would otherwise accumulate visibly across repeated inpaint operations. Set to <c>false</c> only if you need the raw VAE-decoded output (e.g. for downstream stages that re-encode immediately).</summary>
    public bool RecompositeAtEnd { get; init; } = true;
}
