using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>One Qwen-Image DiT ControlNet's contribution to a generation: the loaded adapter, the control image, a strength multiplier, and a step window. The Qwen counterpart of <see cref="FluxControlNetConditioning"/> — pipelines accept a list so users can stack ControlNets. Caller retains ownership of both the adapter and the image.</summary>
public sealed record QwenImageControlNetConditioning
{
    /// <summary>The loaded adapter (weights loaded, ready for <see cref="QwenImageControlNet.Forward"/>).</summary>
    public required QwenImageControlNet Adapter { get; init; }

    /// <summary>Control image <c>[1, 3, H, W]</c> F32 in <c>[-1, 1]</c> at the request's pixel resolution. The pipeline VAE-encodes and 2×2-packs it once per generation (Qwen DiT ControlNets condition on the packed latent, not on a pixel-space hint tower).</summary>
    public required Tensor ControlImage { get; init; }

    /// <summary>Conditioning strength (diffusers <c>controlnet_conditioning_scale</c>).</summary>
    public float Scale { get; init; } = 1.0f;

    /// <summary>Step-fraction at which this ControlNet starts contributing.</summary>
    public float StartFraction { get; init; } = 0.0f;

    /// <summary>Step-fraction at which this ControlNet stops contributing.</summary>
    public float EndFraction { get; init; } = 1.0f;

    /// <summary>True when the zero-based <paramref name="stepIndex"/> of a <paramref name="totalSteps"/>-step schedule falls inside this adapter's <c>[StartFraction, EndFraction]</c> window.</summary>
    public bool IsActiveAtStep(int stepIndex, int totalSteps)
        => IpAdapterScaleSchedule.StepGate(stepIndex, totalSteps, StartFraction, EndFraction) > 0f;
}
