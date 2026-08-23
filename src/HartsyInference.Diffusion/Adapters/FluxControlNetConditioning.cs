using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>One Flux DiT ControlNet's contribution to a generation: the loaded adapter, the control image, a strength multiplier, a step window, and (for union checkpoints) the control mode. The Flux counterpart of <see cref="ControlNetConditioning"/> — pipelines accept a list so users can stack ControlNets, and the same checkpoint can appear multiple times with different images/modes (diffusers <c>FluxMultiControlNetModel</c> union stacking). Caller retains ownership of both the adapter and the image.</summary>
public sealed record FluxControlNetConditioning
{
    /// <summary>The loaded Flux ControlNet adapter (weights loaded, ready for <see cref="FluxControlNet.Forward"/>).</summary>
    public required FluxControlNet Adapter { get; init; }

    /// <summary>Control image <c>[1, 3, H, W]</c> F32 in <c>[-1, 1]</c> at the request's pixel resolution. The pipeline VAE-encodes and 2×2-packs it once per generation (Flux DiT ControlNets condition on the packed latent, not on a pixel-space hint tower).</summary>
    public required Tensor ControlImage { get; init; }

    /// <summary>Conditioning strength (diffusers <c>conditioning_scale</c>). Union-Pro-2.0 recommends 0.7–0.9 depending on the mode.</summary>
    public float Scale { get; init; } = 1.0f;

    /// <summary>Step-fraction at which this ControlNet starts contributing (same semantics as <see cref="ControlNetConditioning.StartFraction"/>).</summary>
    public float StartFraction { get; init; } = 0.0f;

    /// <summary>Step-fraction at which this ControlNet stops contributing.</summary>
    public float EndFraction { get; init; } = 1.0f;

    /// <summary>Union checkpoints: which conditioning type the control image is. Must be null for single-mode / mode-free (Union-Pro-2.0) checkpoints and non-null for union checkpoints with a mode embedder.</summary>
    public FluxUnionControlMode? UnionMode { get; init; }

    /// <summary>True when the zero-based <paramref name="stepIndex"/> of a <paramref name="totalSteps"/>-step schedule falls inside this adapter's <c>[StartFraction, EndFraction]</c> window.</summary>
    public bool IsActiveAtStep(int stepIndex, int totalSteps)
        => IpAdapterScaleSchedule.StepGate(stepIndex, totalSteps, StartFraction, EndFraction) > 0f;
}
