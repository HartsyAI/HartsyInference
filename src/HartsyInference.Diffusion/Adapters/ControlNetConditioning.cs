using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>One ControlNet's contribution to a generation: the loaded adapter, the per-generation control image, and a strength multiplier. Pipelines accept a list of these so users can stack ControlNets (e.g. canny + depth simultaneously) without changing the call signature.
/// <para>The <see cref="ConditionImage"/> dtype doesn't have to match the UNet — <see cref="ControlNet.Forward"/> casts internally if needed. Caller retains ownership of both the adapter and the image; pipelines only borrow them for the duration of the call.</para></summary>
public sealed record ControlNetConditioning
{
    /// <summary>The loaded ControlNet adapter (already weights-loaded, ready to call <see cref="ControlNet.Forward"/>).</summary>
    public required ControlNet Adapter { get; init; }

    /// <summary>Control image at the request's pixel resolution: <c>[1, 3, H, W]</c> typically in <c>[-1, 1]</c> or <c>[0, 1]</c> range per the checkpoint's expectation. Caller computes this from the user's input via the appropriate preprocessor (canny edge, depth map, openpose skeleton, …).</summary>
    public required Tensor ConditionImage { get; init; }

    /// <summary>Conditioning strength. <c>0</c> = adapter contributes nothing, <c>1</c> = full strength (the trained-for default), values in between scale linearly. Comfy / A1111 expose this as "ControlNet Strength".</summary>
    public float Scale { get; init; } = 1.0f;
}
