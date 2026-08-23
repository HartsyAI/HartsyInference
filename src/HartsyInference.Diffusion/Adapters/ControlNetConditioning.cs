using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>One ControlNet's contribution to a generation: the loaded adapter, the per-generation control image, and a strength multiplier. Pipelines accept a list of these so users can stack ControlNets (e.g. canny + depth simultaneously) without changing the call signature. The <see cref="ConditionImage"/> dtype doesn't have to match the UNet — <see cref="ControlNet.Forward"/> casts internally if needed. Caller retains ownership of both the adapter and the image; pipelines only borrow them for the duration of the call.</summary>
public sealed record ControlNetConditioning
{
    /// <summary>The loaded ControlNet adapter (already weights-loaded, ready to call <see cref="ControlNet.Forward"/>).</summary>
    public required ControlNet Adapter { get; init; }

    /// <summary>Control image at the request's pixel resolution: <c>[1, 3, H, W]</c> typically in <c>[-1, 1]</c> or <c>[0, 1]</c> range per the checkpoint's expectation. Caller computes this from the user's input via the appropriate preprocessor (canny edge, depth map, openpose skeleton, …).</summary>
    public required Tensor ConditionImage { get; init; }

    /// <summary>Conditioning strength. <c>0</c> = adapter contributes nothing, <c>1</c> = full strength (the trained-for default), values in between scale linearly. Comfy / A1111 expose this as "ControlNet Strength".</summary>
    public float Scale { get; init; } = 1.0f;

    /// <summary>Step-fraction at which this ControlNet starts contributing. <c>0.0</c> = from the very first step; <c>0.5</c> = only after halfway through the schedule. Same step-fraction window semantics as <see cref="IpAdapterConditioning.StartFraction"/>.</summary>
    public float StartFraction { get; init; } = 0.0f;

    /// <summary>Step-fraction at which this ControlNet stops contributing. <c>1.0</c> = through the last step; e.g. <c>0.5</c> = only the first half (compose-then-release: spatial guidance early, free refinement late).</summary>
    public float EndFraction { get; init; } = 1.0f;

    /// <summary>Union checkpoints (xinsir union SDXL): which conditioning type the control image is. Must be null for single-mode checkpoints and non-null for union checkpoints (<see cref="ControlNetConfig.UnionControlTypeCount"/> &gt; 0) — <see cref="ControlNet.Forward"/> enforces both directions. Stack the same union adapter multiple times with different images/types to combine controls (residuals sum, the diffusers MultiControlNetUnion convention).</summary>
    public SdxlUnionControlType? UnionControlType { get; init; }

    /// <summary>True when the zero-based <paramref name="stepIndex"/> of a <paramref name="totalSteps"/>-step schedule falls inside this adapter's <c>[StartFraction, EndFraction]</c> window.</summary>
    public bool IsActiveAtStep(int stepIndex, int totalSteps)
        => IpAdapterScaleSchedule.StepGate(stepIndex, totalSteps, StartFraction, EndFraction) > 0f;

    /// <summary>Filters a ControlNet stack down to the adapters whose start/end window covers the current step. Returns the original list unchanged when every adapter is active (the common no-window case — no allocation), <c>null</c> when none are, and a trimmed copy otherwise. Pipelines call this once per denoise step.</summary>
    public static IReadOnlyList<ControlNetConditioning>? FilterActive(IReadOnlyList<ControlNetConditioning>? controlNets, int stepIndex, int totalSteps)
    {
        if (controlNets is null || controlNets.Count == 0)
        {
            return null;
        }
        int activeCount = 0;
        for (int i = 0; i < controlNets.Count; i++)
        {
            if (controlNets[i].IsActiveAtStep(stepIndex, totalSteps)) activeCount++;
        }
        if (activeCount == controlNets.Count)
        {
            return controlNets;
        }
        if (activeCount == 0)
        {
            return null;
        }
        List<ControlNetConditioning> active = new(activeCount);
        for (int i = 0; i < controlNets.Count; i++)
        {
            if (controlNets[i].IsActiveAtStep(stepIndex, totalSteps)) active.Add(controlNets[i]);
        }
        return active;
    }
}
