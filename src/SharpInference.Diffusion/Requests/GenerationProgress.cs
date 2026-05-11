using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Utilities;

namespace SharpInference.Diffusion.Requests;

/// <summary>Progress report during image generation. Emitted after each denoising step,
/// and (when a heartbeat is enabled) also at fixed wall-clock intervals during long steps
/// so consumers can render smooth progress bars instead of stair-stepping by step count.</summary>
/// <param name="Step">The 0-indexed denoise step that just completed (or, for heartbeat
/// updates, the step currently in flight).</param>
/// <param name="TotalSteps">The total number of denoise steps configured for this generation.</param>
/// <param name="ElapsedMs">Wall-clock milliseconds since this report's reference point. For
/// final-of-step reports this is the duration of the just-completed step; for heartbeat
/// reports this is elapsed time within the current step so far.</param>
public readonly record struct GenerationProgress(int Step, int TotalSteps, double ElapsedMs)
{
    /// <summary>Pipeline-computed overall progress in <c>[0, 1]</c> across all phases
    /// (text encoding + denoise + VAE decode), including phase weights. When non-negative,
    /// consumers should render this directly instead of dividing <see cref="Step"/> /
    /// <see cref="TotalSteps"/>. Default <c>-1.0</c> = "not provided, fall back to step count".</summary>
    public double OverallPercent { get; init; } = -1.0;

    /// <summary>True when this report is a mid-step heartbeat rather than an end-of-step
    /// completion. Useful for consumers that want to log only completions but render
    /// every heartbeat in a UI progress bar.</summary>
    public bool IsHeartbeat { get; init; } = false;

    /// <summary>Optional snapshot of the in-flight diffusion latent (shape <c>[1, C, H, W]</c>,
    /// F32). Pipelines that want to enable progress-image previews populate this on each
    /// step-completion report; pipelines that don't leave it null. The tensor is borrowed —
    /// the pipeline still owns it and may dispose immediately after the callback returns,
    /// so consumers must finish reading (or copy) before yielding control.
    /// Pair with <see cref="LatentArch"/> to pick the right preview decoder factors.</summary>
    public Tensor? Latent { get; init; } = null;

    /// <summary>The model family <see cref="Latent"/> came from. Determines which factor
    /// matrix <see cref="LatentPreview"/> uses (and which TAESD weight set is appropriate).
    /// Default <see cref="LatentArchitecture.Unknown"/> means "skip preview encoding".</summary>
    public LatentArchitecture LatentArch { get; init; } = LatentArchitecture.Unknown;
}
