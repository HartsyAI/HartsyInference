using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Requests;

/// <summary>Progress report during image generation, emitted after each denoise step and (when a heartbeat is enabled) also at fixed wall-clock intervals during long steps so consumers can render smooth progress bars instead of stair-stepping by step count.</summary>
/// <param name="Step">The 0-indexed denoise step that just completed, or (for heartbeat updates) the step currently in flight.</param>
/// <param name="ElapsedMs">For end-of-step reports, the duration of the just-completed step; for heartbeat reports, elapsed time within the current step so far.</param>
public readonly record struct GenerationProgress(int Step, int TotalSteps, double ElapsedMs)
{
    /// <summary>Pipeline-computed overall progress in <c>[0, 1]</c> across all phases (text encoding + denoise + VAE decode), including phase weights; <c>-1.0</c> means "not provided, fall back to step count".</summary>
    public double OverallPercent { get; init; } = -1.0;

    /// <summary>True when this report is a mid-step heartbeat rather than an end-of-step completion.</summary>
    public bool IsHeartbeat { get; init; } = false;

    /// <summary>Optional snapshot of the in-flight diffusion latent (shape <c>[1, C, H, W]</c>, F32); borrowed — the pipeline may dispose it right after the callback returns, so consumers must finish reading (or copy) before yielding control.</summary>
    public Tensor? Latent { get; init; } = null;

    /// <summary>The model family <see cref="Latent"/> came from, selecting which <see cref="LatentPreview"/> factor matrix / TAESD weight set applies; <see cref="LatentArchitecture.Unknown"/> skips preview encoding.</summary>
    public LatentArchitecture LatentArch { get; init; } = LatentArchitecture.Unknown;
}
