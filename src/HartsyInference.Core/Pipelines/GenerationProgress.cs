namespace HartsyInference.Core.Pipelines;

/// <summary>Progress update emitted at each denoising step during image generation.</summary>
public readonly record struct GenerationProgress(int Step, int TotalSteps, double ElapsedMs)
{
    /// <summary>Whether generation is complete.</summary>
    public bool IsComplete => Step >= TotalSteps;
}
