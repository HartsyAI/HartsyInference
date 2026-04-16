namespace SharpInference.Core.Pipelines;

/// <summary>Base interface for all pipeline request types. Carries common parameters shared across generation requests (cancellation, seed, progress).</summary>
public interface IPipelineRequest
{
    /// <summary>Random seed for reproducibility. Use -1 for random.</summary>
    long Seed { get; }

    /// <summary>Cancellation token to abort generation mid-run.</summary>
    CancellationToken CancellationToken { get; }
}
