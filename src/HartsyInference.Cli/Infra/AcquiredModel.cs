namespace HartsyInference.Cli.Infra;

/// <summary>The spec to run and, after the user declined its downloads, the refusal that holds until this is disposed.</summary>
public sealed class AcquiredModel(ModelSpec spec, IDisposable? refusal) : IDisposable
{
    /// <summary>The spec, pointed at the files on disk when they are there.</summary>
    public ModelSpec Spec { get; } = spec;

    /// <inheritdoc/>
    public void Dispose() => refusal?.Dispose();
}
