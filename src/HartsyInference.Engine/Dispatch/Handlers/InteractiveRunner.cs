using HartsyInference.Engine;
using HartsyInference.Interactive.Pipelines;

namespace HartsyInference.Engine.Dispatch.Handlers;

/// <summary>A loaded Oasis world model (offline rollout) plus the loaders that own its mmap-backed weights.</summary>
public sealed class InteractiveRunner : IModalityRunner
{
    private readonly IReadOnlyList<IDisposable> _owned;

    /// <summary>Creates a runner over <paramref name="pipeline"/>.</summary>
    public InteractiveRunner(string modelId, OasisPipeline pipeline, IReadOnlyList<IDisposable> owned)
    {
        ModelId = modelId;
        Pipeline = pipeline;
        _owned = owned;
    }

    /// <inheritdoc/>
    public Modality Modality => Modality.Interactive;

    /// <inheritdoc/>
    public string ModelId { get; }

    /// <summary>The Oasis pipeline.</summary>
    public OasisPipeline Pipeline { get; }

    /// <inheritdoc/>
    public void Dispose()
    {
        Pipeline.Dispose();
        foreach (IDisposable d in _owned)
            d.Dispose();
    }
}
