using HartsyInference.Cli.Infra;
using HartsyInference.Core.Backends;
using HartsyInference.Vision.Detection;
using HartsyInference.Vision.Embeddings;

namespace HartsyInference.Cli.Dispatch.Handlers;

/// <summary>A loaded vision model for one task (embed or detect), plus the resource that owns its weights.</summary>
public sealed class VisionRunner : IModalityRunner
{
    private readonly IDisposable? _owned;

    /// <summary>Creates an embedding runner (CLIP).</summary>
    public VisionRunner(string modelId, ImageEmbeddingPipeline embed, IBackend backend, IDisposable? owned)
    {
        ModelId = modelId;
        Task = VisionTask.Embed;
        Embed = embed;
        Backend = backend;
        _owned = owned;
    }

    /// <summary>Creates a detection runner (YOLO).</summary>
    public VisionRunner(string modelId, YoloPipeline detect, IBackend backend)
    {
        ModelId = modelId;
        Task = VisionTask.Detect;
        Detect = detect;
        Backend = backend;
        _owned = detect;
    }

    /// <inheritdoc/>
    public Modality Modality => Modality.Vision;

    /// <inheritdoc/>
    public string ModelId { get; }

    /// <summary>Which task this runner performs.</summary>
    public VisionTask Task { get; }

    /// <summary>The embedding pipeline (non-null when <see cref="Task"/> is <see cref="VisionTask.Embed"/>).</summary>
    public ImageEmbeddingPipeline? Embed { get; }

    /// <summary>The detection pipeline (non-null when <see cref="Task"/> is <see cref="VisionTask.Detect"/>).</summary>
    public YoloPipeline? Detect { get; }

    /// <summary>The borrowed compute backend.</summary>
    public IBackend Backend { get; }

    /// <inheritdoc/>
    public void Dispose() => _owned?.Dispose();
}
