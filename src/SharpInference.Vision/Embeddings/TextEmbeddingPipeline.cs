using SharpInference.Core.Backends;
using SharpInference.Core.Pipelines;
using SharpInference.Vision.Clip;

namespace SharpInference.Vision.Embeddings;

/// <summary>High-level façade for CLIP text embedding. Wraps <see cref="ClipTextEmbedder"/> with
/// a stable name + lifetime so it can be registered in the Server / SwarmUI DI containers
/// alongside <see cref="ImageEmbeddingPipeline"/>.</summary>
public sealed class TextEmbeddingPipeline : IVisionPipeline
{
    private readonly IBackend _backend;
    private readonly ClipTextEmbedder _embedder;
    private int _disposed;

    /// <summary>Preset name.</summary>
    public string ModelName { get; }

    /// <summary>Output embedding dimensionality.</summary>
    public int EmbeddingDim => _embedder.EmbeddingDim;

    /// <summary>Constructs the pipeline. The embedder is not owned by this class.</summary>
    public TextEmbeddingPipeline(IBackend backend, ClipTextEmbedder embedder, string modelName)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(embedder);
        ArgumentNullException.ThrowIfNull(modelName);
        _backend = backend;
        _embedder = embedder;
        ModelName = modelName;
    }

    /// <summary>Encodes a single string into an L2-normalized embedding.</summary>
    public TextEmbedding Embed(string text)
    {
        ThrowIfDisposed();
        return _embedder.Encode(_backend, text);
    }

    /// <summary>Encodes a batch of strings in one transformer pass.</summary>
    public IReadOnlyList<TextEmbedding> EmbedBatch(IReadOnlyList<string> texts)
    {
        ThrowIfDisposed();
        return _embedder.EncodeBatch(_backend, texts);
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    /// <summary>Releases pipeline-internal state. The embedder is owned by the loader.</summary>
    public void Dispose()
    {
        Interlocked.Exchange(ref _disposed, 1);
    }
}
