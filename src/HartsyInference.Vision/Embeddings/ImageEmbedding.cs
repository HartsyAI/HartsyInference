using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Embeddings;

/// <summary>A single CLIP image embedding — the projected, L2-normalized feature vector ready for cosine-similarity scoring. Owns its underlying <see cref="Tensor"/> and must be disposed; otherwise the native memory leaks.
/// <para>Shape is <c>[1, embeddingDim]</c>. Two-dim rather than one-dim so the same tensor can be batched alongside text embeddings without a reshape.</para></summary>
public sealed class ImageEmbedding : IDisposable
{
    private Tensor? _vector;
    private int _disposed;

    /// <summary>Number of dimensions in the embedding vector (e.g. 768 for CLIP-L, 1280 for CLIP-bigG).</summary>
    public int EmbeddingDim { get; }

    /// <summary>Optional source identifier (file path, URL, or caller-defined tag). Useful when building search indices but not load-bearing.</summary>
    public string? SourceId { get; }

    /// <summary>The underlying tensor (<c>[1, embeddingDim]</c>, F32). L2-normalized — direct dot product with another normalized embedding yields cosine similarity in <c>[-1, 1]</c>.</summary>
    public Tensor Vector
    {
        get
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            return _vector!;
        }
    }

    /// <summary>Wraps a tensor as an embedding. Takes ownership of the tensor — disposing the embedding disposes the tensor.</summary>
    public ImageEmbedding(Tensor vector, string? sourceId = null)
    {
        ArgumentNullException.ThrowIfNull(vector);
        if (vector.Shape.Rank != 2 || vector.Shape[0] != 1)
            throw new ArgumentException($"Embedding tensor must have shape [1, dim]; got {vector.Shape}.", nameof(vector));
        if (vector.DType != DType.F32)
            throw new ArgumentException($"Embedding tensor must be F32; got {vector.DType}.", nameof(vector));

        _vector = vector;
        EmbeddingDim = (int)vector.Shape[1];
        SourceId = sourceId;
    }

    /// <summary>Returns the embedding as a read-only span of F32 — useful for serializing into a search index or computing similarities without going through the tensor API.</summary>
    public ReadOnlySpan<float> AsSpan() => Vector.AsReadOnlySpan<float>();

    /// <summary>Frees the underlying tensor. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _vector?.Dispose();
        _vector = null;
    }
}
