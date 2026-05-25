using SharpInference.Core.Tensors;

namespace SharpInference.Vision.Embeddings;

/// <summary>A single CLIP text embedding — the projected, L2-normalized feature vector for the EOS-token hidden state. Same shape contract as <see cref="ImageEmbedding"/> so they're interchangeable in similarity scoring.</summary>
public sealed class TextEmbedding : IDisposable
{
    private Tensor? _vector;
    private int _disposed;

    /// <summary>Number of dimensions in the embedding vector.</summary>
    public int EmbeddingDim { get; }

    /// <summary>The original text this embedding represents — useful for de-duplication and debugging, kept by the embedder rather than the caller.</summary>
    public string SourceText { get; }

    /// <summary>The underlying tensor (<c>[1, embeddingDim]</c>, F32). L2-normalized.</summary>
    public Tensor Vector
    {
        get
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            return _vector!;
        }
    }

    /// <summary>Wraps a tensor as an embedding. Takes ownership.</summary>
    public TextEmbedding(Tensor vector, string sourceText)
    {
        ArgumentNullException.ThrowIfNull(vector);
        ArgumentNullException.ThrowIfNull(sourceText);
        if (vector.Shape.Rank != 2 || vector.Shape[0] != 1)
            throw new ArgumentException($"Embedding tensor must have shape [1, dim]; got {vector.Shape}.", nameof(vector));
        if (vector.DType != DType.F32)
            throw new ArgumentException($"Embedding tensor must be F32; got {vector.DType}.", nameof(vector));

        _vector = vector;
        EmbeddingDim = (int)vector.Shape[1];
        SourceText = sourceText;
    }

    /// <summary>Returns the embedding as a read-only span of F32.</summary>
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
