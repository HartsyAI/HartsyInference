using SharpInference.Core.Tensors;

namespace SharpInference.Core.Pipelines;

/// <summary>Embedding pipeline interface for computing vector representations of images or text.</summary>
public interface IEmbeddingPipeline : IVisionPipeline
{
    /// <summary>Dimensionality of the output embedding vectors.</summary>
    int EmbeddingDimension { get; }

    /// <summary>Computes an embedding vector for the given image bytes.</summary>
    Task<Tensor> EmbedImageAsync(ReadOnlyMemory<byte> imageBytes, CancellationToken cancellationToken = default);

    /// <summary>Computes an embedding vector for the given text.</summary>
    Task<Tensor> EmbedTextAsync(string text, CancellationToken cancellationToken = default);
}
