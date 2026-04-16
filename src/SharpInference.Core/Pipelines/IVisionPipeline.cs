using SharpInference.Core.Tensors;

namespace SharpInference.Core.Pipelines;

/// <summary>Interface for vision pipelines (CLIP embeddings, YOLO detection, SAM segmentation).</summary>
public interface IVisionPipeline : IDisposable
{
    /// <summary>Name of the model loaded in this pipeline.</summary>
    string ModelName { get; }
}

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

/// <summary>Object detection pipeline interface.</summary>
public interface IDetectionPipeline : IVisionPipeline
{
    /// <summary>Detects objects in the given image.</summary>
    Task<IReadOnlyList<DetectionResult>> DetectAsync(ReadOnlyMemory<byte> imageBytes, float confidenceThreshold = 0.5f, CancellationToken cancellationToken = default);
}

/// <summary>A single detected object with bounding box and classification.</summary>
public sealed class DetectionResult
{
    /// <summary>Bounding box: X coordinate of top-left corner (normalized 0-1).</summary>
    public float X { get; init; }

    /// <summary>Bounding box: Y coordinate of top-left corner (normalized 0-1).</summary>
    public float Y { get; init; }

    /// <summary>Bounding box width (normalized 0-1).</summary>
    public float Width { get; init; }

    /// <summary>Bounding box height (normalized 0-1).</summary>
    public float Height { get; init; }

    /// <summary>Detection confidence score (0-1).</summary>
    public float Confidence { get; init; }

    /// <summary>Detected class label.</summary>
    public required string Label { get; init; }

    /// <summary>Class index in the model's label set.</summary>
    public int ClassIndex { get; init; }
}
