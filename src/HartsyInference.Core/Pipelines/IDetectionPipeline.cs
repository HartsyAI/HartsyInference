namespace HartsyInference.Core.Pipelines;

/// <summary>Object detection pipeline interface.</summary>
public interface IDetectionPipeline : IVisionPipeline
{
    /// <summary>Detects objects in the given image.</summary>
    Task<IReadOnlyList<DetectionResult>> DetectAsync(ReadOnlyMemory<byte> imageBytes, float confidenceThreshold = 0.5f, CancellationToken cancellationToken = default);
}
