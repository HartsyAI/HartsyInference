namespace SharpInference.Core.Pipelines;

/// <summary>Interface for vision pipelines (CLIP embeddings, YOLO detection, SAM segmentation).</summary>
public interface IVisionPipeline : IDisposable
{
    /// <summary>Name of the model loaded in this pipeline.</summary>
    string ModelName { get; }
}
