namespace SharpInference.Core.Pipelines;

/// <summary>Interface for image generation pipelines (SD1.5, SDXL, Flux, etc.).</summary>
public interface IDiffusionPipeline : IDisposable
{
    /// <summary>Name of the model loaded in this pipeline.</summary>
    string ModelName { get; }

    /// <summary>Generates images from a text prompt, yielding progress updates at each denoising step.</summary>
    IAsyncEnumerable<GenerationProgress> GenerateAsync(IPipelineRequest request);
}
