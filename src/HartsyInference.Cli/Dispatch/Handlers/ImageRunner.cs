using HartsyInference.Cli.Infra;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Tokenizers;

namespace HartsyInference.Cli.Dispatch.Handlers;

/// <summary>A loaded diffusion image pipeline plus the CLIP tokenizer used to encode prompts.</summary>
public sealed class ImageRunner : IModalityRunner
{
    /// <summary>Creates a runner over <paramref name="pipeline"/>, taking ownership of it.</summary>
    public ImageRunner(string modelId, SdxlPipeline pipeline, ClipTokenizer tokenizer)
    {
        ModelId = modelId;
        Pipeline = pipeline;
        Tokenizer = tokenizer;
    }

    /// <inheritdoc/>
    public Modality Modality => Modality.Image;

    /// <inheritdoc/>
    public string ModelId { get; }

    /// <summary>The SDXL pipeline to drive.</summary>
    public SdxlPipeline Pipeline { get; }

    /// <summary>The CLIP tokenizer (embedded vocab) for prompt encoding.</summary>
    public ClipTokenizer Tokenizer { get; }

    /// <inheritdoc/>
    public void Dispose() => Pipeline.Dispose();
}
