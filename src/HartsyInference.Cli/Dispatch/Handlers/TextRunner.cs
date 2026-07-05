using HartsyInference.Cli.Infra;
using HartsyInference.LLM.Generation;
using HartsyInference.Tokenizers;

namespace HartsyInference.Cli.Dispatch.Handlers;

/// <summary>A loaded LLM: the generation pipeline plus the resources whose lifetime must outlive it (the GGUF model or
/// the safetensors loader that owns the mmap-backed weights).</summary>
public sealed class TextRunner : IModalityRunner
{
    private readonly IDisposable? _ownedModel;
    private readonly IDisposable? _ownedLoader;

    /// <summary>Creates a runner over <paramref name="pipeline"/>, taking ownership of the backing resources.</summary>
    public TextRunner(string modelId, TextGenerationPipeline pipeline, ILlmTokenizer tokenizer,
        IDisposable? ownedModel, IDisposable? ownedLoader)
    {
        ModelId = modelId;
        Pipeline = pipeline;
        Tokenizer = tokenizer;
        _ownedModel = ownedModel;
        _ownedLoader = ownedLoader;
    }

    /// <inheritdoc/>
    public Modality Modality => Modality.Text;

    /// <inheritdoc/>
    public string ModelId { get; }

    /// <summary>The text-generation pipeline to drive.</summary>
    public TextGenerationPipeline Pipeline { get; }

    /// <summary>The tokenizer, used for incremental streaming decode.</summary>
    public ILlmTokenizer Tokenizer { get; }

    /// <inheritdoc/>
    public void Dispose()
    {
        _ownedModel?.Dispose();
        _ownedLoader?.Dispose();
    }
}
