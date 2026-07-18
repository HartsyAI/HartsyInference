using HartsyInference.Engine;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Ssm;
using HartsyInference.Tokenizers;

namespace HartsyInference.Engine.Dispatch.Handlers;

/// <summary>A loaded LLM: the generation pipeline plus the resources whose lifetime must outlive it (the GGUF model or
/// the safetensors loader that owns the mmap-backed weights). Exactly one of <see cref="Pipeline"/> /
/// <see cref="SsmPipeline"/> is set — transformer archs use the former, recurrent/hybrid archs
/// (mamba/rwkv/qwen3.5) the latter (see <see cref="SsmLanguageModel.IsSsmArchitecture"/>).</summary>
public sealed class TextRunner : IModalityRunner
{
    private readonly IDisposable? _ownedModel;
    private readonly IDisposable? _ownedLoader;

    /// <summary>Creates a runner over a transformer <paramref name="pipeline"/>, taking ownership of the backing resources.</summary>
    public TextRunner(string modelId, TextGenerationPipeline pipeline, ILlmTokenizer tokenizer,
        IDisposable? ownedModel, IDisposable? ownedLoader)
    {
        ModelId = modelId;
        Pipeline = pipeline;
        Tokenizer = tokenizer;
        _ownedModel = ownedModel;
        _ownedLoader = ownedLoader;
    }

    /// <summary>Creates a runner over a recurrent/hybrid <paramref name="ssmPipeline"/> (mamba/rwkv/qwen3.5).</summary>
    public TextRunner(string modelId, SsmGenerationPipeline ssmPipeline, ILlmTokenizer tokenizer, IDisposable ownedModel)
    {
        ModelId = modelId;
        SsmPipeline = ssmPipeline;
        Tokenizer = tokenizer;
        _ownedModel = ownedModel;
    }

    /// <inheritdoc/>
    public Modality Modality => Modality.Text;

    /// <inheritdoc/>
    public string ModelId { get; }

    /// <summary>The text-generation pipeline to drive (transformer archs). Null for SSM/hybrid archs — see
    /// <see cref="SsmPipeline"/> instead.</summary>
    public TextGenerationPipeline? Pipeline { get; }

    /// <summary>The recurrent-model pipeline to drive (mamba/rwkv/qwen3.5). Null for transformer archs.</summary>
    public SsmGenerationPipeline? SsmPipeline { get; }

    /// <summary>The tokenizer, used for incremental streaming decode.</summary>
    public ILlmTokenizer Tokenizer { get; }

    /// <inheritdoc/>
    public void Dispose()
    {
        _ownedModel?.Dispose();
        _ownedLoader?.Dispose();
    }
}
