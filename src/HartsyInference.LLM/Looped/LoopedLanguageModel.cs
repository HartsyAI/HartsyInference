using HartsyInference.Core.Tensors;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Looped;

/// <summary>Loads a GGUF looped (latent-recurrent) decoder checkpoint, today llama.cpp's <c>dory</c>, into a ready-to-run
/// <see cref="LoopedTransformer"/> with the tokenizer and chat template the file carries; owns the GGUF handle (projection weights are
/// memory-mapped and consumed lazily), so this object must outlive any inference using <see cref="Transformer"/>.</summary>
/// <remarks>Pipeline: <see cref="GgufModelLoader.Load"/> (parse + key remap) → <see cref="GgufLanguageModel.RelabelAndWidenWeights"/> (the stage every GGUF
/// decoder shares) → <see cref="LoopedConfigFactory.FromGguf"/> → <see cref="LoopedTransformer.LoadWeights"/>.</remarks>
public sealed class LoopedLanguageModel : IDisposable
{
    private readonly GgufModelLoader.LoadedGgufModel _handle;
    private int _disposed;

    /// <summary>The architecture config read from the GGUF metadata and weights.</summary>
    public LoopedTransformerConfig Config { get; }

    /// <summary>The ready-to-run transformer (weights loaded).</summary>
    public LoopedTransformer Transformer { get; }

    /// <summary>The GGUF <c>general.architecture</c> string (<c>dory</c>).</summary>
    public string Architecture => _handle.Architecture;

    /// <summary>The tokenizer built from the GGUF's embedded vocabulary.</summary>
    public ILlmTokenizer Tokenizer { get; }

    /// <summary>The chat template: the model's own Jinja <c>chat_template</c> when present, else ChatML.</summary>
    public IChatTemplate Template { get; }

    private LoopedLanguageModel(GgufModelLoader.LoadedGgufModel handle, LoopedTransformerConfig config, LoopedTransformer transformer,
        ILlmTokenizer tokenizer, IChatTemplate template)
    {
        _handle = handle;
        Config = config;
        Transformer = transformer;
        Tokenizer = tokenizer;
        Template = template;
    }

    /// <summary>Whether <paramref name="architecture"/> (GGUF <c>general.architecture</c>) loads through this class rather than <see cref="GgufLanguageModel"/>.</summary>
    public static bool IsLoopedArchitecture(string architecture) => LoopedConfigFactory.Architectures.Contains(architecture);

    /// <summary>Loads the GGUF at <paramref name="path"/>; the dtype options mean what they mean for <see cref="GgufLanguageModel.Load"/>.</summary>
    public static LoopedLanguageModel Load(string path, bool lowVramQuant = false, bool dequantizeToF32 = false, IReadOnlySet<DType>? keepQuantized = null)
    {
        GgufModelLoader.LoadedGgufModel handle = GgufModelLoader.Load(path);
        try
        {
            if (!IsLoopedArchitecture(handle.Architecture))
                throw new NotSupportedException($"'{handle.Architecture}' is not a looped architecture; load it via GgufLanguageModel.");
            Dictionary<string, Tensor> weights = GgufLanguageModel.RelabelAndWidenWeights(handle, dequantizeToF32, keepQuantized);
            LoopedTransformerConfig config = LoopedConfigFactory.FromGguf(handle.Metadata, weights, lowVramQuant);
            LoopedTransformer transformer = new(config);
            try
            {
                transformer.LoadWeights(weights, "model");
            }
            catch
            {
                transformer.Dispose();
                throw;
            }
            ILlmTokenizer tokenizer = GgufLanguageModel.BuildTokenizer(handle.Metadata);
            IChatTemplate template = GgufLanguageModel.BuildTemplate(handle.Metadata, tokenizer);
            return new LoopedLanguageModel(handle, config, transformer, tokenizer, template);
        }
        catch
        {
            handle.Dispose();
            throw;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        Transformer.Dispose();
        _handle.Dispose();
    }
}
