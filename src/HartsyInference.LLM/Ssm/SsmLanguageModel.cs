using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Models.Lfm3;
using HartsyInference.LLM.Models.Spark;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Ssm;

/// <summary>Loads a GGUF recurrent (SSM) decoder — mamba/mamba2/rwkv6/rwkv7 — into a ready-to-run <see cref="ISsmModel"/> plus the tokenizer + chat template <see cref="GgufLanguageModel"/> builds for the transformer family; these architectures have no <c>attention.head_count</c> notion (llama.cpp GGUFs store it as 0), so they must never go through <see cref="GgufConfigFactory"/>/<see cref="GenericTransformer"/>, which derives head_dim = hidden/heads and divides by zero. Use <see cref="IsSsmArchitecture"/> to route instead of <see cref="GgufLanguageModel.Load"/>.</summary>
public sealed class SsmLanguageModel : IDisposable
{
    private int _disposed;

    public ISsmModel Model { get; }
    public string Architecture { get; }
    public ILlmTokenizer Tokenizer { get; }
    public IChatTemplate Template { get; }

    private SsmLanguageModel(ISsmModel model, string architecture, ILlmTokenizer tokenizer, IChatTemplate template)
    {
        Model = model; Architecture = architecture; Tokenizer = tokenizer; Template = template;
    }

    /// <summary>True for the <c>general.architecture</c> values this loader (not <see cref="GgufLanguageModel"/>) handles.</summary>
    public static bool IsSsmArchitecture(string architecture) => GgufLanguageModel.SsmArchitectures.Contains(architecture);

    /// <summary>Loads the GGUF at <paramref name="path"/>, whose <c>general.architecture</c> is <paramref name="architecture"/> (caller already knows this from a cheap metadata peek — see <see cref="IsSsmArchitecture"/>).</summary>
    public static SsmLanguageModel Load(string path, string architecture)
    {
        ISsmModel model = architecture switch
        {
            "mamba2" => Mamba2Model.Load(path),
            "mamba" => MambaModel.Load(path),
            "rwkv7" => Rwkv7Model.Load(path),
            "rwkv6" or "rwkv" => RwkvModel.Load(path),
            "qwen35" or "qwen35moe" => Qwen35Model.Load(path),
            _ => throw new NotSupportedException($"'{architecture}' is not a supported SSM architecture."),
        };
        try
        {
            ILlmTokenizer tokenizer = GgufLanguageModel.BuildTokenizer(model.Metadata);
            IChatTemplate template = GgufLanguageModel.BuildTemplate(model.Metadata, tokenizer);
            return new SsmLanguageModel(model, architecture, tokenizer, template);
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }

    /// <summary>True for the Hugging Face <c>model_type</c> values <see cref="LoadHuggingFace"/> loads.</summary>
    public static bool IsHuggingFaceFamily(string modelType) => modelType is Lfm3Config.ModelType or SparkConfig.ModelType;

    /// <summary>Loads a Hugging Face directory of a family that carries its own state: the decoder plus a tokenizer from <c>tokenizer.json</c> and the template from <c>chat_template.jinja</c>.</summary>
    /// <param name="maxSequenceLength">Longest sequence (prompt plus generation) the model accepts.</param>
    /// <exception cref="NotSupportedException"><paramref name="modelType"/> is not one <see cref="IsHuggingFaceFamily"/> accepts.</exception>
    public static SsmLanguageModel LoadHuggingFace(string directory, string modelType, int maxSequenceLength)
    {
        (ISsmModel model, string bos, string eos, int bosId, int eosId) = modelType switch
        {
            Lfm3Config.ModelType => LoadLfm3(directory, maxSequenceLength),
            SparkConfig.ModelType => LoadSpark(directory, maxSequenceLength),
            _ => throw new NotSupportedException($"'{modelType}' is not a Hugging Face family this loader handles."),
        };
        try
        {
            using FileStream tokenizerJson = File.OpenRead(Path.Combine(directory, "tokenizer.json"));
            ILlmTokenizer tokenizer = HfTokenizerJson.LoadByteLevelBpe(tokenizerJson, [eosId], bos, eos);
            if (tokenizer.BosId != bosId || tokenizer.EosId != eosId)
            {
                throw new InvalidDataException(
                    $"tokenizer.json maps {bos} to {tokenizer.BosId} and {eos} to {tokenizer.EosId}; config.json says {bosId} and {eosId}.");
            }
            IChatTemplate template = new JinjaChatTemplate(File.ReadAllText(Path.Combine(directory, "chat_template.jinja")));
            return new SsmLanguageModel(model, modelType, tokenizer, template);
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }

    private static (ISsmModel, string, string, int, int) LoadLfm3(string directory, int maxSequenceLength)
    {
        Lfm3Model model = Lfm3Model.Load(directory, maxSequenceLength);
        return (model, "<|startoftext|>", "<|im_end|>", model.Config.BosTokenId, model.Config.EosTokenId);
    }

    private static (ISsmModel, string, string, int, int) LoadSpark(string directory, int maxSequenceLength)
    {
        SparkModel model = SparkModel.Load(directory, maxSequenceLength);
        return (model, "<｜start▁of▁sentence｜>", "<｜end▁of▁sentence｜>", model.Config.BosTokenId, model.Config.EosTokenId);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        Model.Dispose();
    }
}
