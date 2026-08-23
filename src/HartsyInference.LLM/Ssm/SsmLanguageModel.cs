using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
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

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        Model.Dispose();
    }
}
