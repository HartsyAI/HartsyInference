using HartsyInference.LLM.ChatTemplates;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Generation;

/// <summary>A checkpoint served by a host reference model on the CPU: its generation model, tokenizer and chat template, loaded together
/// and disposed together.</summary>
public interface IHostReferenceTextModel : IDisposable
{
    /// <summary>The model the generation pipeline drives.</summary>
    IGenerationModel Generation { get; }

    /// <summary>The checkpoint's tokenizer, with begin and end of sentence set.</summary>
    ILlmTokenizer Tokenizer { get; }

    /// <summary>The checkpoint's chat template.</summary>
    IChatTemplate Template { get; }
}
