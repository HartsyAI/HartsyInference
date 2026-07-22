using System.Collections.Generic;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Renders a conversation into model-ready token ids using the model's tokenizer. Works with any
/// <see cref="ILlmTokenizer"/> so non-Qwen models (Llama, Mistral, …) prompt correctly. Implementations emit
/// ids directly (special/control tokens map to their ids, not BPE'd literal text).</summary>
public interface IChatTemplate
{
    /// <summary>Registry key for this template (for example "chatml").</summary>
    string Name { get; }

    /// <summary>Encodes <paramref name="tokenizer"/>-tokenized <paramref name="messages"/> to ids; when
    /// <paramref name="addGenerationPrompt"/> is true, appends a trailing assistant header.
    /// <paramref name="enableThinking"/> sets the template's <c>enable_thinking</c> variable (Qwen3-family
    /// reasoning toggle); null leaves it undefined so the template falls back to its own default.</summary>
    int[] Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt, bool? enableThinking = null);
}
