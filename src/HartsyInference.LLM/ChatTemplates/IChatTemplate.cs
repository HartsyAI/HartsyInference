using System.Collections.Generic;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Renders a conversation into model-ready token ids; implementations emit special/control tokens directly by id rather than BPE'ing their literal text.</summary>
public interface IChatTemplate
{
    /// <summary>Registry key for this template (for example "chatml").</summary>
    string Name { get; }

    /// <summary>Encodes <paramref name="messages"/> to ids, appending a trailing assistant header when <paramref name="addGenerationPrompt"/> is true; <paramref name="enableThinking"/> sets the Qwen3-family <c>enable_thinking</c> toggle, or falls back to the template's default when null.</summary>
    int[] Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt, bool? enableThinking = null);
}
