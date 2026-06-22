using System.Collections.Generic;
using HartsyInference.Tokenizers;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Renders a conversation into model-ready token ids. Implementations emit token ids directly because ChatML special tokens (<c>&lt;|im_start|&gt;</c>, <c>&lt;|im_end|&gt;</c>) do not round-trip through plain BPE encoding of literal text.</summary>
public interface IChatTemplate
{
    /// <summary>Registry key for this template (for example "chatml").</summary>
    string Name { get; }

    /// <summary>Encodes <paramref name="messages"/> to token ids; when <paramref name="addGenerationPrompt"/> is true, appends a trailing assistant header so the model continues as the assistant.</summary>
    int[] Encode(Qwen2Tokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt);
}
