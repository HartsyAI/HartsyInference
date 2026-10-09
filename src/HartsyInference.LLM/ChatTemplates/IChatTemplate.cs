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

    /// <summary>Encodes like the four-argument overload and also offers <paramref name="tools"/> to the model; templates without a tool slot ignore them (this default). Null or empty renders exactly as the tool-less overload.</summary>
    int[] Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt, bool? enableThinking,
        IReadOnlyList<ToolSpec>? tools)
        => Encode(tokenizer, messages, addGenerationPrompt, enableThinking);

    /// <summary>Encodes like the overload above and also takes a <paramref name="reasoningEffort"/> in [1, 100] for templates that have one (DeepSeek-V4.1 thinking mode); null uses the template's default. Templates without an effort slot ignore it (this default).</summary>
    int[] Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt, bool? enableThinking,
        IReadOnlyList<ToolSpec>? tools, int? reasoningEffort)
        => Encode(tokenizer, messages, addGenerationPrompt, enableThinking, tools);
}
