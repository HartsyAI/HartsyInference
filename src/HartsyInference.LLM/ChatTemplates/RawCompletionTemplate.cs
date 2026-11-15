using System;
using System.Collections.Generic;
using System.Text;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Least-common-denominator fallback for tokenizers with no usable chat template and no ChatML control tokens (base/completion models like GPT-2, StarCoder2); concatenates message contents as plain text via <see cref="ILlmTokenizer.EncodeOrdinary"/> so a bare <see cref="Generation.GenerationRequest.Prompt"/> never throws — not a substitute for a model's real turn format when one exists (e.g. LLaVA's Vicuna-v1 template lives in <c>MultimodalGenerator</c>).</summary>
/// <remarks>The text may be wrapped in fixed special ids, such as the ones a Hugging Face <c>tokenizer.json</c> post-processor adds to a raw prompt
/// (<see cref="HfTokenizerJson.ReadSingleSequenceSpecials"/>), so the prompt matches what a serving <c>/v1/completions</c> endpoint feeds the model.
/// One deliberate difference from Hugging Face <c>tokenizer(prompt)</c>: special-token text inside a message (such as <c>&lt;|ERNIE_MASK|&gt;</c>)
/// is encoded as ordinary bytes, not as the special id, so user text cannot inject control tokens.</remarks>
public sealed class RawCompletionTemplate : IChatTemplate
{
    private readonly int[] _prefix;
    private readonly int[] _suffix;

    /// <summary>A template that adds no special ids.</summary>
    public RawCompletionTemplate() : this([], [])
    {
    }

    /// <summary>A template that puts <paramref name="prefixIds"/> before the text and <paramref name="suffixIds"/> after it.</summary>
    public RawCompletionTemplate(IReadOnlyList<int> prefixIds, IReadOnlyList<int> suffixIds)
    {
        ArgumentNullException.ThrowIfNull(prefixIds);
        ArgumentNullException.ThrowIfNull(suffixIds);
        _prefix = [.. prefixIds];
        _suffix = [.. suffixIds];
    }

    /// <inheritdoc/>
    public string Name => "raw";

    /// <inheritdoc/>
    /// <remarks>No chat structure to speak of — <paramref name="addGenerationPrompt"/> and <paramref name="enableThinking"/> are accepted for interface parity but ignored. Messages, the system prompt included, are joined with a blank line.</remarks>
    public int[] Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt, bool? enableThinking = null)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(messages);

        StringBuilder sb = new();
        for (int i = 0; i < messages.Count; i++)
        {
            if (i > 0) sb.Append("\n\n");
            sb.Append(messages[i].Content);
        }
        int[] text = tokenizer.EncodeOrdinary(sb.ToString());
        if (_prefix.Length == 0 && _suffix.Length == 0) return text;
        return [.. _prefix, .. text, .. _suffix];
    }
}
