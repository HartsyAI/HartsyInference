using System;
using System.Collections.Generic;
using System.Text;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Least-common-denominator fallback for tokenizers with no usable chat template and no ChatML control tokens (base/completion models like GPT-2, StarCoder2); concatenates message contents as plain text via <see cref="ILlmTokenizer.EncodeOrdinary"/> so a bare <see cref="Generation.GenerationRequest.Prompt"/> never throws — not a substitute for a model's real turn format when one exists (e.g. LLaVA's Vicuna-v1 template lives in <c>MultimodalGenerator</c>).</summary>
public sealed class RawCompletionTemplate : IChatTemplate
{
    /// <inheritdoc/>
    public string Name => "raw";

    /// <inheritdoc/>
    /// <remarks>No chat structure to speak of — <paramref name="addGenerationPrompt"/> and <paramref name="enableThinking"/> are accepted for interface parity but ignored.</remarks>
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
        return tokenizer.EncodeOrdinary(sb.ToString());
    }
}
