using System;
using System.Collections.Generic;
using System.Text;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Fallback for tokenizers that carry no usable chat template AND lack the ChatML control tokens
/// (<c>&lt;|im_start|&gt;</c>/<c>&lt;|im_end|&gt;</c>) — base/completion models (GPT-2, StarCoder2) and models
/// whose real turn format isn't ChatML (Vicuna-family VLMs like LLaVA use "USER: … ASSISTANT:"). Concatenates
/// message contents as plain text via <see cref="ILlmTokenizer.EncodeOrdinary"/>, no special tokens — the
/// same shape a raw single-turn completion prompt takes. Not a substitute for a model's real turn format when
/// one exists (e.g. LLaVA's Vicuna-v1 template lives in <c>MultimodalGenerator</c>); this is the safe
/// least-common-denominator so a bare <see cref="Generation.GenerationRequest.Prompt"/> never throws.</summary>
public sealed class RawCompletionTemplate : IChatTemplate
{
    /// <inheritdoc/>
    public string Name => "raw";

    /// <inheritdoc/>
    /// <remarks>No chat structure to speak of — <paramref name="addGenerationPrompt"/> and
    /// <paramref name="enableThinking"/> are accepted for interface parity but ignored.</remarks>
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
