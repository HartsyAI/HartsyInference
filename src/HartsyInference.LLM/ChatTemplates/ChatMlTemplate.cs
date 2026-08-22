using System;
using System.Collections.Generic;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Built-in ChatML fallback for Qwen2.5/Qwen3 (and any format-compatible model) used when a GGUF carries no <c>chat_template</c>; requires a tokenizer that knows <c>&lt;|im_start|&gt;</c>/<c>&lt;|im_end|&gt;</c>.</summary>
public sealed class ChatMlTemplate : IChatTemplate
{
    private const string ImStart = "<|im_start|>";
    private const string ImEnd = "<|im_end|>";

    /// <inheritdoc/>
    public string Name => "chatml";

    /// <inheritdoc/>
    /// <remarks>ChatML has no <c>enable_thinking</c> slot; <paramref name="enableThinking"/> is accepted for interface parity but ignored.</remarks>
    public int[] Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt, bool? enableThinking = null)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(messages);

        int imStart = tokenizer.SpecialId(ImStart) ?? throw new InvalidOperationException("Tokenizer has no <|im_start|> token; ChatML template not applicable.");
        int imEnd = tokenizer.SpecialId(ImEnd) ?? throw new InvalidOperationException("Tokenizer has no <|im_end|> token; ChatML template not applicable.");

        List<int> ids = new(messages.Count * 16 + 8);
        for (int i = 0; i < messages.Count; i++)
        {
            ChatMessage message = messages[i];
            ids.Add(imStart);
            ids.AddRange(tokenizer.EncodeOrdinary(message.Role + "\n" + message.Content));
            ids.Add(imEnd);
            ids.AddRange(tokenizer.EncodeOrdinary("\n"));
        }
        if (addGenerationPrompt)
        {
            ids.Add(imStart);
            ids.AddRange(tokenizer.EncodeOrdinary("assistant\n"));
        }
        return ids.ToArray();
    }

    /// <summary>Encodes a one-shot turn: optional system message (null uses the default helpful-assistant prompt, empty string omits it) + a single user turn + a trailing assistant generation prompt.</summary>
    public static int[] EncodeSingleTurn(ILlmTokenizer tok, string userPrompt, string? systemPrompt)
    {
        ArgumentNullException.ThrowIfNull(tok);
        ArgumentNullException.ThrowIfNull(userPrompt);

        string system = systemPrompt ?? "You are a helpful assistant.";
        List<ChatMessage> messages = new(2);
        if (system.Length > 0) messages.Add(ChatMessage.System(system));
        messages.Add(ChatMessage.User(userPrompt));
        return new ChatMlTemplate().Encode(tok, messages, addGenerationPrompt: true);
    }
}
