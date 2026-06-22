using System;
using System.Collections.Generic;
using HartsyInference.Tokenizers;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>The Qwen ChatML template (shared by Qwen2.5 and Qwen3 text models, which use the same format and vocab). Mirrors <see cref="Qwen2Tokenizer.EncodeChat(string, string?, bool)"/> but generalized to an arbitrary list of turns.</summary>
public sealed class ChatMlTemplate : IChatTemplate
{
    /// <inheritdoc/>
    public string Name => "chatml";

    /// <inheritdoc/>
    public int[] Encode(Qwen2Tokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(messages);

        List<int> ids = new(messages.Count * 16 + 8);
        for (int i = 0; i < messages.Count; i++)
        {
            ChatMessage message = messages[i];
            ids.Add(Qwen2Tokenizer.ImStartId);
            AppendRaw(ids, tokenizer, message.Role + "\n" + message.Content);
            ids.Add(Qwen2Tokenizer.ImEndId);
            AppendRaw(ids, tokenizer, "\n");
        }
        if (addGenerationPrompt)
        {
            ids.Add(Qwen2Tokenizer.ImStartId);
            AppendRaw(ids, tokenizer, "assistant\n");
        }
        return ids.ToArray();
    }

    /// <summary>Encodes a one-shot turn: an optional system message (null uses <see cref="Qwen2Tokenizer.DefaultSystemPrompt"/>, empty string omits it) followed by a single user turn, with a trailing assistant generation prompt.</summary>
    public static int[] EncodeSingleTurn(Qwen2Tokenizer tok, string userPrompt, string? systemPrompt)
    {
        ArgumentNullException.ThrowIfNull(tok);
        ArgumentNullException.ThrowIfNull(userPrompt);

        string system = systemPrompt ?? Qwen2Tokenizer.DefaultSystemPrompt;
        List<ChatMessage> messages = new(2);
        if (system.Length > 0)
            messages.Add(ChatMessage.System(system));
        messages.Add(ChatMessage.User(userPrompt));
        return new ChatMlTemplate().Encode(tok, messages, addGenerationPrompt: true);
    }

    private static void AppendRaw(List<int> dst, Qwen2Tokenizer tokenizer, string text)
    {
        IReadOnlyList<int> ids = tokenizer.EncodeRaw(text);
        for (int i = 0; i < ids.Count; i++)
            dst.Add(ids[i]);
    }
}
