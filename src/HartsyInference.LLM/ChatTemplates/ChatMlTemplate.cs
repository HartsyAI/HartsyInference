using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Built-in ChatML fallback for Qwen2.5/Qwen3 (and any format-compatible model) used when a GGUF carries no <c>chat_template</c>; requires a tokenizer that knows <c>&lt;|im_start|&gt;</c>/<c>&lt;|im_end|&gt;</c>. Tool schemas, assistant tool calls and tool results render in Qwen2.5's shape (<c># Tools</c> system block, <c>&lt;tool_call&gt;</c>, <c>&lt;tool_response&gt;</c> inside a user turn).</summary>
public sealed class ChatMlTemplate : IChatTemplate
{
    private const string ImStart = "<|im_start|>";
    private const string ImEnd = "<|im_end|>";
    private const string DefaultSystem = "You are a helpful assistant.";
    /// <inheritdoc/>
    public string Name => "chatml";

    /// <inheritdoc/>
    /// <remarks>ChatML has no <c>enable_thinking</c> slot; <paramref name="enableThinking"/> is accepted for interface parity but ignored.</remarks>
    public int[] Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt, bool? enableThinking = null)
        => Encode(tokenizer, messages, addGenerationPrompt, enableThinking, tools: null);

    /// <inheritdoc/>
    public int[] Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt, bool? enableThinking,
        IReadOnlyList<ToolSpec>? tools)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(messages);

        int imStart = tokenizer.SpecialId(ImStart) ?? throw new InvalidOperationException("Tokenizer has no <|im_start|> token; ChatML template not applicable.");
        int imEnd = tokenizer.SpecialId(ImEnd) ?? throw new InvalidOperationException("Tokenizer has no <|im_end|> token; ChatML template not applicable.");

        List<int> ids = new(messages.Count * 16 + 8);
        void Turn(string text)
        {
            ids.Add(imStart);
            ids.AddRange(tokenizer.EncodeOrdinary(text));
            ids.Add(imEnd);
            ids.AddRange(tokenizer.EncodeOrdinary("\n"));
        }

        bool withTools = tools is { Count: > 0 };
        bool leadingSystem = messages.Count > 0 && IsRole(messages[0], "system");
        if (withTools)
            Turn("system\n" + (leadingSystem ? messages[0].Content : DefaultSystem) + HermesToolPrompt.ToolsBlock(tools!));
        for (int i = 0; i < messages.Count; i++)
        {
            ChatMessage message = messages[i];
            if (IsRole(message, "system") && i == 0 && withTools)
                continue;
            if (IsRole(message, "tool"))
            {
                // Consecutive tool results share one user turn, as Qwen's own template renders them.
                StringBuilder responses = new("user");
                for (; i < messages.Count && IsRole(messages[i], "tool"); i++)
                    responses.Append("\n<tool_response>\n").Append(messages[i].Content).Append("\n</tool_response>");
                i--;
                Turn(responses.ToString());
                continue;
            }
            if (IsRole(message, "assistant") && message.ToolCalls is { Count: > 0 })
            {
                StringBuilder body = new("assistant");
                if (!string.IsNullOrEmpty(message.Content)) body.Append('\n').Append(message.Content);
                foreach (ChatToolCall call in message.ToolCalls)
                    body.Append("\n<tool_call>\n{\"name\": ").Append(Values.ToJson(call.Name)).Append(", \"arguments\": ").Append(HermesToolPrompt.ArgumentsJson(call.ArgumentsJson)).Append("}\n</tool_call>");
                Turn(body.ToString());
                continue;
            }
            Turn(message.Role + "\n" + message.Content);
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

        string system = systemPrompt ?? DefaultSystem;
        List<ChatMessage> messages = new(2);
        if (system.Length > 0) messages.Add(ChatMessage.System(system));
        messages.Add(ChatMessage.User(userPrompt));
        return new ChatMlTemplate().Encode(tok, messages, addGenerationPrompt: true);
    }

    private static bool IsRole(ChatMessage message, string role) => string.Equals(message.Role, role, StringComparison.OrdinalIgnoreCase);

}
