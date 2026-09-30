using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Drives an <see cref="IChatTemplate"/> from a model's own Jinja <c>chat_template</c> (GGUF metadata / <c>tokenizer_config.json</c>) via <see cref="JinjaEngine"/>, giving non-Qwen models (Llama-3, Mistral, Phi, Gemma, DeepSeek) their correct prompt format instead of a hardcoded one.</summary>
public sealed class JinjaChatTemplate : IChatTemplate
{
    private readonly JinjaEngine _engine;

    public string Name => "jinja";

    /// <summary>Compiles the model's chat-template source.</summary>
    public JinjaChatTemplate(string chatTemplate)
    {
        ArgumentNullException.ThrowIfNull(chatTemplate);
        _engine = new JinjaEngine(chatTemplate);
    }

    /// <inheritdoc/>
    public int[] Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt, bool? enableThinking = null)
        => Encode(tokenizer, messages, addGenerationPrompt, enableThinking, tools: null);

    /// <inheritdoc/>
    public int[] Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt, bool? enableThinking,
        IReadOnlyList<ToolSpec>? tools)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(messages);

        string rendered;
        try
        {
            try
            {
                rendered = Render(tokenizer, messages, addGenerationPrompt, enableThinking, tools);
            }
            catch (ChatTemplateRaiseException)
            {
                // The template rejected the message structure — almost always a model (Mistral, Gemma, …)
                // whose template has no system role and/or demands strict user/assistant alternation. Fold any
                // system content into the first user turn and merge consecutive same-role turns, then retry once.
                // Matches what llama.cpp/Ollama do for system-less templates. If it still fails, surface the
                // original error (a genuine template problem, not a structure one).
                rendered = Render(tokenizer, NormalizeForStrictTemplate(messages), addGenerationPrompt, enableThinking, tools);
            }
        }
        catch (Exception ex) when (ex is not ChatTemplateRaiseException)
        {
            // Same tolerance GgufLanguageModel.BuildTemplate applies at compile time (some templates use
            // constructs the engine doesn't support yet) — a construct that only shows up on a real conversation
            // shape can still fail at render time even though the template compiled fine in isolation. Degrade
            // to ChatML rather than failing the whole generation; if the tokenizer has no ChatML control tokens
            // either, that Encode call throws its own clear error instead of this one.
            Console.Error.WriteLine($"[WRN] GGUF: chat template failed to render ({ex.Message}); falling back to ChatML for this request.");
            return new ChatMlTemplate().Encode(tokenizer, messages, addGenerationPrompt, enableThinking, tools);
        }
        // The template emits the bos_token literal itself, so don't double-add specials beyond literal matching.
        return tokenizer.Encode(rendered, addSpecial: true);
    }

    /// <summary>Renders the conversation through the model's Jinja template; an unset <paramref name="enableThinking"/> leaves <c>enable_thinking</c> undefined so <c>{% if enable_thinking is defined %}</c> branches fall through to the template's own default instead of being forced off.</summary>
    private string Render(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt, bool? enableThinking,
        IReadOnlyList<ToolSpec>? tools)
    {
        List<object?> msgList = new(messages.Count);
        foreach (ChatMessage m in messages)
            msgList.Add(MessageValue(m));

        Dictionary<string, object?> context = new()
        {
            ["messages"] = msgList,
            ["add_generation_prompt"] = addGenerationPrompt,
            ["bos_token"] = tokenizer.BosToken ?? string.Empty,
            ["eos_token"] = tokenizer.EosToken ?? string.Empty,
            // Null rather than an empty list when nothing is offered: templates branch on `tools is not none`.
            ["tools"] = ToolsValue(tools),
            ["documents"] = null,
        };
        if (enableThinking.HasValue)
            context["enable_thinking"] = enableThinking.Value;
        return _engine.Render(context);
    }

    /// <summary>The OpenAI-shaped message dictionary Hugging Face templates expect: <c>tool_calls[i].function.{name,arguments}</c> with arguments as a parsed object when they are JSON, plus <c>tool_call_id</c>, <c>name</c> and <c>reasoning_content</c> when set.</summary>
    private static Dictionary<string, object?> MessageValue(ChatMessage m)
    {
        Dictionary<string, object?> value = new() { ["role"] = m.Role, ["content"] = m.Content };
        if (m.ToolCalls is { Count: > 0 })
        {
            List<object?> calls = new(m.ToolCalls.Count);
            foreach (ChatToolCall call in m.ToolCalls)
            {
                calls.Add(new Dictionary<string, object?>
                {
                    ["id"] = call.Id,
                    ["type"] = "function",
                    ["function"] = new Dictionary<string, object?> { ["name"] = call.Name, ["arguments"] = ArgumentsValue(call.ArgumentsJson) },
                });
            }
            value["tool_calls"] = calls;
        }
        if (m.ToolCallId is not null) value["tool_call_id"] = m.ToolCallId;
        if (m.Name is not null) value["name"] = m.Name;
        if (m.ReasoningContent is not null) value["reasoning_content"] = m.ReasoningContent;
        return value;
    }

    private static List<object?>? ToolsValue(IReadOnlyList<ToolSpec>? tools)
    {
        if (tools is not { Count: > 0 }) return null;
        List<object?> value = new(tools.Count);
        foreach (ToolSpec tool in tools) value.Add(Values.ParseJson(tool.Json));
        return value;
    }

    private static object? ArgumentsValue(string argumentsJson)
    {
        try
        {
            return Values.ParseJson(argumentsJson);
        }
        catch (JsonException)
        {
            // Not JSON: templates render a string argument verbatim through their `arguments is string` branch.
            return argumentsJson;
        }
    }

    /// <summary>Rewrites a conversation into the shape a system-less, strictly-alternating template accepts: system content folded into the first user turn, consecutive same-role turns merged (always merges, even if already valid).</summary>
    private static List<ChatMessage> NormalizeForStrictTemplate(IReadOnlyList<ChatMessage> messages)
    {
        // Pull system content aside (preserving order) and keep the rest.
        StringBuilder system = new();
        List<ChatMessage> rest = [];
        foreach (ChatMessage m in messages)
        {
            if (string.Equals(m.Role, "system", StringComparison.OrdinalIgnoreCase))
            {
                if (!string.IsNullOrEmpty(m.Content))
                {
                    if (system.Length > 0) system.Append("\n\n");
                    system.Append(m.Content);
                }
            }
            else
            {
                rest.Add(m);
            }
        }

        // Merge consecutive same-role turns so the sequence strictly alternates.
        List<ChatMessage> merged = [];
        foreach (ChatMessage m in rest)
        {
            if (merged.Count > 0 && string.Equals(merged[^1].Role, m.Role, StringComparison.OrdinalIgnoreCase))
            {
                string joined = string.IsNullOrEmpty(merged[^1].Content) ? (m.Content ?? "")
                    : string.IsNullOrEmpty(m.Content) ? merged[^1].Content
                    : merged[^1].Content + "\n\n" + m.Content;
                merged[^1] = merged[^1] with { Content = joined };
            }
            else
            {
                merged.Add(m);
            }
        }

        // Fold system content into the first user turn (or prepend one if the convo doesn't start with user).
        if (system.Length > 0)
        {
            if (merged.Count > 0 && string.Equals(merged[0].Role, "user", StringComparison.OrdinalIgnoreCase))
            {
                string content = string.IsNullOrEmpty(merged[0].Content) ? system.ToString()
                    : system + "\n\n" + merged[0].Content;
                merged[0] = merged[0] with { Content = content };
            }
            else
            {
                merged.Insert(0, ChatMessage.User(system.ToString()));
            }
        }
        return merged;
    }
}
