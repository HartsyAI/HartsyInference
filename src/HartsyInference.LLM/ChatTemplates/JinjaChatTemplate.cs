using System;
using System.Collections.Generic;
using HartsyInference.Tokenizers;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>An <see cref="IChatTemplate"/> driven by a model's own Jinja <c>chat_template</c> (from GGUF
/// metadata / <c>tokenizer_config.json</c>), rendered by <see cref="JinjaEngine"/>. This is how non-Qwen models
/// (Llama-3, Mistral, Phi, Gemma, DeepSeek) get their correct prompt format instead of a hardcoded one. The
/// rendered string is tokenized special-token-aware so control-token literals map to their ids.</summary>
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
    public int[] Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(messages);

        List<object?> msgList = new(messages.Count);
        foreach (ChatMessage m in messages)
            msgList.Add(new Dictionary<string, object?> { ["role"] = m.Role, ["content"] = m.Content });

        Dictionary<string, object?> context = new()
        {
            ["messages"] = msgList,
            ["add_generation_prompt"] = addGenerationPrompt,
            ["bos_token"] = tokenizer.BosToken ?? string.Empty,
            ["eos_token"] = tokenizer.EosToken ?? string.Empty,
            ["tools"] = null,
            ["documents"] = null,
        };

        string rendered = _engine.Render(context);
        // The template emits the bos_token literal itself, so don't double-add specials beyond literal matching.
        return tokenizer.Encode(rendered, addSpecial: true);
    }
}
