using HartsyInference.LLM.ChatTemplates;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>The Hy4 Flash chat template: the checkpoint's own Jinja template, which defaults <c>reasoning_effort</c> to <c>high</c>, with the <c>no_think</c>
/// variant selected by a request effort of 1. Any other effort, or none, renders the default.</summary>
/// <remarks>The template reads <c>reasoning_effort</c> only as a variable, so the variant defines it at the top of the source. The Jinja engine has no per-call
/// context hook, so both variants are compiled once at load. <c>enable_thinking</c> is not a Hy4 switch and is ignored.</remarks>
public sealed class Hy4FlashChatTemplate : IChatTemplate
{
    private readonly JinjaChatTemplate _default;
    private readonly JinjaChatTemplate _noThink;

    /// <param name="source">The checkpoint's <c>chat_template.jinja</c>.</param>
    public Hy4FlashChatTemplate(string source)
    {
        ArgumentException.ThrowIfNullOrEmpty(source);
        _default = new JinjaChatTemplate(source);
        _noThink = new JinjaChatTemplate("{% set reasoning_effort = 'no_think' %}\n" + source);
    }

    /// <inheritdoc />
    public string Name => "hy4-flash-jinja";

    /// <inheritdoc />
    public string? Source => _default.Source;

    /// <inheritdoc />
    public int[] Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt, bool? enableThinking = null)
        => _default.Encode(tokenizer, messages, addGenerationPrompt, enableThinking);

    /// <inheritdoc />
    public int[] Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt, bool? enableThinking,
        IReadOnlyList<ToolSpec>? tools)
        => _default.Encode(tokenizer, messages, addGenerationPrompt, enableThinking, tools);

    /// <inheritdoc />
    public int[] Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt, bool? enableThinking,
        IReadOnlyList<ToolSpec>? tools, int? reasoningEffort)
        => (reasoningEffort == 1 ? _noThink : _default).Encode(tokenizer, messages, addGenerationPrompt, enableThinking, tools);
}
