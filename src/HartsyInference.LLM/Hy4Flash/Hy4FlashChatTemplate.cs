using HartsyInference.LLM.ChatTemplates;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>The Hy4 Flash chat template: the checkpoint's own Jinja template, which defaults <c>reasoning_effort</c> to <c>high</c>. The <c>no_think</c> variant is selected when
/// thinking is switched off (<c>enable_thinking</c> false) or the request effort is at or below <see cref="LowEffort"/> on the 1 to 100 scale. Any other request renders the default.</summary>
/// <remarks>The template reads <c>reasoning_effort</c> only as a variable, so the variant defines it at the top of the source. The Jinja engine has no per-call
/// context hook, so both variants are compiled once at load.</remarks>
public sealed class Hy4FlashChatTemplate : IChatTemplate
{
    /// <summary>The named effort "low" on the 1 to 100 scale. Efforts at or below it, and thinking switched off, select the no_think variant.</summary>
    public const int LowEffort = 50;

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
        => Variant(enableThinking, reasoningEffort: null).Encode(tokenizer, messages, addGenerationPrompt, enableThinking);

    /// <inheritdoc />
    public int[] Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt, bool? enableThinking,
        IReadOnlyList<ToolSpec>? tools)
        => Variant(enableThinking, reasoningEffort: null).Encode(tokenizer, messages, addGenerationPrompt, enableThinking, tools);

    /// <inheritdoc />
    public int[] Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt, bool? enableThinking,
        IReadOnlyList<ToolSpec>? tools, int? reasoningEffort)
        => Variant(enableThinking, reasoningEffort).Encode(tokenizer, messages, addGenerationPrompt, enableThinking, tools);

    // no_think when thinking is off or the effort is low (LowEffort or less); otherwise the checkpoint's own default, which is high.
    private JinjaChatTemplate Variant(bool? enableThinking, int? reasoningEffort) =>
        enableThinking == false || reasoningEffort is <= LowEffort ? _noThink : _default;
}
