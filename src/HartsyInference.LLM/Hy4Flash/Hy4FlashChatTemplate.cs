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

    /// <summary>The think literals the template writes, <c>&lt;think:opensource&gt;</c> and <c>&lt;/think:opensource&gt;</c> (the template's <c>think_begin_token</c> and <c>think_end_token</c>).
    /// They are not the <c>&lt;think&gt;</c> pair the other templates use, so the reasoning split reads these.</summary>
    public static IReadOnlyList<string> ReasoningMarkers { get; } = ["<think:opensource>", "</think:opensource>"];

    private readonly JinjaChatTemplate _default;
    private readonly JinjaChatTemplate _noThink;

    /// <param name="source">The checkpoint's <c>chat_template.jinja</c>.</param>
    public Hy4FlashChatTemplate(string source)
    {
        ArgumentException.ThrowIfNullOrEmpty(source);
        _default = new JinjaChatTemplate(source, renderImageBlocks: true);
        _noThink = new JinjaChatTemplate("{% set reasoning_effort = 'no_think' %}\n" + source, renderImageBlocks: true);
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

    /// <summary>Whether the prompt for <paramref name="messages"/> ends inside a think block: the variant the request selects writes the
    /// <see cref="ReasoningMarkers"/> open and does not close them.</summary>
    public ParserInitialState ResolveParserState(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool? enableThinking,
        IReadOnlyList<ToolSpec>? tools, int? reasoningEffort) =>
        Variant(enableThinking, reasoningEffort).ResolveParserState(tokenizer, messages, enableThinking, tools, ReasoningMarkers);

    /// <summary>The image placeholder literals the template writes around each image: start, one pad, end.</summary>
    public const string ImageStartLiteral = "<｜hy_image▁start:opensource｜>";

    /// <summary>The single pad the template writes per image; <see cref="EncodeWithImages"/> expands it to the image's feature rows.</summary>
    public const string ImagePadLiteral = "<｜hy_image▁pad:opensource｜>";

    /// <summary>The image end literal the template writes after the pad.</summary>
    public const string ImageEndLiteral = "<｜hy_image▁end:opensource｜>";

    /// <summary>Encodes a conversation whose user turns may carry images (<c>ImageBlock</c>), then expands each image's pad to the
    /// tokens its features fill: <c>(grid height / merge) * (grid width / merge + 1)</c>, one newline per merged row. <paramref name="grids"/> lists the
    /// patch grids in the order the images appear.</summary>
    /// <exception cref="ArgumentException">The prompt's pads do not match the grids, or a grid is not a positive multiple of the merge.</exception>
    public int[] EncodeWithImages(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt, bool? enableThinking,
        IReadOnlyList<ToolSpec>? tools, int? reasoningEffort, IReadOnlyList<(int GridHeight, int GridWidth)> grids, int mergeSize)
    {
        ArgumentNullException.ThrowIfNull(grids);
        ArgumentOutOfRangeException.ThrowIfLessThan(mergeSize, 1);
        int[] ids = Variant(enableThinking, reasoningEffort).Encode(tokenizer, messages, addGenerationPrompt, enableThinking, tools);
        int pad = tokenizer.SpecialId(ImagePadLiteral) ?? throw new ArgumentException($"The tokenizer has no id for {ImagePadLiteral}.");
        List<int> expanded = new(ids.Length + grids.Count * 8);
        int image = 0;
        foreach (int id in ids)
        {
            if (id != pad)
            {
                expanded.Add(id);
                continue;
            }
            if (image >= grids.Count) throw new ArgumentException($"The prompt has more image pads than the {grids.Count} image(s) given.", nameof(grids));
            (int gridHeight, int gridWidth) = grids[image++];
            if (gridHeight < 1 || gridWidth < 1 || gridHeight % mergeSize != 0 || gridWidth % mergeSize != 0)
                throw new ArgumentException($"Image {image}'s patch grid must be a positive multiple of the merge size {mergeSize} on each side.", nameof(grids));
            int tokens = gridHeight / mergeSize * (gridWidth / mergeSize + 1);
            for (int t = 0; t < tokens; t++) expanded.Add(pad);
        }
        if (image != grids.Count) throw new ArgumentException($"The prompt has {image} image pad(s) for {grids.Count} image(s).", nameof(grids));
        return [.. expanded];
    }

    // no_think when thinking is off or the effort is low (LowEffort or less); otherwise the checkpoint's own default, which is high.
    private JinjaChatTemplate Variant(bool? enableThinking, int? reasoningEffort) =>
        enableThinking == false || reasoningEffort is <= LowEffort ? _noThink : _default;
}
