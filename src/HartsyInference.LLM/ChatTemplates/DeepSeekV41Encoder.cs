using HartsyInference.LLM.OutputParsing;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>DeepSeek-V4.1 prompt encoder: renders the reference format's text, tokenizes it with special literals recognised, then expands image placeholders into their token spans.</summary>
public sealed class DeepSeekV41Encoder : IConversationEncoder
{
    /// <summary>Literal of the image placeholder token whose id fills every image-span position.</summary>
    public const string ImageTokenLiteral = DeepSeekV41PromptRenderer.ImagePlaceholder;

    /// <inheritdoc />
    public string Name => "deepseek_v41";

    /// <summary>Renders the exact prompt string (BOS included) without tokenizing.</summary>
    public static string RenderText(IReadOnlyList<ChatMessage> messages, EncodeOptions options)
    {
        ArgumentNullException.ThrowIfNull(messages);
        ArgumentNullException.ThrowIfNull(options);
        return DeepSeekV41PromptRenderer.Render(messages, options, []);
    }

    /// <inheritdoc />
    public EncodedConversation Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, EncodeOptions options)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(messages);
        ArgumentNullException.ThrowIfNull(options);
        List<int> imageOrder = [];
        string text = DeepSeekV41PromptRenderer.Render(messages, options, imageOrder);
        int[] raw = tokenizer.Encode(text, true);
        bool reasoningOpen = text.EndsWith(DeepSeekV41Tools.ThinkStart, StringComparison.Ordinal);
        if (imageOrder.Count == 0)
        {
            bool[] none = new bool[raw.Length];
            return new EncodedConversation(raw, [], none, (bool[])none.Clone(), raw.Length, new ParserInitialState(reasoningOpen));
        }
        int placeholderId = tokenizer.SpecialId(ImageTokenLiteral)
            ?? throw new InvalidOperationException($"Tokenizer has no '{ImageTokenLiteral}' token.");
        return ExpandImages(raw, placeholderId, imageOrder, options.Images!, reasoningOpen);
    }

    /// <inheritdoc />
    public ParserInitialState ResolveParserState(IReadOnlyList<ChatMessage> messages, EncodeOptions options)
    {
        int imageSlots = messages.Select(MaxImageIndex).DefaultIfEmpty(0).Max();
        EncodeOptions probe = options.Images is null && imageSlots > 0
            ? options with { Images = Enumerable.Repeat(new ImageGrid(1, 1), imageSlots).ToList() }
            : options;
        string text = RenderText(messages, probe);
        return new ParserInitialState(text.EndsWith(DeepSeekV41Tools.ThinkStart, StringComparison.Ordinal));
    }

    /// <inheritdoc />
    public IOutputParser CreateParser(ILlmTokenizer tokenizer, OutputParserState state)
        => new DeepSeekV41OutputParser(tokenizer, state);

    private static int MaxImageIndex(ChatMessage message)
        => message.Blocks?.OfType<ImageBlock>().Select(b => b.ImageIndex + 1).DefaultIfEmpty(0).Max() ?? 0;

    private static EncodedConversation ExpandImages(int[] raw, int placeholderId, List<int> imageOrder,
        IReadOnlyList<ImageGrid> grids, bool reasoningOpen)
    {
        List<int> ids = new(raw.Length);
        List<bool> mask = new(raw.Length);
        List<ImageSpan> spans = new(imageOrder.Count);
        int next = 0;
        foreach (int id in raw)
        {
            if (id != placeholderId)
            {
                ids.Add(id);
                mask.Add(false);
                continue;
            }
            if (next >= imageOrder.Count)
                throw new InvalidOperationException("Encoded prompt holds more image placeholders than image blocks.");
            int imageIndex = imageOrder[next++];
            ImageGrid grid = grids[imageIndex];
            spans.Add(new ImageSpan(imageIndex, ids.Count, grid.TokenCount, grid));
            for (int i = 0; i < grid.TokenCount; i++)
            {
                ids.Add(placeholderId);
                mask.Add(true);
            }
        }
        if (next != imageOrder.Count)
            throw new InvalidOperationException("Encoded prompt lost image placeholders during tokenization.");
        bool[] dead = mask.ToArray();
        return new EncodedConversation([.. ids], spans, dead, (bool[])dead.Clone(), ids.Count, new ParserInitialState(reasoningOpen));
    }
}
