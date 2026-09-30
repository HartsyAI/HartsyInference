using HartsyInference.LLM.OutputParsing;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Exposes an <see cref="IConversationEncoder"/> through <see cref="IChatTemplate"/> so existing callers keep working; only the ids are returned.</summary>
public sealed class ChatTemplateEncoderAdapter : IChatTemplate
{
    private readonly IConversationEncoder _encoder;

    /// <summary>Wraps <paramref name="encoder"/>.</summary>
    public ChatTemplateEncoderAdapter(IConversationEncoder encoder)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        _encoder = encoder;
    }

    /// <inheritdoc />
    public string Name => _encoder.Name;

    /// <inheritdoc />
    public int[] Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt,
        bool? enableThinking = null)
        => Encode(tokenizer, messages, addGenerationPrompt, enableThinking, tools: null);

    /// <inheritdoc />
    public int[] Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt,
        bool? enableThinking, IReadOnlyList<ToolSpec>? tools)
        => _encoder.Encode(tokenizer, messages, Options(addGenerationPrompt, enableThinking, tools)).Ids;

    /// <summary>Builds the parser for completions of the prompt these <paramref name="messages"/> render to (with <paramref name="tools"/> offered, when any).</summary>
    public IOutputParser CreateParser(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool? enableThinking,
        IReadOnlyList<ToolSpec>? tools = null)
    {
        ParserInitialState initial = _encoder.ResolveParserState(messages, Options(addGenerationPrompt: true, enableThinking, tools));
        return _encoder.CreateParser(tokenizer, initial.ToParserState());
    }

    private static EncodeOptions Options(bool addGenerationPrompt, bool? enableThinking, IReadOnlyList<ToolSpec>? tools) => new()
    {
        AddGenerationPrompt = addGenerationPrompt,
        Thinking = enableThinking ?? false,
        Tools = tools is { Count: > 0 } ? tools : null,
    };
}
