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
    {
        EncodeOptions options = new() { AddGenerationPrompt = addGenerationPrompt, Thinking = enableThinking ?? false };
        return _encoder.Encode(tokenizer, messages, options).Ids;
    }

    /// <summary>Builds the parser for completions of the prompt these <paramref name="messages"/> render to.</summary>
    public IOutputParser CreateParser(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool? enableThinking)
    {
        EncodeOptions options = new() { AddGenerationPrompt = true, Thinking = enableThinking ?? false };
        ParserInitialState initial = _encoder.ResolveParserState(messages, options);
        return _encoder.CreateParser(tokenizer, initial.ToParserState());
    }
}
