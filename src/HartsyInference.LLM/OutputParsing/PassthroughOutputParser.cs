using System.Text;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.OutputParsing;

/// <summary>Parser for every template without a structured output format: all decoded text is content, control tokens are skipped, so streaming output equals a plain decode.
/// When the prompt ended inside a think block (a template that prefills <c>&lt;think&gt;</c>), the text up to the reasoning close token is reasoning instead.</summary>
public sealed class PassthroughOutputParser : IOutputParser
{
    private readonly IncrementalDetokenizer _detok;
    private readonly StringBuilder _reasoning = new();
    private readonly StringBuilder _content = new();
    private readonly int _reasoningCloseId;
    private bool _inReasoning;
    private bool _trimLeading;
    private bool _done;

    /// <summary>Creates a passthrough parser over <paramref name="tokenizer"/>.</summary>
    public PassthroughOutputParser(ILlmTokenizer tokenizer) : this(tokenizer, null)
    {
    }

    /// <summary>Creates a passthrough parser that also decodes the control tokens in <paramref name="literalSpecialIds"/> as text (for tool-call markers a filter asked to see).</summary>
    public PassthroughOutputParser(ILlmTokenizer tokenizer, IReadOnlySet<int>? literalSpecialIds) : this(tokenizer, literalSpecialIds, null)
    {
    }

    /// <summary>Creates a passthrough parser; a non-null <paramref name="openReasoningCloseId"/> means the prompt ended inside a think block, so the
    /// completion is reasoning until that token, which is consumed, and content after it, with the whitespace that opens the answer dropped.</summary>
    public PassthroughOutputParser(ILlmTokenizer tokenizer, IReadOnlySet<int>? literalSpecialIds, int? openReasoningCloseId)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        _detok = new IncrementalDetokenizer(tokenizer, includeSpecial: false, literalSpecialIds);
        _inReasoning = openReasoningCloseId.HasValue;
        _reasoningCloseId = openReasoningCloseId ?? -1;
    }

    /// <inheritdoc />
    public ParsedAssistant Result => new(_reasoning.ToString(), _content.ToString(), [], false, _done);

    /// <inheritdoc />
    public void Push(int tokenId, Action<ParsedEvent> sink)
    {
        ArgumentNullException.ThrowIfNull(sink);
        if (_done) return;
        if (_inReasoning && tokenId == _reasoningCloseId)
        {
            Emit(_detok.Flush(), sink);
            _inReasoning = false;
            _trimLeading = true;
            return;
        }
        Emit(_detok.Push(tokenId), sink);
    }

    /// <inheritdoc />
    public void Finish(Action<ParsedEvent> sink)
    {
        ArgumentNullException.ThrowIfNull(sink);
        if (_done) return;
        Emit(_detok.Flush(), sink);
        _done = true;
        sink(new ParsedEvent(ParsedEventKind.Stop));
    }

    private void Emit(string text, Action<ParsedEvent> sink)
    {
        if (_inReasoning)
        {
            if (text.Length == 0) return;
            _reasoning.Append(text);
            sink(new ParsedEvent(ParsedEventKind.ReasoningDelta, text));
            return;
        }
        if (_trimLeading)
        {
            text = text.TrimStart();
            if (text.Length == 0) return;
            _trimLeading = false;
        }
        if (text.Length == 0) return;
        _content.Append(text);
        sink(new ParsedEvent(ParsedEventKind.ContentDelta, text));
    }
}
