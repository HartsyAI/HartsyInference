using System.Text;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.OutputParsing;

/// <summary>Parser for every template without a structured output format: all decoded text is content, control tokens are skipped, so streaming output equals a plain decode.</summary>
public sealed class PassthroughOutputParser : IOutputParser
{
    private readonly IncrementalDetokenizer _detok;
    private readonly StringBuilder _content = new();
    private bool _done;

    /// <summary>Creates a passthrough parser over <paramref name="tokenizer"/>.</summary>
    public PassthroughOutputParser(ILlmTokenizer tokenizer) => _detok = new IncrementalDetokenizer(tokenizer, includeSpecial: false);

    /// <inheritdoc />
    public ParsedAssistant Result => new("", _content.ToString(), [], false, _done);

    /// <inheritdoc />
    public void Push(int tokenId, Action<ParsedEvent> sink)
    {
        ArgumentNullException.ThrowIfNull(sink);
        if (_done) return;
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
        if (text.Length == 0) return;
        _content.Append(text);
        sink(new ParsedEvent(ParsedEventKind.ContentDelta, text));
    }
}
