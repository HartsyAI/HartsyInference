using System.Text;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.OutputParsing;

/// <summary>Streaming parser for DeepSeek-V4.1 completions: reasoning up to <c>&lt;/think&gt;</c>, answer text, then an optional DSML tool-calls block ended by EOS. Text is matched on the detokenized stream, so a marker split across any token boundary is still found.</summary>
public sealed class DeepSeekV41OutputParser : IOutputParser
{
    /// <summary>Marker that opens the tool-calls block; not a single token, so it is matched at text level.</summary>
    public const string CallsMarker = "\n\n<" + DeepSeekV41Tools.DsmlToken + DeepSeekV41Tools.ToolCallsBlockName;

    private const string Dsml = DeepSeekV41Tools.DsmlToken;
    private const string ThinkEnd = DeepSeekV41Tools.ThinkEnd;
    private const string ThinkStart = DeepSeekV41Tools.ThinkStart;

    private static readonly string[] ReasoningStrict = [ThinkEnd, DeepSeekV41PromptRenderer.Eos, ThinkStart, DeepSeekV41PromptRenderer.Bos, Dsml];
    private static readonly string[] ReasoningLenient = [ThinkEnd, DeepSeekV41PromptRenderer.Eos];
    private static readonly string[] ContentStrict =
        [DeepSeekV41PromptRenderer.Eos, CallsMarker, ThinkEnd, ThinkStart, DeepSeekV41PromptRenderer.Bos, Dsml];
    private static readonly string[] ContentLenient = [DeepSeekV41PromptRenderer.Eos, CallsMarker];
    private static readonly string[] ToolMarkers = [DeepSeekV41PromptRenderer.Eos];

    private readonly IncrementalDetokenizer _detok;
    private readonly ILlmTokenizer _tokenizer;
    private readonly Dictionary<int, string> _literalById = [];
    private readonly StringBuilder _reasoning = new();
    private readonly StringBuilder _content = new();
    private readonly List<ChatToolCall> _calls = [];
    private OutputParserSection _section;
    private DsmlCallsParser? _dsml;
    private string _pending = "";
    private bool _lenient;
    private bool _malformed;
    private bool _completed;

    /// <summary>Creates a parser starting in <paramref name="state"/>; throws when the tokenizer lacks the pinned special tokens.</summary>
    public DeepSeekV41OutputParser(ILlmTokenizer tokenizer, OutputParserState state)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        _tokenizer = tokenizer;
        _detok = new IncrementalDetokenizer(tokenizer, includeSpecial: true);
        foreach (string literal in new[] { DeepSeekV41PromptRenderer.Eos, DeepSeekV41PromptRenderer.Bos, ThinkStart, ThinkEnd, Dsml })
        {
            int id = tokenizer.SpecialId(literal)
                ?? throw new InvalidOperationException($"Tokenizer has no '{literal}' special token; not a DeepSeek-V4.1 vocabulary.");
            _literalById[id] = literal;
        }
        _section = state.Section == OutputParserSection.Done || state.Section == OutputParserSection.ToolCalls
            ? OutputParserSection.Content : state.Section;
    }

    /// <inheritdoc />
    public ParsedAssistant Result => new(_reasoning.ToString(), _content.ToString(), [.. _calls], _malformed, _completed);

    /// <inheritdoc />
    public void Push(int tokenId, Action<ParsedEvent> sink)
    {
        ArgumentNullException.ThrowIfNull(sink);
        if (_section == OutputParserSection.Done) return;
        string text;
        if (_literalById.TryGetValue(tokenId, out string? literal))
            text = _detok.Flush() + literal;
        else
            text = _detok.Push(tokenId);
        Process(text, final: false, sink);
    }

    /// <inheritdoc />
    public void Finish(Action<ParsedEvent> sink)
    {
        ArgumentNullException.ThrowIfNull(sink);
        if (_section == OutputParserSection.Done) return;
        Process(_detok.Flush(), final: true, sink);
        if (_section == OutputParserSection.Done) return;
        _dsml?.End(Wrap(sink));
        EndTurn(sink);
    }

    private Action<ParsedEvent> Wrap(Action<ParsedEvent> sink) => e =>
    {
        if (e.Kind == ParsedEventKind.Malformed) _malformed = true;
        sink(e);
    };

    private void Process(string text, bool final, Action<ParsedEvent> sink)
    {
        _pending += text;
        while (_section != OutputParserSection.Done)
        {
            string[] markers = MarkersFor();
            MarkerScanner.Scan(_pending, markers, out int at, out int which, out int hold);
            if (at >= 0 && (final || hold < 0 || at < hold))
            {
                EmitText(_pending[..at], sink);
                string marker = markers[which];
                _pending = _pending[(at + marker.Length)..];
                OnMarker(marker, sink);
                continue;
            }
            int cut = final || hold < 0 ? _pending.Length : hold;
            EmitText(_pending[..cut], sink);
            _pending = _pending[cut..];
            return;
        }
        _pending = "";
    }

    private string[] MarkersFor() => _section switch
    {
        OutputParserSection.Reasoning => _lenient ? ReasoningLenient : ReasoningStrict,
        OutputParserSection.Content => _lenient ? ContentLenient : ContentStrict,
        _ => ToolMarkers,
    };

    private void EmitText(string text, Action<ParsedEvent> sink)
    {
        if (text.Length == 0) return;
        switch (_section)
        {
            case OutputParserSection.Reasoning:
                _reasoning.Append(text);
                sink(new ParsedEvent(ParsedEventKind.ReasoningDelta, text));
                break;
            case OutputParserSection.Content:
                _content.Append(text);
                sink(new ParsedEvent(ParsedEventKind.ContentDelta, text));
                break;
            case OutputParserSection.ToolCalls:
                _dsml!.Feed(text, Wrap(sink));
                break;
        }
    }

    private void OnMarker(string marker, Action<ParsedEvent> sink)
    {
        if (marker == DeepSeekV41PromptRenderer.Eos)
        {
            if (_section == OutputParserSection.Reasoning) Flag(sink, "Invalid thinking format: missing </think>");
            _dsml?.End(Wrap(sink));
            EndTurn(sink);
        }
        else if (marker == ThinkEnd && _section == OutputParserSection.Reasoning)
        {
            _section = OutputParserSection.Content;
        }
        else if (marker == CallsMarker)
        {
            _section = OutputParserSection.ToolCalls;
            _dsml = new DsmlCallsParser(_calls);
        }
        else
        {
            Flag(sink, $"Unexpected special token '{marker}' in {_section}");
            _lenient = true;
            EmitText(marker, sink);
        }
    }

    private void Flag(Action<ParsedEvent> sink, string reason)
    {
        _malformed = true;
        sink(new ParsedEvent(ParsedEventKind.Malformed, reason));
    }

    private void EndTurn(Action<ParsedEvent> sink)
    {
        _section = OutputParserSection.Done;
        _completed = true;
        sink(new ParsedEvent(ParsedEventKind.Stop));
    }
}
