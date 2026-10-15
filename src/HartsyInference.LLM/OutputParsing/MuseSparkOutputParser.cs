using System.Buffers;
using System.Net;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.RegularExpressions;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.OutputParsing;

/// <summary>Streaming parser for Muse Spark turns: messages <c>&lt;|start|&gt;assistant to=R&lt;|message|&gt;…</c> ended by <c>&lt;|eom|&gt;</c>
/// or <c>&lt;|eot|&gt;</c>, where R <c>self</c> is reasoning, <c>user</c>/none/<c>commentary</c> is answer text and any other R is an
/// <c>&lt;atem:invoke&gt;</c> tool call typed by the tool's schema (llama.cpp <c>common/parsers/muse-spark.cpp</c>).</summary>
public sealed partial class MuseSparkOutputParser : IOutputParser
{
    private const string Start = "<|start|>";
    private const string Message = "<|message|>";
    private const string Eom = "<|eom|>";
    private const string Eot = "<|eot|>";
    private const string EndOfText = "<|end_of_text|>";
    private const string AssistantRole = "assistant";
    private const string RecipientPrefix = "to=";
    private const string AtemPrefix = "<atem:";
    private const string ReasoningHeader = Start + AssistantRole + " " + RecipientPrefix + "self" + Message;
    private const int MaxHeaderChars = 256;

    private static readonly string[] HeaderMarkers = [Message, Eom, Eot, EndOfText, Start];
    private static readonly string[] BodyMarkers = [Eom, Eot, EndOfText, Start];
    private static readonly string[] Literals = [Start, Message, Eom, Eot, EndOfText];

    // Arguments are API JSON, not HTML: non-ASCII text and quotes stay readable instead of becoming \u escapes.
    private static readonly JsonWriterOptions ArgumentWriterOptions = new() { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    private enum Channel { Reasoning, Content, Tool }

    private readonly IncrementalDetokenizer _detok;
    private readonly IReadOnlyDictionary<string, IReadOnlyDictionary<string, bool>> _stringParameters;
    private readonly Dictionary<int, string> _literalById = [];
    private readonly StringBuilder _reasoning = new();
    private readonly StringBuilder _content = new();
    private readonly StringBuilder _toolBody = new();
    private readonly List<ChatToolCall> _calls = [];
    private OutputParserSection _section = OutputParserSection.Content;
    private bool _inHeader = true;
    private bool _betweenMessages;
    private Channel _channel;
    private string _pending = "";
    private bool _malformed;
    private bool _completed;

    /// <summary>Creates a parser for a completion of a prompt that ended with <c>&lt;|start|&gt;assistant</c>; <paramref name="tools"/> are the
    /// functions that prompt offered, whose schemas decide which argument values are raw text and which are JSON.</summary>
    /// <exception cref="InvalidOperationException">The tokenizer lacks the Muse Spark control tokens.</exception>
    public MuseSparkOutputParser(ILlmTokenizer tokenizer, IReadOnlyList<ToolSpec>? tools = null)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        _detok = new IncrementalDetokenizer(tokenizer, includeSpecial: false);
        _stringParameters = ToolSchema.StringParameters(tools);
        foreach (string literal in Literals)
        {
            int id = tokenizer.SpecialId(literal)
                ?? throw new InvalidOperationException($"Tokenizer has no '{literal}' control token; not a Muse Spark vocabulary.");
            _literalById[id] = literal;
        }
    }

    /// <summary>True when <paramref name="template"/> renders Muse Spark's message format (its reasoning header is in the source) and
    /// <paramref name="tokenizer"/> carries the control tokens that frame it.</summary>
    public static bool Handles(IChatTemplate template, ILlmTokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(template);
        ArgumentNullException.ThrowIfNull(tokenizer);
        if (template.Source is not { } source || !source.Contains(ReasoningHeader, StringComparison.Ordinal)) return false;
        foreach (string literal in Literals)
        {
            if (tokenizer.SpecialId(literal) is null) return false;
        }
        return true;
    }

    /// <inheritdoc />
    public ParsedAssistant Result => new(_reasoning.ToString(), _content.ToString(), [.. _calls], _malformed, _completed);

    /// <inheritdoc />
    public void Push(int tokenId, Action<ParsedEvent> sink)
    {
        ArgumentNullException.ThrowIfNull(sink);
        if (_section == OutputParserSection.Done) return;
        string text = _literalById.TryGetValue(tokenId, out string? literal) ? _detok.Flush() + literal : _detok.Push(tokenId);
        Process(text, final: false, sink);
    }

    /// <inheritdoc />
    public void Finish(Action<ParsedEvent> sink)
    {
        ArgumentNullException.ThrowIfNull(sink);
        if (_section == OutputParserSection.Done) return;
        Process(_detok.Flush(), final: true, sink);
        if (_section == OutputParserSection.Done) return;
        if (!_inHeader && !_betweenMessages) EndMessage(sink);
        EndTurn(sink);
    }

    private void Process(string text, bool final, Action<ParsedEvent> sink)
    {
        _pending += text;
        while (_section != OutputParserSection.Done)
        {
            string[] markers = _inHeader ? HeaderMarkers : BodyMarkers;
            MarkerScanner.Scan(_pending, markers, out int at, out int which, out int hold);
            if (at >= 0 && (final || hold < 0 || at < hold))
            {
                string before = _pending[..at];
                string marker = markers[which];
                _pending = _pending[(at + marker.Length)..];
                OnMarker(before, marker, sink);
                continue;
            }
            if (_inHeader)
            {
                // A header waits for its <|message|>; one that runs on means the model left the format, so its text is shown, not lost.
                if (_pending.Length <= MaxHeaderChars) return;
                Flag(sink, "Message header has no <|message|>");
                _inHeader = false;
                _channel = Channel.Content;
                continue;
            }
            int cut = final || hold < 0 ? _pending.Length : hold;
            Emit(_pending[..cut], sink);
            _pending = _pending[cut..];
            return;
        }
        _pending = "";
    }

    private void OnMarker(string before, string marker, Action<ParsedEvent> sink)
    {
        if (_inHeader)
        {
            switch (marker)
            {
                case Message:
                    OpenMessage(before, sink);
                    return;
                case Start:
                    Flag(sink, $"Message header '{before}' restarted");
                    return;
                case Eom:
                    Flag(sink, $"Message header '{before}' has no body");
                    _inHeader = false;
                    _betweenMessages = true;
                    return;
                default:
                    if (before.Trim().Length > 0) Flag(sink, $"Turn ended inside message header '{before}'");
                    EndTurn(sink);
                    return;
            }
        }
        Emit(before, sink);
        switch (marker)
        {
            case Start:
                if (!_betweenMessages)
                {
                    Flag(sink, "Message not closed before the next <|start|>");
                    EndMessage(sink);
                }
                _betweenMessages = false;
                _inHeader = true;
                return;
            case Eom:
                if (_betweenMessages)
                {
                    Flag(sink, "<|eom|> outside a message");
                    return;
                }
                EndMessage(sink);
                _betweenMessages = true;
                return;
            default:
                if (!_betweenMessages) EndMessage(sink);
                EndTurn(sink);
                return;
        }
    }

    /// <summary>Starts the message whose header (the text between <c>&lt;|start|&gt;</c> and <c>&lt;|message|&gt;</c>) is <paramref name="header"/>. The
    /// first header holds only the recipient, since the prompt already wrote the role.</summary>
    private void OpenMessage(string header, Action<ParsedEvent> sink)
    {
        string h = header.Trim();
        if (h.StartsWith(AssistantRole, StringComparison.Ordinal)) h = h[AssistantRole.Length..].TrimStart();
        else if (h.Length > 0 && !h.StartsWith(RecipientPrefix, StringComparison.Ordinal))
        {
            // The model began another role's message (a user or tool turn of its own): the assistant turn is over.
            Flag(sink, $"Assistant turn ran into a '{h}' message");
            EndTurn(sink);
            return;
        }
        string recipient = h.StartsWith(RecipientPrefix, StringComparison.Ordinal) ? h[RecipientPrefix.Length..] : "";
        int space = recipient.IndexOfAny([' ', '\t']);
        if (space >= 0) recipient = recipient[..space];
        _channel = recipient switch
        {
            "self" => Channel.Reasoning,
            "" or "user" or "commentary" => Channel.Content,
            _ => Channel.Tool,
        };
        _inHeader = false;
        _toolBody.Clear();
    }

    /// <summary>Text of the current message: reasoning and content stream as they arrive, a tool message is held until it ends.</summary>
    private void Emit(string text, Action<ParsedEvent> sink)
    {
        if (text.Length == 0) return;
        if (_betweenMessages)
        {
            if (text.Trim().Length > 0) Flag(sink, $"Text outside a message: '{text}'");
            return;
        }
        switch (_channel)
        {
            case Channel.Reasoning:
                _reasoning.Append(text);
                sink(new ParsedEvent(ParsedEventKind.ReasoningDelta, text));
                break;
            case Channel.Content:
                AppendContent(text, sink);
                break;
            default:
                _toolBody.Append(text);
                break;
        }
    }

    private void AppendContent(string text, Action<ParsedEvent> sink)
    {
        _content.Append(text);
        sink(new ParsedEvent(ParsedEventKind.ContentDelta, text));
    }

    /// <summary>Closes the current message; a tool message's body becomes its calls. A body without any markup is shown as content; a
    /// call cut off before <c>&lt;/atem:invoke&gt;</c> (the length limit) is dropped.</summary>
    private void EndMessage(Action<ParsedEvent> sink)
    {
        if (_channel != Channel.Tool) return;
        string body = _toolBody.ToString();
        _toolBody.Clear();
        int found = 0;
        foreach (Match invoke in InvokeRegex().Matches(body))
        {
            int index = _calls.Count;
            string name = WebUtility.HtmlDecode(invoke.Groups[1].Value);
            string args = ArgumentsJson(name, invoke.Groups[2].Value);
            sink(new ParsedEvent(ParsedEventKind.ToolCallBegin, name, index));
            sink(new ParsedEvent(ParsedEventKind.ToolCallArgsDelta, args, index));
            sink(new ParsedEvent(ParsedEventKind.ToolCallEnd, null, index));
            _calls.Add(new ChatToolCall($"call_{index}", name, args));
            found++;
        }
        if (found > 0) return;
        Flag(sink, "Tool message without a complete <atem:invoke> block");
        if (!body.Contains(AtemPrefix, StringComparison.Ordinal)) AppendContent(body, sink);
    }

    /// <summary>The parameters of one invoke of <paramref name="tool"/> as a JSON object. A parameter whose schema admits a string keeps its text;
    /// any other is read as JSON. Without a schema for it, a value is JSON when it parses as such and text otherwise, since the template writes
    /// strings bare and every other value as JSON. A value that should be JSON but is not stays text.</summary>
    private string ArgumentsJson(string tool, string invokeBody)
    {
        IReadOnlyDictionary<string, bool>? schema = _stringParameters.GetValueOrDefault(tool);
        ArrayBufferWriter<byte> buffer = new();
        using (Utf8JsonWriter json = new(buffer, ArgumentWriterOptions))
        {
            json.WriteStartObject();
            foreach (Match p in ParameterRegex().Matches(invokeBody))
            {
                string name = WebUtility.HtmlDecode(p.Groups[1].Value);
                string raw = p.Groups[2].Value;
                json.WritePropertyName(name);
                bool? isString = schema is not null && schema.TryGetValue(name, out bool s) ? s : null;
                if (isString != true && TryParseJson(raw, strict: isString is null, out JsonDocument? doc))
                {
                    using (doc) doc.RootElement.WriteTo(json);
                }
                else
                {
                    json.WriteStringValue(raw);
                }
            }
            json.WriteEndObject();
        }
        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }

    /// <summary>Parses <paramref name="raw"/> as one JSON value; <paramref name="strict"/> (no schema to go by) refuses surrounding whitespace, which
    /// the template never writes around a JSON value, so such text stays a string.</summary>
    private static bool TryParseJson(string raw, bool strict, [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out JsonDocument? doc)
    {
        doc = null;
        if (raw.Trim().Length == 0 || (strict && raw.Trim().Length != raw.Length)) return false;
        try
        {
            doc = JsonDocument.Parse(raw);
            return true;
        }
        catch (JsonException)
        {
            return false;
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

    [GeneratedRegex("<atem:invoke\\s+name=\"([^\"]*)\">(.*?)</atem:invoke>", RegexOptions.Singleline | RegexOptions.CultureInvariant)]
    private static partial Regex InvokeRegex();

    [GeneratedRegex("<atem:parameter\\s+name=\"([^\"]*)\">(.*?)</atem:parameter>", RegexOptions.Singleline | RegexOptions.CultureInvariant)]
    private static partial Regex ParameterRegex();
}
