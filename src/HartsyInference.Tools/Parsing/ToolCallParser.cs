using System.Text;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Tools.Parsing;

/// <summary>Incremental tool-call parser over decoded text deltas for one <see cref="ToolCallFormat"/>. Text outside a call span is forwarded as it arrives, holding back only the characters that may still begin a marker; a span is read until its JSON (or Gemma block) value balances, then becomes one or more <see cref="NativeToolCall"/>s with ids <c>call_0</c>, <c>call_1</c>, … per parser. Anything that does not resolve into a call (invalid JSON, no <c>name</c>, a closing tag before the value balanced, a span past the cap, an unterminated span at <see cref="Flush"/>) is forwarded as plain text, so model output never throws. The bare forms (<see cref="ToolCallMarker.Strict"/>) are held back only while they can still be a call: a JSON span must open with <c>"name"</c> and, when the parser knows the offered tool names, the call must name one of them (a bare <c>name{</c> opens only on a complete offered name); the tagged forms stay permissive so a mistyped tool name reaches the host as a call it can answer with an error.</summary>
/// <remarks>Runs on the decode thread inside the engine's slot lock: every operation is bounded by the delta length plus the marker length, with one string per call span and no allocation on the plain-text path when nothing is held.</remarks>
public sealed class ToolCallParser
{
    /// <summary>Default cap on one call span; a longer span is forwarded as text.</summary>
    public const int DefaultMaxSpanChars = 64 * 1024;

    private const int MaxIdentifierChars = 64;
    private const string GemmaCallPrefix = "call:";
    private const string ObjectProbe = "{\"name\"";
    private const string ArrayProbe = "[{\"name\"";

    private readonly ToolCallFormatRules _rules;
    private readonly int _maxSpanChars;
    private readonly StringBuilder _forward = new();
    private readonly StringBuilder _hold = new();
    private readonly StringBuilder _span = new();
    private readonly StringBuilder _name = new();
    private readonly HashSet<string>? _knownNames;
    private readonly string[]? _knownNameList;
    private List<NativeToolCall>? _calls;
    private State _state;
    private Phase _phase;
    private ToolCallPayload _payload;
    private string? _presetName;
    private int _payloadStart;
    private bool _lineStart = true;
    private bool _holdLineStart;
    private bool _holdIsIdentifier;
    private ToolCallMarker? _holdSingle;
    private bool _closeSeen;
    private bool _strictSpan;
    private string? _probe;
    private int _probeIndex;
    private JsonBalance _json;
    private GemmaBalance _gemma;
    private int _completed;
    private readonly string _idPrefix;

    /// <summary>Creates a parser for <paramref name="format"/>; <paramref name="knownTools"/> (the offered tool names) makes the bare forms resolve only to those names, null keeps them permissive; <paramref name="idPrefix"/> numbers the calls' ids.</summary>
    public ToolCallParser(ToolCallFormat format, int maxSpanChars = DefaultMaxSpanChars, IEnumerable<string>? knownTools = null, string idPrefix = ToolCallJson.IdPrefix)
        : this(ToolCallFormats.RulesFor(format), maxSpanChars, knownTools, idPrefix)
    {
    }

    /// <summary>Creates a parser over custom <paramref name="rules"/>; see the format overload for <paramref name="knownTools"/> and <paramref name="idPrefix"/>.</summary>
    public ToolCallParser(ToolCallFormatRules rules, int maxSpanChars = DefaultMaxSpanChars, IEnumerable<string>? knownTools = null, string idPrefix = ToolCallJson.IdPrefix)
    {
        _idPrefix = idPrefix ?? throw new ArgumentNullException(nameof(idPrefix));
        ArgumentNullException.ThrowIfNull(rules);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxSpanChars, 16);
        if (rules.Markers.Count == 0) throw new ArgumentException("Rules need at least one opening marker.", nameof(rules));
        foreach (ToolCallMarker marker in rules.Markers)
        {
            if (string.IsNullOrEmpty(marker.Text)) throw new ArgumentException("Marker text must be non-empty.", nameof(rules));
        }
        _rules = rules;
        _maxSpanChars = maxSpanChars;
        if (knownTools is not null)
        {
            _knownNames = new HashSet<string>(knownTools, StringComparer.Ordinal);
            _knownNameList = [.. _knownNames];
        }
    }

    /// <summary>The format being parsed.</summary>
    public ToolCallFormat Format => _rules.Format;

    /// <summary>Calls completed so far; also the index the next id is built from.</summary>
    public int CompletedCalls => _completed;

    /// <summary>True while a call span is being read (its text is held, not forwarded).</summary>
    public bool InCall => _state == State.Call;

    /// <summary>The offered tool names the bare forms are restricted to, or null when every name is accepted.</summary>
    public IReadOnlyCollection<string>? KnownTools => _knownNames;

    /// <summary>Feeds one delta and returns the text to forward plus any calls it completed.</summary>
    public ToolCallParseResult Push(string delta)
    {
        ArgumentNullException.ThrowIfNull(delta);
        bool heldBefore = _hold.Length > 0 || _state != State.Text;
        _forward.Clear();
        _calls = null;
        for (int i = 0; i < delta.Length; i++) Feed(delta[i]);
        bool untouched = !heldBefore && _calls is null && _state == State.Text && _hold.Length == 0 && _forward.Length == delta.Length;
        return new ToolCallParseResult(untouched ? delta : _forward.ToString(), _calls);
    }

    /// <summary>Ends the stream: held text and an unterminated span come back as plain text; the scan state resets while ids keep counting.</summary>
    public ToolCallParseResult Flush()
    {
        _forward.Clear();
        _calls = null;
        switch (_state)
        {
            case State.Text:
            case State.AfterCall:
                if (_hold.Length > 0) Forward(_hold.ToString());
                break;
            case State.Call:
                Forward(_span.ToString());
                break;
        }
        ResetScan();
        return new ToolCallParseResult(_forward.Length == 0 ? "" : _forward.ToString(), null);
    }

    /// <summary>Clears all state, including the id counter, for reuse on a new request.</summary>
    public void Reset()
    {
        ResetScan();
        _forward.Clear();
        _calls = null;
        _completed = 0;
    }

    private void ResetScan()
    {
        _hold.Clear();
        _span.Clear();
        _name.Clear();
        _presetName = null;
        _state = State.Text;
        _phase = Phase.Prefix;
        _lineStart = true;
        _holdIsIdentifier = false;
        _holdSingle = null;
    }

    private void Feed(char c)
    {
        switch (_state)
        {
            case State.Text:
                FeedText(c);
                break;
            case State.Call:
                FeedCall(c);
                break;
            default:
                FeedAfterCall(c);
                break;
        }
    }

    private void FeedText(char c)
    {
        if (_hold.Length > 0)
        {
            ExtendHold(c);
            return;
        }
        if (TryBeginHold(c)) return;
        Forward(c);
    }

    private bool TryBeginHold(char c)
    {
        if (_rules.NamedFormAtLineStart && _lineStart && IsIdentifierStart(c))
        {
            // Each line's first word may open name{…}; known names end the hold once no offered name matches.
            _hold.Append(c);
            if (_knownNameList is null || PrefixesKnownName())
            {
                _holdIsIdentifier = true;
                _holdLineStart = _lineStart;
                _holdSingle = null;
                return true;
            }
            _hold.Clear();
        }
        ToolCallMarker? single = null;
        bool longer = false;
        IReadOnlyList<ToolCallMarker> markers = _rules.Markers;
        for (int i = 0; i < markers.Count; i++)
        {
            ToolCallMarker marker = markers[i];
            if (marker.LineStartOnly && !_lineStart) continue;
            if (marker.Text[0] != c) continue;
            if (marker.Text.Length == 1) single ??= marker;
            else longer = true;
        }
        if (!longer)
        {
            if (single is not { } immediate) return false;
            StartCall(immediate);
            return true;
        }
        // A longer marker shares the first character: hold, and fall back to the one-character marker on divergence.
        _holdIsIdentifier = false;
        _holdLineStart = _lineStart;
        _holdSingle = single;
        _hold.Append(c);
        return true;
    }

    private void ExtendHold(char c)
    {
        if (_holdIsIdentifier)
        {
            if (c == '{')
            {
                string name = _hold.ToString();
                // A mere prefix of an offered name would otherwise hold the text after it until its braces balance.
                if (_knownNames is not null && !_knownNames.Contains(name))
                {
                    ReleaseHold();
                    Feed(c);
                    return;
                }
                _hold.Clear();
                _holdIsIdentifier = false;
                StartNamedCall(name, ToolCallPayload.JsonObject);
                FeedCall(c);
                return;
            }
            if (IsIdentifierChar(c) && _hold.Length < MaxIdentifierChars)
            {
                _hold.Append(c);
                if (_knownNameList is null || PrefixesKnownName()) return;
                ReleaseHold();
                return;
            }
            ReleaseHold();
            Feed(c);
            return;
        }
        _hold.Append(c);
        bool prefix = false;
        IReadOnlyList<ToolCallMarker> markers = _rules.Markers;
        for (int i = 0; i < markers.Count; i++)
        {
            ToolCallMarker marker = markers[i];
            if (marker.LineStartOnly && !_holdLineStart) continue;
            if (marker.Text.Length < _hold.Length || !StartsWith(marker.Text, _hold)) continue;
            if (marker.Text.Length == _hold.Length)
            {
                _hold.Clear();
                _holdSingle = null;
                StartCall(marker);
                return;
            }
            prefix = true;
        }
        if (!prefix) ReleaseHold();
    }

    /// <summary>The held text can no longer become a longer marker: its first character opens the one-character marker it also matched, or is plain text; the rest is rescanned.</summary>
    private void ReleaseHold()
    {
        string held = _hold.ToString();
        ToolCallMarker? single = _holdSingle;
        _hold.Clear();
        _holdIsIdentifier = false;
        _holdSingle = null;
        if (single is { } marker) StartCall(marker);
        else Forward(held[0]);
        for (int i = 1; i < held.Length; i++) Feed(held[i]);
    }

    private void StartCall(ToolCallMarker marker)
    {
        BeginSpan(marker.Payload, presetName: null);
        _strictSpan = marker.Strict;
        // A bare JSON span is only worth holding while it still opens with "name": code and JSON answers are released
        // at their first key instead of at the balancing brace.
        if (marker.Strict && marker.Payload == ToolCallPayload.JsonObject) _probe = ObjectProbe;
        else if (marker.Strict && marker.Payload == ToolCallPayload.JsonArray) _probe = ArrayProbe;
        if (marker.TextIsPayload)
        {
            for (int i = 0; i < marker.Text.Length; i++) FeedCall(marker.Text[i]);
        }
        else
        {
            _span.Append(marker.Text);
        }
    }

    private void StartNamedCall(string name, ToolCallPayload payload)
    {
        BeginSpan(payload, name);
        _strictSpan = true;
        _span.Append(name);
    }

    private void BeginSpan(ToolCallPayload payload, string? presetName)
    {
        _state = State.Call;
        _phase = Phase.Prefix;
        _payload = payload;
        _presetName = presetName;
        _strictSpan = false;
        _probe = null;
        _probeIndex = 0;
        _span.Clear();
        _name.Clear();
        _json = default;
        _gemma = default;
    }

    private void FeedCall(char c)
    {
        _span.Append(c);
        if (_span.Length > _maxSpanChars)
        {
            Abort();
            return;
        }
        if (_phase == Phase.Prefix)
        {
            FeedPrefix(c);
            return;
        }
        if (_probe is not null && !ProbeChar(c))
        {
            Abort();
            return;
        }
        bool inString;
        bool complete;
        if (_payload == ToolCallPayload.GemmaCall)
        {
            _gemma.Feed(c);
            inString = _gemma.InString;
            complete = _gemma.Complete;
        }
        else
        {
            _json.Feed(c);
            inString = _json.InString;
            complete = _json.Complete;
        }
        if (_rules.CloseMarker is { } close && !inString && c == close[^1] && EndsWith(_span, close))
        {
            Abort();
            return;
        }
        if (complete) Complete();
    }

    private void FeedPrefix(char c)
    {
        switch (_payload)
        {
            case ToolCallPayload.JsonObject:
                if (_presetName is null && char.IsWhiteSpace(c)) return;
                if (c == '{') BeginValue(c);
                else Abort();
                return;
            case ToolCallPayload.JsonArray:
                if (char.IsWhiteSpace(c)) return;
                if (c == '[') BeginValue(c);
                else Abort();
                return;
            case ToolCallPayload.JsonAny:
                if (_name.Length == 0)
                {
                    if (char.IsWhiteSpace(c)) return;
                    if (c is '[' or '{') BeginValue(c);
                    else if (IsIdentifierStart(c)) _name.Append(c);
                    else Abort();
                    return;
                }
                if (c == '{')
                {
                    _presetName = _name.ToString();
                    BeginValue(c);
                }
                else if (IsIdentifierChar(c) && _name.Length < MaxIdentifierChars) _name.Append(c);
                else Abort();
                return;
            default:
                FeedGemmaPrefix(c);
                return;
        }
    }

    /// <summary>Gemma's prefix is <c>[call:]name</c> up to the block's opening brace, with the marker's own <c>call:</c> already consumed when the model emitted it.</summary>
    private void FeedGemmaPrefix(char c)
    {
        if (_name.Length == 0 && char.IsWhiteSpace(c)) return;
        if (c == '{')
        {
            string name = _name.ToString();
            if (name.StartsWith(GemmaCallPrefix, StringComparison.Ordinal)) name = name[GemmaCallPrefix.Length..];
            if (!IsIdentifier(name) || (_strictSpan && _knownNames is not null && !_knownNames.Contains(name)))
            {
                Abort();
                return;
            }
            _presetName = name;
            BeginValue(c);
            return;
        }
        if ((IsIdentifierChar(c) || c == ':') && _name.Length < MaxIdentifierChars + GemmaCallPrefix.Length) _name.Append(c);
        else Abort();
    }

    private void BeginValue(char first)
    {
        _phase = Phase.Value;
        _payloadStart = _span.Length - 1;
        if (_probe is not null && !ProbeChar(first))
        {
            Abort();
            return;
        }
        if (_payload == ToolCallPayload.GemmaCall) _gemma.Feed(first);
        else _json.Feed(first);
    }

    /// <summary>Advances the opening-key probe; whitespace is skipped between the tokens before the key but not inside the quoted key, and a mismatch means the span is not a call.</summary>
    private bool ProbeChar(char c)
    {
        if (char.IsWhiteSpace(c) && _probeIndex <= _probe!.IndexOf('"')) return true;
        if (c != _probe![_probeIndex]) return false;
        if (++_probeIndex == _probe.Length) _probe = null;
        return true;
    }

    private bool PrefixesKnownName()
    {
        foreach (string name in _knownNameList!)
        {
            if (name.Length >= _hold.Length && StartsWith(name, _hold)) return true;
        }
        return false;
    }

    private bool AllKnown(List<NativeToolCall> calls)
    {
        if (_knownNames is null || !_strictSpan) return true;
        foreach (NativeToolCall call in calls)
        {
            if (!_knownNames.Contains(call.Name)) return false;
        }
        return true;
    }

    private void Complete()
    {
        string value = _span.ToString(_payloadStart, _span.Length - _payloadStart);
        List<NativeToolCall> found = new(1);
        bool ok = _payload == ToolCallPayload.GemmaCall
            ? TryConvertGemma(value, found)
            : ToolCallJson.TryParse(value, _rules.ArgumentKeys, _presetName, found, _completed, _idPrefix);
        if (!ok || found.Count == 0 || !AllKnown(found))
        {
            Abort();
            return;
        }
        (_calls ??= new List<NativeToolCall>(found.Count)).AddRange(found);
        _completed += found.Count;
        _span.Clear();
        _name.Clear();
        _presetName = null;
        _phase = Phase.Prefix;
        _lineStart = true;
        _closeSeen = false;
        _state = State.AfterCall;
    }

    private bool TryConvertGemma(string block, List<NativeToolCall> into)
    {
        if (!GemmaCallDsl.TryConvert(block, out string json)) return false;
        into.Add(new NativeToolCall { Id = ToolCallJson.IdFor(_idPrefix, _completed), Name = _presetName!, Arguments = json });
        return true;
    }

    /// <summary>The span is not a call: everything read for it is plain text.</summary>
    private void Abort()
    {
        string text = _span.ToString();
        _span.Clear();
        _name.Clear();
        _presetName = null;
        _phase = Phase.Prefix;
        _state = State.Text;
        Forward(text);
    }

    /// <summary>After a call: the whitespace around it and the format's closing marker (once) are consumed; anything else resumes plain scanning at line-start.</summary>
    private void FeedAfterCall(char c)
    {
        string? close = _closeSeen ? null : _rules.CloseMarker;
        if (_hold.Length == 0)
        {
            if (char.IsWhiteSpace(c)) return;
            if (close is null || c != close[0])
            {
                _state = State.Text;
                FeedText(c);
                return;
            }
            if (close.Length == 1)
            {
                _closeSeen = true;
                return;
            }
            _holdIsIdentifier = false;
            _holdLineStart = true;
            _holdSingle = null;
            _hold.Append(c);
            return;
        }
        _hold.Append(c);
        if (!StartsWith(close!, _hold))
        {
            string held = _hold.ToString();
            _hold.Clear();
            _state = State.Text;
            for (int i = 0; i < held.Length; i++) Feed(held[i]);
            return;
        }
        if (_hold.Length == close!.Length)
        {
            _hold.Clear();
            _closeSeen = true;
        }
    }

    private void Forward(char c)
    {
        _forward.Append(c);
        _lineStart = IsLineBreak(c) || (_lineStart && IsBlank(c));
    }

    private void Forward(string text)
    {
        _forward.Append(text);
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            _lineStart = IsLineBreak(c) || (_lineStart && IsBlank(c));
        }
    }

    private static bool IsLineBreak(char c) => c is '\n' or '\r';

    private static bool IsBlank(char c) => c is ' ' or '\t';

    private static bool StartsWith(string text, StringBuilder prefix)
    {
        if (prefix.Length > text.Length) return false;
        for (int i = 0; i < prefix.Length; i++)
        {
            if (text[i] != prefix[i]) return false;
        }
        return true;
    }

    private static bool EndsWith(StringBuilder text, string suffix)
    {
        if (suffix.Length > text.Length) return false;
        int offset = text.Length - suffix.Length;
        for (int i = 0; i < suffix.Length; i++)
        {
            if (text[offset + i] != suffix[i]) return false;
        }
        return true;
    }

    private static bool IsIdentifierStart(char c) => char.IsLetter(c) || c == '_';

    private static bool IsIdentifierChar(char c) => char.IsLetterOrDigit(c) || c is '_' or '-' or '.';

    private static bool IsIdentifier(string s)
    {
        if (s.Length == 0 || !IsIdentifierStart(s[0])) return false;
        for (int i = 1; i < s.Length; i++)
        {
            if (!IsIdentifierChar(s[i])) return false;
        }
        return true;
    }

    private enum State
    {
        Text,
        Call,
        AfterCall,
    }

    private enum Phase
    {
        Prefix,
        Value,
    }

    /// <summary>Tracks brace/bracket depth through a JSON value, ignoring structure inside strings.</summary>
    private struct JsonBalance
    {
        private int _depth;
        private bool _escape;

        public bool InString { get; private set; }

        public bool Complete { get; private set; }

        public void Feed(char c)
        {
            if (Complete) return;
            if (InString)
            {
                if (_escape) _escape = false;
                else if (c == '\\') _escape = true;
                else if (c == '"') InString = false;
                return;
            }
            switch (c)
            {
                case '"':
                    InString = true;
                    break;
                case '{':
                case '[':
                    _depth++;
                    break;
                case '}':
                case ']':
                    _depth--;
                    if (_depth <= 0) Complete = true;
                    break;
            }
        }
    }

    /// <summary>Tracks brace/bracket depth through a Gemma block, where strings are delimited by <see cref="GemmaCallDsl.Quote"/>.</summary>
    private struct GemmaBalance
    {
        private int _depth;
        private int _quoteMatch;

        public bool InString { get; private set; }

        public bool Complete { get; private set; }

        public void Feed(char c)
        {
            if (Complete) return;
            string quote = GemmaCallDsl.Quote;
            if (c == quote[_quoteMatch])
            {
                if (++_quoteMatch == quote.Length)
                {
                    _quoteMatch = 0;
                    InString = !InString;
                }
                return;
            }
            if (_quoteMatch > 0)
            {
                _quoteMatch = c == quote[0] ? 1 : 0;
                if (_quoteMatch == 1) return;
            }
            if (InString) return;
            switch (c)
            {
                case '{':
                case '[':
                    _depth++;
                    break;
                case '}':
                case ']':
                    _depth--;
                    if (_depth <= 0) Complete = true;
                    break;
            }
        }
    }
}
