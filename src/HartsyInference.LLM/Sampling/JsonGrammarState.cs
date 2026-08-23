namespace HartsyInference.LLM.Sampling;

/// <summary>Incremental (streaming) validator for RFC 8259 JSON syntax, tracking exactly enough state to answer "is the text so far still a valid PREFIX of some valid JSON document" without re-parsing; used by <see cref="JsonGrammarStep"/> to test candidate tokens cheaply since <see cref="Clone"/> is a shallow stack copy, not a re-parse (mirrors how llama.cpp's GBNF grammar sampler works, specialized to "any valid JSON").</summary>
/// <remarks>Whitespace (space/tab/newline/CR) is accepted between structural tokens per RFC 8259 (not inside strings, where it's ordinary string content).</remarks>
public sealed class JsonGrammarState
{
    private enum Expect
    {
        ValueStart,           // top-level: haven't started the root value yet
        ObjectKeyOrEnd,       // just saw '{': expect '"' (start of a key) or '}'
        ObjectKey,            // just saw ',' inside an object: expect '"' (start of the next key)
        ObjectColon,          // just finished a key string: expect ':'
        ObjectValue,          // just saw ':': expect a value
        ObjectCommaOrEnd,     // just finished a value inside an object: expect ',' or '}'
        ArrayValueOrEnd,      // just saw '[': expect a value or ']'
        ArrayValue,           // just saw ',' inside an array: expect a value
        ArrayCommaOrEnd,      // just finished a value inside an array: expect ',' or ']'
        Done,                 // root value is complete; only trailing whitespace (or EOS) is valid from here
    }

    private enum NumPhase { Sign, LeadingZero, IntDigits, FracStart, FracDigits, ExpStart, ExpSignSeen, ExpDigits }

    private enum ContainerKind { Object, Array }

    private readonly List<ContainerKind> _stack; // List, not Stack<T>, so Clone() is one array copy
    private Expect _expect;

    // Sub-parser state for the token currently in progress (at most one of these is active at a time).
    private bool _inString;
    private bool _stringEscaped;
    private int _stringUnicodeRemaining; // remaining hex digits of a \uXXXX escape
    private bool _inNumber;
    private NumPhase _numPhase;
    private string? _literalTarget; // "true" | "false" | "null" while mid-match
    private int _literalPos;

    public JsonGrammarState()
    {
        _stack = [];
        _expect = Expect.ValueStart;
    }

    private JsonGrammarState(JsonGrammarState src)
    {
        _stack = [.. src._stack];
        _expect = src._expect;
        _inString = src._inString;
        _stringEscaped = src._stringEscaped;
        _stringUnicodeRemaining = src._stringUnicodeRemaining;
        _inNumber = src._inNumber;
        _numPhase = src._numPhase;
        _literalTarget = src._literalTarget;
        _literalPos = src._literalPos;
        _pendingAfterContainer = [.. src._pendingAfterContainer];
        _pendingAfterScalar = src._pendingAfterScalar;
    }

    /// <summary>Cheap copy (no re-parsing) so a candidate token can be trial-fed without disturbing the real state — the caller clones, feeds the candidate's text into the clone, and keeps the clone only if the real token is actually chosen.</summary>
    public JsonGrammarState Clone() => new(this);

    /// <summary>True once the root value is fully formed. Numbers (unlike strings/literals) have no explicit terminator character in JSON — "123" is complete the instant input stops — so a number in progress still counts as complete here PROVIDED its digits already form a valid number on their own (not mid-sign/mid-fraction-start/mid-exponent-start) AND it was the root value; <see cref="_pendingAfterScalar"/> already encodes what finishing this scalar would transition to. Strings always need an explicit closing quote and literals ("true"/"false"/"null") only stop existing (<see cref="_literalTarget"/> goes null) at the instant they fully match, so neither has this ambiguity.</summary>
    public bool IsComplete
    {
        get
        {
            if (_inString || _literalTarget is not null) return false;
            if (_inNumber)
            {
                bool validNumberEnd = _numPhase is NumPhase.LeadingZero or NumPhase.IntDigits or NumPhase.FracDigits or NumPhase.ExpDigits;
                return validNumberEnd && _pendingAfterScalar == Expect.Done;
            }
            return _expect == Expect.Done;
        }
    }

    /// <summary>Feeds every character of <paramref name="text"/> in order; returns false (state left unspecified — caller must discard this instance) the moment any character is invalid.</summary>
    public bool TryFeed(string text)
    {
        foreach (char c in text)
        {
            if (!TryFeedChar(c)) return false;
        }
        return true;
    }

    private bool TryFeedChar(char c)
    {
        if (_inString) return FeedString(c);
        if (_inNumber) return FeedNumber(c);
        if (_literalTarget is not null) return FeedLiteral(c);

        // Whitespace is valid between structural tokens everywhere except where a value/key must START
        // (matched below) — it's always safe to accept it while we're between tokens.
        if (c is ' ' or '\t' or '\n' or '\r')
        {
            // Only meaningful once something has started (RFC 8259 allows leading/trailing/inter-token ws);
            // accepting it at ValueStart too is harmless (leading whitespace before the root value).
            return true;
        }

        switch (_expect)
        {
            case Expect.ValueStart: return StartValue(c, afterValueIsDone: true);
            case Expect.ObjectValue: return StartValue(c, afterValueIsDone: false, onCompleteExpect: Expect.ObjectCommaOrEnd);
            case Expect.ArrayValueOrEnd:
                if (c == ']') { Pop(); return true; }
                return StartValue(c, afterValueIsDone: false, onCompleteExpect: Expect.ArrayCommaOrEnd);
            case Expect.ArrayValue: return StartValue(c, afterValueIsDone: false, onCompleteExpect: Expect.ArrayCommaOrEnd);

            case Expect.ObjectKeyOrEnd:
                if (c == '}') { Pop(); return true; }
                if (c == '"') { BeginString(); _pendingAfterScalar = Expect.ObjectColon; return true; }
                return false;
            case Expect.ObjectKey:
                if (c == '"') { BeginString(); _pendingAfterScalar = Expect.ObjectColon; return true; }
                return false;
            case Expect.ObjectColon:
                if (c == ':') { _expect = Expect.ObjectValue; return true; }
                return false;
            case Expect.ObjectCommaOrEnd:
                if (c == ',') { _expect = Expect.ObjectKey; return true; }
                if (c == '}') { Pop(); return true; }
                return false;
            case Expect.ArrayCommaOrEnd:
                if (c == ',') { _expect = Expect.ArrayValue; return true; }
                if (c == ']') { Pop(); return true; }
                return false;

            case Expect.Done:
                return false; // any non-whitespace after the root value completes is invalid JSON

            default:
                return false;
        }
    }

    /// <summary>Starts parsing a value (object/array/string/number/true/false/null) from its first character; <paramref name="onCompleteExpect"/> is what to transition to once this value finishes (ignored — replaced with Done — when <paramref name="afterValueIsDone"/>, i.e. this is the root value).</summary>
    private bool StartValue(char c, bool afterValueIsDone, Expect onCompleteExpect = Expect.Done)
    {
        Expect after = afterValueIsDone ? Expect.Done : onCompleteExpect;
        switch (c)
        {
            case '{':
                _stack.Add(ContainerKind.Object);
                _pendingAfterContainer.Add(after);
                _expect = Expect.ObjectKeyOrEnd;
                return true;
            case '[':
                _stack.Add(ContainerKind.Array);
                _pendingAfterContainer.Add(after);
                _expect = Expect.ArrayValueOrEnd;
                return true;
            case '"':
                BeginString();
                _pendingAfterScalar = after;
                return true;
            case '-':
                _inNumber = true; _numPhase = NumPhase.Sign; // must see a digit next
                _pendingAfterScalar = after;
                return true;
            case '0':
                _inNumber = true; _numPhase = NumPhase.LeadingZero; // no more int digits may follow (no "01")
                _pendingAfterScalar = after;
                return true;
            case >= '1' and <= '9':
                _inNumber = true; _numPhase = NumPhase.IntDigits;
                _pendingAfterScalar = after;
                return true;
            case 't': _literalTarget = "true"; _literalPos = 1; _pendingAfterScalar = after; return true;
            case 'f': _literalTarget = "false"; _literalPos = 1; _pendingAfterScalar = after; return true;
            case 'n': _literalTarget = "null"; _literalPos = 1; _pendingAfterScalar = after; return true;
            default:
                return false;
        }
    }

    // Parallel to _stack: what Expect to restore when the container at the same depth closes. Kept as a
    // separate list (not embedded in ContainerKind) to avoid a tuple-list churn on the hot Clone() path.
    private readonly List<Expect> _pendingAfterContainer = [];
    // What to transition to once the CURRENT scalar (string/number/literal) finishes — only meaningful while
    // _inString/_inNumber/_literalTarget is active.
    private Expect _pendingAfterScalar;

    private void BeginString() { _inString = true; _stringEscaped = false; _stringUnicodeRemaining = 0; }

    private bool FeedString(char c)
    {
        if (_stringUnicodeRemaining > 0)
        {
            bool hex = c is >= '0' and <= '9' or >= 'a' and <= 'f' or >= 'A' and <= 'F';
            if (!hex) return false;
            _stringUnicodeRemaining--;
            return true;
        }
        if (_stringEscaped)
        {
            _stringEscaped = false;
            return c switch
            {
                '"' or '\\' or '/' or 'b' or 'f' or 'n' or 'r' or 't' => true,
                'u' => Begin4HexEscape(),
                _ => false,
            };
        }
        if (c == '\\') { _stringEscaped = true; return true; }
        if (c == '"') { _inString = false; return CompleteScalar(); }
        // Per RFC 8259, raw control characters (U+0000–U+001F) are NOT allowed unescaped in a JSON string.
        if (c < 0x20) return false;
        return true;
    }

    private bool Begin4HexEscape() { _stringUnicodeRemaining = 4; return true; }

    private bool FeedNumber(char c)
    {
        switch (_numPhase)
        {
            case NumPhase.Sign: // just saw '-', need the first digit
                if (c == '0') { _numPhase = NumPhase.LeadingZero; return true; }
                if (c is >= '1' and <= '9') { _numPhase = NumPhase.IntDigits; return true; }
                return false;
            case NumPhase.LeadingZero: // int part is "0" — no more int digits allowed, only '.' / 'e' / end
                return FeedNumberContinuation(c);
            case NumPhase.IntDigits:
                if (c is >= '0' and <= '9') return true;
                return FeedNumberContinuation(c);
            case NumPhase.FracStart:
                if (c is >= '0' and <= '9') { _numPhase = NumPhase.FracDigits; return true; }
                return false;
            case NumPhase.FracDigits:
                if (c is >= '0' and <= '9') return true;
                return FeedNumberContinuation(c);
            case NumPhase.ExpStart:
                if (c is '+' or '-') { _numPhase = NumPhase.ExpSignSeen; return true; }
                if (c is >= '0' and <= '9') { _numPhase = NumPhase.ExpDigits; return true; }
                return false;
            case NumPhase.ExpSignSeen:
                if (c is >= '0' and <= '9') { _numPhase = NumPhase.ExpDigits; return true; }
                return false;
            case NumPhase.ExpDigits:
                if (c is >= '0' and <= '9') return true;
                return FeedNumberContinuation(c);
            default:
                return false;
        }
    }

    /// <summary>Called on a non-digit while in a "could extend or could end here" number phase: try to extend into '.'/'e', otherwise treat the number as finished and re-dispatch <paramref name="c"/> as the first character AFTER the number.</summary>
    private bool FeedNumberContinuation(char c)
    {
        if (c == '.') { _numPhase = NumPhase.FracStart; return true; }
        if (c is 'e' or 'E') { _numPhase = NumPhase.ExpStart; return true; }
        _inNumber = false;
        if (!CompleteScalar()) return false;
        return TryFeedChar(c);
    }

    private bool FeedLiteral(char c)
    {
        if (c != _literalTarget![_literalPos]) return false;
        _literalPos++;
        if (_literalPos < _literalTarget.Length) return true;
        _literalTarget = null;
        return CompleteScalar();
    }

    /// <summary>Called when a string/number/literal value just finished — transitions to whatever was pending before it started (root-done, or back into the enclosing object/array).</summary>
    private bool CompleteScalar()
    {
        _expect = _pendingAfterScalar;
        return true;
    }

    private void Pop()
    {
        _stack.RemoveAt(_stack.Count - 1);
        Expect after = _pendingAfterContainer[^1];
        _pendingAfterContainer.RemoveAt(_pendingAfterContainer.Count - 1);
        _expect = after;
    }
}
