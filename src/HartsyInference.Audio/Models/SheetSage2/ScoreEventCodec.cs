namespace HartsyInference.Audio.Models.SheetSage2;

/// <summary>Reads a decoded token stream as musical events, and writes events back as tokens.
///
/// <para>The stream is a run of subbeat shifts followed by that position's fields, repeated. Shifts accumulate,
/// so a position is the running total rather than an absolute — which is why a window's events have to be
/// re-based before they can be replayed as context for the next one.</para>
///
/// <para><b>Re-encoding is canonical, not byte-exact.</b> Decoding keeps only the running total, so the shape of
/// a shift run is lost: the released implementation collapses <c>shift(100) shift(100)</c> to a single
/// <c>shift(200)</c> and drops a leading <c>shift(0)</c>, and this port reproduces that. Field tokens ARE
/// preserved verbatim — that is what <see cref="ScoreEvent.TokensByField"/> is for — so a replayed prefix
/// differs from the original only in how it spells the gaps between events, and only when the model wrote a
/// gap non-canonically.</para></summary>
public sealed class ScoreEventCodec(ScoreTokenizer tokenizer)
{
    /// <summary>Which event field each token vocabulary contributes to.</summary>
    private static readonly Dictionary<string, string> FieldOfTokenType = new(StringComparer.Ordinal)
    {
        ["time"] = "timestamp",
        ["meter"] = "rhythm",
        ["eighth_position"] = "rhythm",
        ["structure"] = "structure",
        ["key"] = "key",
        ["chord_full"] = "chord",
        ["pitch"] = "melody",
        ["duration"] = "melody",
    };

    private readonly ScoreTokenizer _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));

    /// <summary>Reads everything after the prompt prefix up to the end token.</summary>
    /// <remarks>A trailing run of shifts carrying no fields is dropped, as in the released implementation. That
    /// happens when a window stops mid-stream at its <see cref="SheetSage2Window.GenerationStop"/>, and it is
    /// safe to lose: those shifts sit past everything the window is trusted for, and the next window's prefix is
    /// selected by timestamp, which a bare shift does not carry.</remarks>
    /// <exception cref="ArgumentException">The stream has no prompt prefix, or an event has no beat position.</exception>
    public List<ScoreEvent> DecodeSequence(IReadOnlyList<int> tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        int position = IndexOf(tokens, ScoreTokenizer.OutToken);
        if (position < 0)
        {
            throw new ArgumentException("SheetSage2's token stream has no prompt prefix to read past.", nameof(tokens));
        }
        position++;
        int subbeat = 0;
        List<ScoreEvent> events = [];
        while (position < tokens.Count && tokens[position] != ScoreTokenizer.EosToken)
        {
            if (_tokenizer.TokenType(tokens[position]) != "subbeat_shift")
            {
                throw new ArgumentException("SheetSage2 produced an event without a beat position.", nameof(tokens));
            }
            while (position < tokens.Count && _tokenizer.TokenType(tokens[position]) == "subbeat_shift")
            {
                subbeat += tokens[position] - _tokenizer.SubbeatShiftStart;
                position++;
            }
            Dictionary<string, List<int>> payload = new(StringComparer.Ordinal);
            while (position < tokens.Count)
            {
                string type = _tokenizer.TokenType(tokens[position]);
                if (type == "subbeat_shift" || type == "eos") break;
                if (!FieldOfTokenType.TryGetValue(type, out string? field))
                {
                    throw new ArgumentException($"SheetSage2 produced a '{type}' token inside an event.", nameof(tokens));
                }
                if (!payload.TryGetValue(field, out List<int>? bucket))
                {
                    payload[field] = bucket = [];
                }
                bucket.Add(tokens[position]);
                position++;
            }
            if (payload.Count > 0) events.Add(BuildEvent(subbeat, payload));
        }
        return events;
    }

    /// <summary>Writes events back as a token stream, opening with the prompt prefix.</summary>
    /// <remarks><para>Field order follows <see cref="ScoreTokenizer.EventFields"/>, and a gap wider than one
    /// shift can express is written as several of the largest step. A gap past four of them exceeds what the
    /// decode grammar would allow itself to write, but the released encoder emits it anyway and a replayed
    /// prefix is not subject to the grammar, so it is reproduced rather than capped.</para>
    /// <para>Events must be ordered and re-based to non-negative positions. The released encoder does neither
    /// check and turns a backwards position into a token id outside the shift range — silently, since nothing
    /// downstream validates it.</para></remarks>
    /// <exception cref="ArgumentException">Events move backwards, so a shift cannot be written.</exception>
    public List<int> EncodeEvents(IReadOnlyList<ScoreEvent> events)
    {
        ArgumentNullException.ThrowIfNull(events);
        // The largest step one shift token can name. Derived, because re-stating it would let it drift from the
        // id used to write it and silently mis-total every gap after the range moved.
        int maxStep = _tokenizer.SubbeatShiftEnd - _tokenizer.SubbeatShiftStart - 1;
        List<int> tokens = [.. ScoreTokenizer.PromptPrefix()];
        int previous = 0;
        foreach (ScoreEvent item in events)
        {
            int shift = item.Subbeat - previous;
            if (shift < 0)
            {
                throw new ArgumentException(
                    $"A SheetSage2 event at subbeat {item.Subbeat} moves back from {previous}; events must be "
                    + "ordered and re-based to non-negative positions before they can be encoded.", nameof(events));
            }
            while (shift > maxStep)
            {
                tokens.Add(_tokenizer.SubbeatShiftEnd - 1);
                shift -= maxStep;
            }
            tokens.Add(_tokenizer.SubbeatShiftStart + shift);
            previous = item.Subbeat;
            foreach (string field in ScoreTokenizer.EventFields)
            {
                if (item.TokensByField.TryGetValue(field, out List<int>? bucket)) tokens.AddRange(bucket);
            }
        }
        return tokens;
    }

    private ScoreEvent BuildEvent(int subbeat, Dictionary<string, List<int>> payload)
    {
        return new ScoreEvent
        {
            Subbeat = subbeat,
            TokensByField = payload,
            Timestamp = payload.TryGetValue("timestamp", out List<int>? time) ? _tokenizer.TokenToSeconds(time[0]) : null,
            Rhythm = payload.TryGetValue("rhythm", out List<int>? rhythm) ? ReadRhythm(rhythm) : null,
            Structure = payload.TryGetValue("structure", out List<int>? structure)
                ? ScoreTokenizer.StructureLabels[structure[0] - _tokenizer.StructureStart] : null,
            Key = payload.TryGetValue("key", out List<int>? key) ? ReadKey(key[0]) : null,
            Chord = payload.TryGetValue("chord", out List<int>? chord)
                ? _tokenizer.FullChordLabels[chord[0] - _tokenizer.FullChordStart] : null,
            Melody = payload.TryGetValue("melody", out List<int>? melody) ? ReadMelody(melody) : null,
        };
    }

    private ScoreRhythm ReadRhythm(List<int> tokens)
    {
        (int, int)? meter = null;
        int? eighth = null;
        foreach (int token in tokens)
        {
            if (_tokenizer.TokenType(token) == "meter") meter = _tokenizer.MeterPairs[token - _tokenizer.MeterStart];
            else eighth = token - _tokenizer.EighthPositionStart;
        }
        return new ScoreRhythm(meter, eighth);
    }

    private string ReadKey(int token)
    {
        int index = token - _tokenizer.KeyStart;
        return $"{ScoreTokenizer.ChromaticSharps[index % 12]}:{(index >= 12 ? "minor" : "major")}";
    }

    /// <summary>A melody field is a run of pitches, each optionally followed by its length.</summary>
    /// <remarks>The pitch slot is type-checked rather than read blind: the released implementation reads a
    /// stray duration token there as pitch 256, which lands as a phantom low note on the instrumental line
    /// instead of an error. The grammar forbids the shape, but this method takes an arbitrary stream.</remarks>
    private List<ScoreNote> ReadMelody(List<int> tokens)
    {
        List<ScoreNote> notes = [];
        int index = 0;
        while (index < tokens.Count)
        {
            if (_tokenizer.TokenType(tokens[index]) != "pitch")
            {
                throw new ArgumentException(
                    "SheetSage2 wrote a note length with no pitch before it.", nameof(tokens));
            }
            int pitch = tokens[index] - _tokenizer.PitchStart;
            index++;
            int durationBin = 0;
            if (index < tokens.Count && _tokenizer.TokenType(tokens[index]) == "duration")
            {
                durationBin = tokens[index] - _tokenizer.DurationStart;
                index++;
            }
            // The pitch range holds both melody lines: the low 128 ids are one track, the high 128 the other.
            notes.Add(new ScoreNote(pitch % 128, pitch >= 128 ? 1 : 0, durationBin));
        }
        return notes;
    }

    private static int IndexOf(IReadOnlyList<int> tokens, int value)
    {
        for (int i = 0; i < tokens.Count; i++)
        {
            if (tokens[i] == value) return i;
        }
        return -1;
    }
}
