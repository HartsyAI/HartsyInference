namespace HartsyInference.Audio.Models.SheetSage2;

/// <summary>The grammar SheetSage2 decodes under: which tokens may follow what has been written so far.
///
/// <para>Decoding is greedy argmax over logits masked to this set, so the grammar is not a safety net around a
/// sampler — it decides the result. Two rules do the work. Fields within an event must appear in
/// <see cref="ScoreTokenizer.EventFields"/> order, so once a field is written only later ones stay open (melody
/// is the exception: repeated notes are legal, so it stays open at its own index). And two tokens demand a
/// partner immediately: a meter must be followed by its position in the bar, and a pitch by either its duration
/// or another pitch.</para>
///
/// <para>A run of subbeat shifts separates events; the fourth consecutive one is refused, which is what stops a
/// decode walking forward forever without writing anything.</para></summary>
public sealed class PromptGrammar(ScoreTokenizer tokenizer)
{
    /// <summary>What a token is still waiting on, when anything.</summary>
    private enum Pending
    {
        None,
        /// <summary>A meter was written; its position in the bar has to follow.</summary>
        RhythmAfterMeter,
        /// <summary>A pitch was written; a duration or another pitch has to follow.</summary>
        MelodyAfterPitch,
    }

    private readonly ScoreTokenizer _tokenizer = tokenizer;
    private bool _inShift = true;
    private int _shiftRun;
    private int _payloadCount;
    private int _lastFieldIndex = -1;
    private Pending _pending = Pending.None;

    /// <summary>Events completed so far.</summary>
    public int EventCount { get; private set; }

    /// <summary>Fills <paramref name="mask"/> with the tokens that may legally come next.</summary>
    /// <remarks>Takes the buffer rather than returning one: this runs once per decoded token over a
    /// 31,678-wide vocabulary, so allocating here would allocate per step of every transcription.</remarks>
    public void Allowed(Span<bool> mask)
    {
        if (mask.Length < _tokenizer.TokenCount)
        {
            throw new ArgumentException(
                $"The mask holds {mask.Length} tokens; SheetSage2's vocabulary is {_tokenizer.TokenCount}.", nameof(mask));
        }
        mask.Clear();
        // Ending is only legal once the current event has written something.
        if (_payloadCount > 0) mask[ScoreTokenizer.EosToken] = true;
        if ((_payloadCount > 0 || _inShift) && _shiftRun < 4)
        {
            Fill(mask, _tokenizer.SubbeatShiftStart, _tokenizer.SubbeatShiftEnd);
        }
        // A half-written field admits only its partner — these short-circuit the ordering rule entirely.
        if (_pending == Pending.RhythmAfterMeter)
        {
            Fill(mask, _tokenizer.EighthPositionStart, _tokenizer.EighthPositionEnd);
            return;
        }
        if (_pending == Pending.MelodyAfterPitch)
        {
            Fill(mask, _tokenizer.DurationStart, _tokenizer.DurationEnd);
            Fill(mask, _tokenizer.PitchStart, _tokenizer.PitchEnd);
            return;
        }
        AllowFieldStarts(mask);
    }

    /// <summary>Allocating convenience over <see cref="Allowed(Span{bool})"/>, for tests and one-off inspection.</summary>
    public bool[] Allowed()
    {
        bool[] mask = new bool[_tokenizer.TokenCount];
        Allowed(mask);
        return mask;
    }

    private void AllowFieldStarts(Span<bool> allowed)
    {
        if (_lastFieldIndex < ScoreTokenizer.FieldIndex("timestamp"))
        {
            Fill(allowed, _tokenizer.TimeStart, _tokenizer.TimeEnd);
        }
        if (_lastFieldIndex < ScoreTokenizer.FieldIndex("rhythm"))
        {
            Fill(allowed, _tokenizer.MeterStart, _tokenizer.MeterEnd);
            Fill(allowed, _tokenizer.EighthPositionStart, _tokenizer.EighthPositionEnd);
        }
        if (_lastFieldIndex < ScoreTokenizer.FieldIndex("structure"))
        {
            Fill(allowed, _tokenizer.StructureStart, _tokenizer.StructureEnd);
        }
        if (_lastFieldIndex < ScoreTokenizer.FieldIndex("key"))
        {
            Fill(allowed, _tokenizer.KeyStart, _tokenizer.KeyEnd);
        }
        if (_lastFieldIndex < ScoreTokenizer.FieldIndex("chord"))
        {
            Fill(allowed, _tokenizer.FullChordStart, _tokenizer.FullChordEnd);
        }
        // Melody alone uses <=, so an event may carry several notes.
        if (_lastFieldIndex <= ScoreTokenizer.FieldIndex("melody"))
        {
            Fill(allowed, _tokenizer.PitchStart, _tokenizer.PitchEnd);
        }
    }

    private static void Fill(Span<bool> mask, int start, int end) => mask[start..end].Fill(true);

    /// <summary>Advances the state by a token that was just emitted.</summary>
    /// <returns>True when that token ended the sequence.</returns>
    public bool Update(int token)
    {
        if (token == ScoreTokenizer.EosToken) return true;
        string type = _tokenizer.TokenType(token);
        if (type == "subbeat_shift")
        {
            // The first shift after a written event closes it.
            if (!_inShift && _payloadCount > 0)
            {
                EventCount++;
                _payloadCount = 0;
                _lastFieldIndex = -1;
                _pending = Pending.None;
            }
            _inShift = true;
            _shiftRun++;
            return false;
        }

        _inShift = false;
        _shiftRun = 0;
        _payloadCount++;
        (_lastFieldIndex, _pending) = type switch
        {
            "time" => (ScoreTokenizer.FieldIndex("timestamp"), Pending.None),
            "meter" => (ScoreTokenizer.FieldIndex("rhythm"), Pending.RhythmAfterMeter),
            "eighth_position" => (ScoreTokenizer.FieldIndex("rhythm"), Pending.None),
            "structure" => (ScoreTokenizer.FieldIndex("structure"), Pending.None),
            "key" => (ScoreTokenizer.FieldIndex("key"), Pending.None),
            "chord_full" => (ScoreTokenizer.FieldIndex("chord"), Pending.None),
            "pitch" => (ScoreTokenizer.FieldIndex("melody"), Pending.MelodyAfterPitch),
            "duration" => (ScoreTokenizer.FieldIndex("melody"), Pending.None),
            _ => throw new InvalidOperationException($"SheetSage2 emitted an unexpected token type '{type}'."),
        };
        return false;
    }
}
