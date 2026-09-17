namespace HartsyInference.Audio.Models.SheetSage2;

/// <summary>What one window's decoded timestamps say about where a subbeat falls in that window.
///
/// <para>Only some events carry a timestamp, so the rest are placed by interpolating between the ones that do,
/// and extrapolating at a fixed tempo past the outermost of them. The tempo is the median of what the anchors
/// themselves say, which survives a single wild timestamp where a mean would not.</para></summary>
public sealed class SubbeatTimeMap
{
    /// <summary>Seconds one subbeat lasts when the window says nothing usable: four subbeats to the beat at 120
    /// BPM. The released implementation states it outright in every fallback and nothing in the vocabulary
    /// implies it, so it is reproduced as the constant it is.</summary>
    public const double DefaultSubbeatSeconds = 0.125;

    private readonly double[] _steps;
    private readonly double[] _times;
    private readonly double _targetSeconds;

    /// <summary>Reads the anchors out of one window's decoded events.</summary>
    /// <param name="events">That window's events, in decode order; the ones carrying a timestamp anchor the map.</param>
    /// <param name="targetSeconds">Longest time the map will report. The released pipeline passes a whole window
    /// even for the short last one, so a map may well name a time past the end of the clip — what keeps events
    /// inside it is <see cref="WindowStitcher.WindowEvents"/>, not this.</param>
    public SubbeatTimeMap(IReadOnlyList<ScoreEvent> events, double targetSeconds = SlidingWindowPlan.WindowSeconds)
    {
        ArgumentNullException.ThrowIfNull(events);
        _targetSeconds = targetSeconds;
        // A repeated position keeps its LAST timestamp, as the reference's dict comprehension does.
        Dictionary<int, double> anchors = [];
        foreach (ScoreEvent item in events)
        {
            if (item.Timestamp is double seconds) anchors[item.Subbeat] = seconds;
        }
        int[] positions = [.. anchors.Keys];
        Array.Sort(positions);
        _steps = new double[positions.Length];
        _times = new double[positions.Length];
        for (int i = 0; i < positions.Length; i++)
        {
            _steps[i] = positions[i];
            _times[i] = anchors[positions[i]];
        }
        SubbeatSeconds = Tempo();
    }

    /// <summary>Seconds one subbeat lasts, used to extrapolate past the outermost anchor.</summary>
    public double SubbeatSeconds { get; }

    /// <summary>How many timestamps anchor the map.</summary>
    public int AnchorCount => _steps.Length;

    /// <summary>Where a position falls in the window, in seconds.</summary>
    /// <param name="subbeat">Position, counted from the window's own start.</param>
    public double Seconds(double subbeat)
    {
        if (_steps.Length == 0)
        {
            // Not one timestamp in the whole window: the reference reads it at a fixed tempo rather than refusing it.
            return Math.Min(_targetSeconds, Math.Max(0.0, subbeat * DefaultSubbeatSeconds));
        }
        if (subbeat <= _steps[0]) return Clip(_times[0] + (subbeat - _steps[0]) * SubbeatSeconds);
        if (subbeat >= _steps[^1]) return Clip(_times[^1] + (subbeat - _steps[^1]) * SubbeatSeconds);
        int index = Array.BinarySearch(_steps, subbeat);
        if (index < 0) index = ~index - 1;
        double slope = (_times[index + 1] - _times[index]) / (_steps[index + 1] - _steps[index]);
        // Interpolation is deliberately not clipped, matching the reference: only the two extrapolations are.
        return slope * (subbeat - _steps[index]) + _times[index];
    }

    private double Clip(double seconds) => Math.Min(Math.Max(seconds, 0.0), _targetSeconds);

    /// <summary>The median of what consecutive anchors say a subbeat lasts.</summary>
    private double Tempo()
    {
        if (_steps.Length < 2) return DefaultSubbeatSeconds;
        double[] rates = new double[_steps.Length - 1];
        for (int i = 0; i < rates.Length; i++)
        {
            // The reference floors the gap at one subbeat; positions are unique and sorted, so it never bites.
            rates[i] = (_times[i + 1] - _times[i]) / Math.Max(_steps[i + 1] - _steps[i], 1.0);
        }
        Array.Sort(rates);
        int middle = rates.Length / 2;
        double median = (rates.Length & 1) == 1 ? rates[middle] : (rates[middle - 1] + rates[middle]) / 2.0;
        // Timestamps that do not advance with position give a flat or backwards tempo, which places nothing.
        return double.IsFinite(median) && median > 0 ? median : DefaultSubbeatSeconds;
    }
}

/// <summary>Joins the windows of a long transcription into one stream of events.
///
/// <para>Each window is decoded on its own, in its own coordinates, and the windows overlap heavily. Two things
/// make the seams disappear. Going forward, a window is only trusted for its accepted span, and the spans tile
/// the clip exactly once — which is why the tolerance on a span's edges is asymmetric: a window takes an event
/// sitting a tolerance BEFORE its span and hands one a tolerance before its END to the next window, so an event
/// landing exactly on a seam is accepted once rather than twice or never. Going backwards, the part of the new
/// window that the previous ones already read is replayed as the decoder's prompt, re-based and re-timed into
/// the new window's coordinates, so the model continues a reading instead of starting a fresh one.</para></summary>
public sealed class WindowStitcher(ScoreTokenizer tokenizer)
{
    /// <summary>Shortest a placed note may be. A note whose length maps to less than this — or to nothing, when
    /// the time map has flattened — would otherwise end where it starts.</summary>
    public const double MinimumNoteSeconds = 0.04;

    /// <summary>Slack on an accepted span's edges, in seconds. Asymmetric on purpose; see the class summary.</summary>
    private const double SeamToleranceSeconds = 1e-4;

    /// <summary>Slack when deciding whether a field was already in force at the replayed prefix's first event.</summary>
    private const double ContextToleranceSeconds = 1e-6;

    /// <summary>Fields the replayed prefix inherits when its first event does not restate them. These stay in
    /// force until something changes them, so a window that opened mid-verse has to be told the verse.</summary>
    private static readonly string[] CarriedFields = ["structure", "key", "chord"];

    private readonly ScoreTokenizer _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
    private readonly ScoreEventCodec _codec = new(tokenizer);

    /// <summary>Places one window's events on the clip's timeline and keeps those it is trusted for.</summary>
    /// <param name="decoded">The window's events, as <see cref="ScoreEventCodec.DecodeSequence"/> read them. They
    /// are borrowed, not copied: the accepted events point back at them and nothing here rewrites them.</param>
    /// <param name="times">That window's time map.</param>
    /// <param name="window">The window, from <see cref="SlidingWindowPlan.For"/>.</param>
    /// <param name="durationSeconds">Length of the whole clip.</param>
    /// <param name="globalSubbeatBase">Where the window's own subbeat zero sits in the whole piece — the base
    /// <see cref="OverlapPrefix"/> returned when it built this window's prompt.</param>
    public List<StitchedScoreEvent> WindowEvents(IReadOnlyList<ScoreEvent> decoded, SubbeatTimeMap times,
        in SheetSage2Window window, double durationSeconds, int globalSubbeatBase = 0)
    {
        ArgumentNullException.ThrowIfNull(decoded);
        ArgumentNullException.ThrowIfNull(times);
        double start = window.Start;
        List<StitchedScoreEvent> accepted = [];
        foreach (ScoreEvent source in decoded)
        {
            double placed = start + times.Seconds(source.Subbeat);
            if (placed < window.AcceptStart - SeamToleranceSeconds) continue;
            if (placed >= window.AcceptEnd - SeamToleranceSeconds) continue;
            if (placed >= durationSeconds - SeamToleranceSeconds) continue;
            double seconds = Math.Min(Math.Max(placed, 0.0), durationSeconds);
            List<StitchedNote>? melody = null;
            if (source.Melody is not null)
            {
                melody = new List<StitchedNote>(source.Melody.Count);
                foreach (ScoreNote note in source.Melody)
                {
                    double end = Math.Min(durationSeconds, Math.Max(seconds + MinimumNoteSeconds,
                        start + times.Seconds(source.Subbeat + note.DurationSteps)));
                    melody.Add(new StitchedNote(note, end));
                }
            }
            accepted.Add(new StitchedScoreEvent
            {
                Source = source,
                Seconds = seconds,
                GlobalSubbeat = globalSubbeatBase + source.Subbeat,
                Melody = melody,
            });
        }
        return accepted;
    }

    /// <summary>Writes the part of a window that earlier windows already read back as that window's prompt.</summary>
    /// <param name="stitched">Everything accepted so far, in the order it was accepted.</param>
    /// <param name="window">The window about to be decoded; its <see cref="SheetSage2Window.Start"/> sets the
    /// coordinates the prefix is rewritten into and its <see cref="SheetSage2Window.PrefixEnd"/> ends it.</param>
    /// <returns>The prompt, and the position in the whole piece that its subbeat zero sits at — which is what
    /// <see cref="WindowEvents"/> then needs as its base. The prompt is null when the overlap holds nothing that
    /// states a position, which is the one case where a window legitimately starts over.</returns>
    public (List<int>? Tokens, int GlobalSubbeatBase) OverlapPrefix(
        IReadOnlyList<StitchedScoreEvent> stitched, in SheetSage2Window window)
    {
        ArgumentNullException.ThrowIfNull(stitched);
        double start = window.Start;
        List<(int Global, double Seconds, int Index)> order = [];
        for (int i = 0; i < stitched.Count; i++)
        {
            double seconds = stitched[i].Seconds;
            if (seconds >= start - SeamToleranceSeconds && seconds < window.PrefixEnd - SeamToleranceSeconds)
            {
                order.Add((stitched[i].GlobalSubbeat, seconds, i));
            }
        }
        // Ordering by position then time still ties when a window wrote two events at one position, so the
        // accepted order breaks it — an unstable sort would reorder the prompt's tokens from run to run.
        order.Sort(static (a, b) =>
        {
            int comparison = a.Global.CompareTo(b.Global);
            if (comparison != 0) return comparison;
            comparison = a.Seconds.CompareTo(b.Seconds);
            return comparison != 0 ? comparison : a.Index.CompareTo(b.Index);
        });
        // The prompt has to open on an event that states a position, since everything after it is written as an
        // offset from that one. The reference tests the read values; they are present exactly when the tokens
        // that produced them are, and the rest of this works in tokens.
        int first = -1;
        for (int k = 0; k < order.Count; k++)
        {
            Dictionary<string, List<int>> fields = stitched[order[k].Index].Source.TokensByField;
            if (fields.ContainsKey("timestamp") || fields.ContainsKey("rhythm"))
            {
                first = k;
                break;
            }
        }
        if (first < 0) return (null, 0);

        StitchedScoreEvent opening = stitched[order[first].Index];
        int baseSubbeat = opening.GlobalSubbeat;
        Dictionary<string, List<int>> context = CarriedContext(stitched, opening.Seconds, out int? meter);
        // The largest id the timestamp range can name, derived so it cannot drift from the range it indexes.
        int maxTimeId = _tokenizer.TimeEnd - _tokenizer.TimeStart - 1;
        List<ScoreEvent> prefix = new(order.Count - first);
        for (int k = first; k < order.Count; k++)
        {
            StitchedScoreEvent item = stitched[order[k].Index];
            Dictionary<string, List<int>> fields = new(item.Source.TokensByField.Count, StringComparer.Ordinal);
            foreach (KeyValuePair<string, List<int>> entry in item.Source.TokensByField)
            {
                fields[entry.Key] = [.. entry.Value];
            }
            if (fields.ContainsKey("timestamp"))
            {
                // Clamped before the cast, not after: an absurd span would otherwise wrap to a valid-looking id.
                // The clamp cannot bite on a real plan, whose overlap is at most a window's worth of seconds.
                double ticks = Math.Round((item.Seconds - start) * ScoreTokenizer.TimeHz);
                fields["timestamp"] = [_tokenizer.TimeStart + (int)Math.Clamp(ticks, 0.0, maxTimeId)];
            }
            prefix.Add(new ScoreEvent
            {
                Subbeat = Math.Max(0, item.GlobalSubbeat - baseSubbeat),
                TokensByField = fields,
            });
        }
        Restate(prefix[0].TokensByField, context, meter);
        return (_codec.EncodeEvents(prefix), baseSubbeat);
    }

    /// <summary>The fields in force at the moment the prompt opens: the last value each was given anywhere at or
    /// before that moment, including by windows that ended before this one began.</summary>
    private Dictionary<string, List<int>> CarriedContext(IReadOnlyList<StitchedScoreEvent> stitched,
        double openingSeconds, out int? meter)
    {
        meter = null;
        Dictionary<string, List<int>> context = new(StringComparer.Ordinal);
        foreach (StitchedScoreEvent item in stitched)
        {
            if (item.Seconds > openingSeconds + ContextToleranceSeconds) continue;
            foreach (string field in CarriedFields)
            {
                if (item.Source.TokensByField.TryGetValue(field, out List<int>? stated) && stated.Count > 0)
                {
                    context[field] = stated;
                }
            }
            if (!item.Source.TokensByField.TryGetValue("rhythm", out List<int>? rhythm)) continue;
            foreach (int token in rhythm)
            {
                if (_tokenizer.TokenType(token) == "meter") meter = token;
            }
        }
        return context;
    }

    /// <summary>Writes the inherited fields onto the prompt's first event, which is the only place a window can
    /// be told what was already in force.</summary>
    private void Restate(Dictionary<string, List<int>> fields, Dictionary<string, List<int>> context, int? meter)
    {
        foreach (string field in CarriedFields)
        {
            if (!fields.ContainsKey(field) && context.TryGetValue(field, out List<int>? stated))
            {
                fields[field] = [.. stated];
            }
        }
        // Only a bar position that has lost its time signature gets one back. An opening event with no rhythm at
        // all is left alone, as in the reference: a bare meter would claim a downbeat the window never wrote.
        if (meter is not int meterToken || !fields.TryGetValue("rhythm", out List<int>? rhythmTokens)) return;
        bool position = false;
        foreach (int token in rhythmTokens)
        {
            string type = _tokenizer.TokenType(token);
            if (type == "meter") return;
            if (type == "eighth_position") position = true;
        }
        if (!position) return;
        List<int> restated = new(rhythmTokens.Count + 1) { meterToken };
        restated.AddRange(rhythmTokens);
        fields["rhythm"] = restated;
    }
}
