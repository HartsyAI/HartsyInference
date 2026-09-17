using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Models.SheetSage2;

/// <summary>Writes a decoded SheetSage2 transcription as two-voice ABC.
///
/// <para>Nothing about the output is free-form: it has to satisfy the validator the SwarmUI extension ships.
/// Exactly two voices, <c>V: Vocal</c> and <c>V: Ins</c>, interleaved in parallel chunks that carry the same bar
/// count on both sides; <c>Zn</c> pads the idle voice for n bars; chord symbols are quoted and ride the vocal
/// line; sections are <c>%</c> comments; there are no <c>w:</c> lyric lines.</para>
///
/// <para>The work is in getting from decoded times to bars. The model emits beats with a claimed meter, and the
/// claim routinely disagrees with where the downbeats actually land. Downbeats win: a bar is the span between
/// them, and a span shorter than the meter it declares is notated at the declared meter with a rest making up
/// the difference, rather than written out as a bar of the wrong length.</para>
///
/// <para><b>Melody mode is not a chord strip.</b> <c>melodyOnly</c> changes the spelling of the whole score, not
/// just whether chord symbols appear — see <see cref="EventsToAbc"/>.</para></summary>
public static partial class AbcSerializer
{
    /// <summary>Grid steps per beat. A format constant of the released serializer, not derived from anything:
    /// it fixes how finely a note onset can be quantized, and every per-subbeat array is sized from it.</summary>
    public const int SubbeatDivision = 4;

    /// <summary>Seconds within which two times count as the same one, used for beat-grid coverage and for
    /// trimming a note that overlaps the next.</summary>
    private const double TimeEpsilon = 1e-6;

    /// <summary>Beats the tempo estimate is taken over. Only the tail is used, so a rubato opening does not
    /// stretch the synthesized beats that continue the grid past the last decoded one.</summary>
    private const int TempoWindowBeats = 9;

    /// <summary>Bars per parallel chunk before a new one is opened. Chunks also break at a meter change, a key
    /// change and a section label.</summary>
    private const int MeasuresPerGroup = 4;

    /// <summary>Finest ABC unit length that is still worth writing; past it the layout is meaningless.</summary>
    private const int MaxUnitDenominator = 1024;

    /// <summary>The reference's own ceiling for a single ABC length token; longer notes are split and tied.</summary>
    private const int LargestSingleDurationUnit = 48;

    /// <summary>The two voice ids, in the order they are written. Index order is also melody track order.</summary>
    public static readonly string[] VoiceIds = ["Vocal", "Ins"];

    /// <summary>Note lengths a strict parser takes as one ABC value. These are the leading
    /// <see cref="ScoreTokenizer.DurationTemplates"/> entries, derived rather than restated so the two cannot
    /// drift apart if the vocabulary moves.</summary>
    private static readonly int[] _supportedDurationUnits =
        [.. ScoreTokenizer.DurationTemplates.TakeWhile(value => value <= LargestSingleDurationUnit)];

    // Quoted chord symbol, inline key change, or a note/rest with its optional length and tie.
    [GeneratedRegex(@"""(?<quoted>[^""]*)""|\[K:(?<key>[^\]]+)\]|(?<note>[_=^]*[A-Ga-gz][,']*)(?<duration>\d*)(?<tie>-?)")]
    private static partial Regex MusicElementRegex();

    /// <summary>Writes timed events as ABC.</summary>
    /// <param name="events">Decoded events in time order, as the stitching pass leaves them.</param>
    /// <param name="duration">Length of the clip, which bounds note ends and how far the beat grid runs.</param>
    /// <param name="melodyOnly">Drops the chord track.</param>
    /// <remarks><para><b>The two modes are not related by post-processing.</b> Removing the quoted chord symbols
    /// from a full rendering does not produce the melody-only rendering of the same events, because chords change
    /// how the rest of the score is spelled. A chord change mid-note splits that note into tied halves; a bar
    /// carrying a chord cannot be folded into a multi-bar <c>Z</c> rest, so four idle bars are <c>Z4|</c> in
    /// melody mode and four separate bars in full mode; and within a bar the rest after a phrase is grouped
    /// differently. A caller that wants both has to serialize twice.</para></remarks>
    /// <exception cref="BeatGridException">Fewer than two beats, beats out of order, or a beat position outside
    /// its own declared meter.</exception>
    /// <exception cref="MelodyVoiceException">A note that the subbeat grid cannot hold.</exception>
    /// <exception cref="ChordSymbolException">A chord label with no ABC spelling.</exception>
    /// <exception cref="AbcRebuildException">No key was decoded, or the score needs a finer unit length than ABC
    /// can usefully express.</exception>
    public static string EventsToAbc(IReadOnlyList<TimedScoreEvent> events, double duration, bool melodyOnly = true)
        => ScoreToAbc(BuildScore(events, duration, melodyOnly));

    /// <summary>Resolves timed events onto a subbeat grid, one step short of writing the ABC out.</summary>
    /// <remarks>Split out from <see cref="EventsToAbc"/> so the inferred bar table and its diagnostics can be
    /// inspected; none of that survives into the ABC text.</remarks>
    public static RebuiltAbcScore BuildScore(IReadOnlyList<TimedScoreEvent> events, double duration, bool melodyOnly = true)
    {
        ArgumentNullException.ThrowIfNull(events);
        List<BeatEvent> beats = [];
        Dictionary<string, List<NoteSpan>> notes = new(StringComparer.Ordinal);
        foreach (string voice in VoiceIds)
        {
            notes[voice] = [];
        }
        (int Numerator, int Denominator)? meter = null;
        foreach (TimedScoreEvent item in events)
        {
            if (item.Rhythm is ScoreRhythm rhythm)
            {
                if (rhythm.Meter is { } declared) meter = declared;
                if (rhythm.EighthPosition is int eighth && meter is { } active)
                {
                    // The position is in eighths of a bar; scaling by the beat value has to land on a whole beat.
                    int scaled = eighth * active.Denominator;
                    if (scaled % 8 != 0 || scaled < 0 || scaled / 8 >= active.Numerator)
                    {
                        throw new BeatGridException($"Beat position {eighth} is outside the decoded "
                            + $"{active.Numerator}/{active.Denominator} grid.");
                    }
                    beats.Add(new BeatEvent(item.Time, (scaled / 8) + 1, active.Numerator, active.Denominator));
                }
            }
            if (item.Melody is { } melody)
            {
                foreach (TimedScoreNote note in melody)
                {
                    double end = Math.Min(duration, note.EndTime);
                    if (end > item.Time) notes[VoiceIdFor(note.Track)].Add(new NoteSpan(item.Time, end, note.Pitch));
                }
            }
        }
        if (beats.Count < 2)
        {
            throw new BeatGridException("SheetSage2 needs at least two decoded beats to produce ABC.");
        }
        for (int i = 1; i < beats.Count; i++)
        {
            if (beats[i].Time <= beats[i - 1].Time)
            {
                throw new BeatGridException("SheetSage2 decoded beat times are not increasing.");
            }
        }
        ExtendBeatsToDuration(beats, duration);

        double first = beats[0].Time;
        double last = beats[^1].Time;
        List<IntervalRow> keyRows = ClipRows(IntervalRows(events, static item => item.Key, duration), first, last);
        List<IntervalRow> structureRows =
            ClipRows(IntervalRows(events, static item => item.Structure, duration), first, last);
        List<IntervalRow> chordRows = ClipRows(IntervalRows(events, static item => item.Chord, duration), first, last);
        if (keyRows.Count == 0)
        {
            throw new AbcRebuildException("SheetSage2 did not decode a key for the ABC score.");
        }
        for (int i = 0; i < keyRows.Count; i++)
        {
            keyRows[i] = keyRows[i] with { Value = AbcPitchSpelling.KeySymbolToAbc(keyRows[i].Value) };
        }

        (IReadOnlyList<Measure> measures, IReadOnlyList<string> diagnostics) = InferMeasures(beats);
        (double[] times, double[] quarters, int[] denominators) = BuildGrid(beats, measures);
        Dictionary<string, int[]> voices = new(StringComparer.Ordinal);
        foreach (string voice in VoiceIds)
        {
            List<NoteSpan> track = notes[voice];
            track.Sort(static (left, right) =>
            {
                int order = left.Start.CompareTo(right.Start);
                if (order != 0) return order;
                order = left.Pitch.CompareTo(right.Pitch);
                return order != 0 ? order : left.End.CompareTo(right.End);
            });
            // A note that runs into the next one is trimmed, not refused: the model states a length per note and
            // nothing makes it agree with the next onset.
            for (int i = 0; i + 1 < track.Count; i++)
            {
                if (track[i].End > track[i + 1].Start + TimeEpsilon) track[i] = track[i] with { End = track[i + 1].Start };
            }
            List<NoteSpan> kept = [];
            foreach (NoteSpan span in track)
            {
                if (span.End > span.Start + TimeEpsilon) kept.Add(span);
            }
            voices[voice] = NotesToArray(kept, times, voice);
        }

        string[] chordArr;
        if (melodyOnly)
        {
            chordArr = new string[times.Length];
            Array.Fill(chordArr, "N");
        }
        else
        {
            chordArr = FillIntervals(chordRows, times, "N");
        }
        return new RebuiltAbcScore
        {
            Beats = beats,
            Measures = measures,
            SubbeatTimes = times,
            SubbeatQuarters = quarters,
            SubbeatDenominators = denominators,
            KeyArr = FillIntervals(keyRows, times, keyRows[0].Value),
            ChordArr = chordArr,
            StructureEvents = StructureEventList(structureRows, times),
            VoiceArrs = voices,
            Diagnostics = diagnostics,
        };
    }

    /// <summary>Writes a resolved score out as ABC text, ending with a newline.</summary>
    public static string ScoreToAbc(RebuiltAbcScore score)
    {
        ArgumentNullException.ThrowIfNull(score);
        int unitDenominator = AbcUnitDenominator(score);
        Measure firstMeasure = score.Measures[0];
        List<string> lines =
        [
            "X:1",
            "T:",
            $"M:{firstMeasure.AbcNumerator}/{firstMeasure.AbcDenominator}",
            $"L:1/{unitDenominator}",
            $"Q:1/4={(int)Math.Round(EstimateTempo(score))}",
            "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"",
            "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"",
            $"K:{score.KeyArr[firstMeasure.StartT]}",
        ];
        foreach (MeasureGroup group in MeasureGroups(score))
        {
            foreach (string label in group.StructureLabels)
            {
                lines.Add($"% {label}");
            }
            Measure groupStart = group.Measures[0];
            foreach (string voiceId in VoiceIds)
            {
                lines.Add($"V: {voiceId}");
                if (group.MeterChanged) lines.Add($"M:{groupStart.AbcNumerator}/{groupStart.AbcDenominator}");
                if (group.KeyChanged) lines.Add($"K:{score.KeyArr[groupStart.StartT]}");
                lines.Add(RenderVoiceGroup(score, voiceId, group.Measures, unitDenominator));
            }
        }
        return string.Join("\n", lines) + "\n";
    }

    /// <summary>Lays bars out over the decoded beats, taking the downbeats as the bar boundaries.</summary>
    /// <returns>The bars, and a line per bar whose meter had to be reconstructed rather than read.</returns>
    /// <exception cref="BeatGridException">No beat is marked as a downbeat.</exception>
    public static (IReadOnlyList<Measure> Measures, IReadOnlyList<string> Diagnostics) InferMeasures(
        IReadOnlyList<BeatEvent> beats)
    {
        ArgumentNullException.ThrowIfNull(beats);
        List<int> downbeats = [];
        for (int i = 0; i < beats.Count; i++)
        {
            if (beats[i].BeatId == 1) downbeats.Add(i);
        }
        if (downbeats.Count == 0)
        {
            throw new BeatGridException("No downbeat (beat ID 1) exists in the beat lab");
        }
        List<(int Start, int End, bool Pickup, bool Partial)> spans = [];
        if (downbeats[0] > 0) spans.Add((0, downbeats[0], true, false));
        for (int i = 0; i + 1 < downbeats.Count; i++)
        {
            spans.Add((downbeats[i], downbeats[i + 1], false, false));
        }
        // The last row of a beat grid marks the score end. A final span reaching it without a downbeat is a bar
        // the clip cut short, not a bar of that length.
        if (downbeats[^1] < beats.Count - 1) spans.Add((downbeats[^1], beats.Count - 1, false, true));
        if (spans.Count == 0)
        {
            throw new BeatGridException("No positive-length measure exists between downbeats");
        }

        List<string> diagnostics = [];
        List<Measure> measures = [];
        for (int measureIndex = 0; measureIndex < spans.Count; measureIndex++)
        {
            (int start, int end, bool pickup, bool partial) = spans[measureIndex];
            int beatCount = end - start;
            int[] denominators = new int[beatCount];
            int[] declaredNumerators = new int[beatCount];
            for (int i = 0; i < beatCount; i++)
            {
                denominators[i] = beats[start + i].Denominator;
                declaredNumerators[i] = beats[start + i].DeclaredNumerator;
            }
            int denominator = ModeWithFirstTiebreak(denominators);
            int declaredNumerator = ModeWithFirstTiebreak(declaredNumerators);
            bool numeratorConflict = false;
            bool denominatorConflict = false;
            bool uniformNumerators = true;
            foreach (int value in declaredNumerators)
            {
                if (value != beatCount) numeratorConflict = true;
                if (value != declaredNumerators[0]) uniformNumerators = false;
            }
            foreach (int value in denominators)
            {
                if (value != denominator) denominatorConflict = true;
            }
            bool padFinalPartial = partial && uniformNumerators && !denominatorConflict && declaredNumerator >= beatCount;
            bool inferred = pickup || partial || numeratorConflict || denominatorConflict;
            if (padFinalPartial && declaredNumerator > beatCount)
            {
                diagnostics.Add($"measure {measureIndex}: padded final {beatCount}/{denominator} span "
                    + $"to declared {declaredNumerator}/{denominator} with trailing rest");
            }
            else if (numeratorConflict)
            {
                diagnostics.Add($"measure {measureIndex}: inferred {beatCount}/{denominator} from downbeat span; "
                    + $"declared numerators were {PythonListText(declaredNumerators)}");
            }
            if (denominatorConflict)
            {
                diagnostics.Add($"measure {measureIndex}: placed denominator {denominator} at the measure boundary; "
                    + $"row declarations were {PythonListText(denominators)}");
            }
            measures.Add(new Measure(measureIndex, start, end, beatCount, denominator, pickup, partial, inferred,
                padFinalPartial ? declaredNumerator : beatCount, null, false));
        }
        if (measures.Count >= 2)
        {
            Measure opening = measures[0];
            Measure following = measures[1];
            double openingDuration = opening.Numerator / (double)opening.Denominator;
            double followingDuration = following.AbcNumerator / (double)following.AbcDenominator;
            if (openingDuration < followingDuration)
            {
                measures[0] = opening with
                {
                    Inferred = true,
                    NotatedNumerator = following.AbcNumerator,
                    NotatedDenominator = following.AbcDenominator,
                    PadBefore = true,
                };
                diagnostics.Add($"measure 0: padded leading {opening.Numerator}/{opening.Denominator} span "
                    + $"to {following.AbcNumerator}/{following.AbcDenominator} with preceding rest");
            }
        }
        return (measures, diagnostics);
    }

    /// <summary>The ABC unit length every bar in the score can be written against exactly.</summary>
    /// <exception cref="AbcRebuildException">The bars disagree enough that no reasonable unit length fits.</exception>
    public static int AbcUnitDenominator(RebuiltAbcScore score)
    {
        ArgumentNullException.ThrowIfNull(score);
        long denominator = 1;
        foreach (Measure measure in score.Measures)
        {
            denominator = Lcm(denominator, (long)measure.Denominator * score.SubbeatDiv);
            denominator = Lcm(denominator, (long)measure.AbcDenominator * score.SubbeatDiv);
        }
        if (denominator > MaxUnitDenominator)
        {
            throw new AbcRebuildException($"Required ABC unit length 1/{denominator} is unreasonably small");
        }
        return (int)denominator;
    }

    /// <summary>Quarter notes per minute over the whole score.</summary>
    /// <exception cref="AbcRebuildException">The score spans no time or no note value.</exception>
    public static double EstimateTempo(RebuiltAbcScore score)
    {
        ArgumentNullException.ThrowIfNull(score);
        double seconds = score.SubbeatTimes[^1] - score.SubbeatTimes[0];
        double quarterNotes = score.SubbeatQuarters[^1] - score.SubbeatQuarters[0];
        if (seconds <= 0 || quarterNotes <= 0)
        {
            throw new AbcRebuildException("Cannot estimate tempo from a zero-duration score");
        }
        return quarterNotes / seconds * 60.0;
    }

    /// <summary>Continues the beat grid at the tail tempo until it reaches the end of the clip.</summary>
    /// <remarks>The reference walks this loop with no bound. A period below the decoder's own timestamp
    /// resolution is not a tempo but a mis-decode, and walking one out to the end of a song would synthesize
    /// millions of beats, so this refuses instead of grinding — a deliberate divergence.</remarks>
    private static void ExtendBeatsToDuration(List<BeatEvent> beats, double duration)
    {
        if (beats[^1].Time >= duration - TimeEpsilon) return;
        double period = MedianBeatPeriod(beats);
        if (!double.IsFinite(period) || period < 1.0 / ScoreTokenizer.TimeHz)
        {
            throw new BeatGridException($"SheetSage2 decoded a beat period of {period.ToString("G6", CultureInfo.InvariantCulture)}s, "
                + "which is below its own timestamp resolution; the beat grid cannot be continued to the end of the clip.");
        }
        while (beats[^1].Time < duration - TimeEpsilon)
        {
            BeatEvent previous = beats[^1];
            beats.Add(new BeatEvent(previous.Time + period, (previous.BeatId % previous.DeclaredNumerator) + 1,
                previous.DeclaredNumerator, previous.Denominator));
        }
    }

    private static double MedianBeatPeriod(List<BeatEvent> beats)
    {
        int from = Math.Max(0, beats.Count - TempoWindowBeats);
        double[] diffs = new double[beats.Count - from - 1];
        for (int i = 0; i < diffs.Length; i++)
        {
            diffs[i] = beats[from + i + 1].Time - beats[from + i].Time;
        }
        Array.Sort(diffs);
        int middle = diffs.Length / 2;
        return diffs.Length % 2 == 1 ? diffs[middle] : (diffs[middle - 1] + diffs[middle]) / 2;
    }

    private static (double[] Times, double[] Quarters, int[] Denominators) BuildGrid(
        List<BeatEvent> beats, IReadOnlyList<Measure> measures)
    {
        int[] intervalDenominators = new int[beats.Count - 1];
        foreach (Measure measure in measures)
        {
            for (int i = measure.StartBeat; i < Math.Min(measure.EndBeat, intervalDenominators.Length); i++)
            {
                intervalDenominators[i] = measure.Denominator;
            }
        }
        foreach (int value in intervalDenominators)
        {
            if (value == 0) throw new BeatGridException("Downbeat spans do not cover every beat interval");
        }
        double[] times = new double[((beats.Count - 1) * SubbeatDivision) + 1];
        double[] quarters = new double[times.Length];
        int[] denominators = new int[times.Length];
        double currentQuarter = 0.0;
        int cursor = 0;
        for (int index = 0; index < beats.Count - 1; index++)
        {
            double start = beats[index].Time;
            // The reference cuts the interval with numpy's linspace and drops its endpoint; start + k * step is
            // that computation bit for bit.
            double step = (beats[index + 1].Time - start) / SubbeatDivision;
            double quarterStep = 4.0 / intervalDenominators[index] / SubbeatDivision;
            for (int k = 0; k < SubbeatDivision; k++)
            {
                times[cursor] = (k * step) + start;
                denominators[cursor] = intervalDenominators[index];
                currentQuarter += quarterStep;
                quarters[cursor + 1] = currentQuarter;
                cursor++;
            }
        }
        times[cursor] = beats[^1].Time;
        denominators[cursor] = intervalDenominators[^1];
        return (times, quarters, denominators);
    }

    /// <summary>Which subbeat a time falls on, by nearest midpoint.</summary>
    private static int QuantizeTime(double time, double[] subbeatTimes)
    {
        // The reference searches the midpoints between consecutive subbeats, taking the count that fall strictly
        // below the time. Midpoints are strictly increasing, so a binary search finds the same index.
        int low = 0;
        int high = subbeatTimes.Length - 1;
        while (low < high)
        {
            int middle = (low + high) / 2;
            if ((subbeatTimes[middle] + subbeatTimes[middle + 1]) / 2 < time) low = middle + 1;
            else high = middle;
        }
        return low;
    }

    private static string[] FillIntervals(List<IntervalRow> rows, double[] subbeatTimes, string defaultValue)
    {
        string[] result = new string[subbeatTimes.Length];
        Array.Fill(result, defaultValue);
        foreach (IntervalRow row in rows)
        {
            int startT = Math.Clamp(QuantizeTime(row.Start, subbeatTimes), 0, result.Length - 1);
            int endT = Math.Clamp(QuantizeTime(row.End, subbeatTimes), 0, result.Length - 1);
            // An interval that quantizes entirely onto the final subbeat is dropped; that step exists to close
            // the grid, not to hold a value.
            if (startT == endT && endT == result.Length - 1) continue;
            if (endT <= startT)
            {
                throw new AbcRebuildException(
                    $"Interval {Seconds(row.Start)}-{Seconds(row.End)} ({row.Value}) is shorter than the ABC subbeat grid");
            }
            for (int t = startT; t < endT; t++)
            {
                result[t] = row.Value;
            }
        }
        // The closing step is never written to, so it takes the value that runs into it.
        if (result.Length > 1) result[^1] = result[^2];
        return result;
    }

    private static List<(int Subbeat, string Label)> StructureEventList(List<IntervalRow> rows, double[] subbeatTimes)
    {
        List<(int, string)> events = [];
        foreach (IntervalRow row in rows)
        {
            events.Add((Math.Clamp(QuantizeTime(row.Start, subbeatTimes), 0, subbeatTimes.Length - 1), row.Value));
        }
        return events;
    }

    private static int[] NotesToArray(List<NoteSpan> notes, double[] subbeatTimes, string voiceId)
    {
        int[] result = new int[subbeatTimes.Length];
        List<NoteSpan> ordered = [.. notes];
        ordered.Sort(static (left, right) =>
        {
            int order = left.Start.CompareTo(right.Start);
            if (order != 0) return order;
            order = left.End.CompareTo(right.End);
            return order != 0 ? order : left.Pitch.CompareTo(right.Pitch);
        });
        foreach (NoteSpan note in ordered)
        {
            int startT = Math.Clamp(QuantizeTime(note.Start, subbeatTimes), 0, result.Length - 1);
            int endT = Math.Clamp(QuantizeTime(note.End, subbeatTimes), 0, result.Length - 1);
            if (startT == endT && endT == result.Length - 1) continue;
            if (endT <= startT)
            {
                throw new MelodyVoiceException($"{voiceId}: note pitch={note.Pitch} at "
                    + $"{Seconds(note.Start)}-{Seconds(note.End)} cannot be represented on the decoded subbeat grid");
            }
            for (int t = startT; t < endT; t++)
            {
                if (result[t] != 0)
                {
                    throw new MelodyVoiceException(
                        $"{voiceId}: overlapping quantized melody notes at subbeats {startT}:{endT}");
                }
            }
            int sustain = (note.Pitch * 2) + 2;
            for (int t = startT; t < endT; t++)
            {
                result[t] = sustain;
            }
            result[startT] = sustain + 1;
        }
        return result;
    }

    private static List<IntervalRow> IntervalRows(
        IReadOnlyList<TimedScoreEvent> events, Func<TimedScoreEvent, string?> field, double duration)
    {
        List<IntervalRow> rows = [];
        foreach (TimedScoreEvent item in events)
        {
            string? value = field(item);
            if (value is not null) rows.Add(new IntervalRow(item.Time, duration, value));
        }
        // Each value runs until the next event states one; only the last reaches the end of the clip.
        for (int i = 0; i + 1 < rows.Count; i++)
        {
            rows[i] = rows[i] with { End = rows[i + 1].Start };
        }
        List<IntervalRow> kept = [];
        foreach (IntervalRow row in rows)
        {
            if (row.End > row.Start) kept.Add(row);
        }
        return kept;
    }

    private static List<IntervalRow> ClipRows(List<IntervalRow> rows, double first, double last)
    {
        List<IntervalRow> kept = [];
        foreach (IntervalRow row in rows)
        {
            if (row.End > first && row.Start < last)
            {
                kept.Add(new IntervalRow(Math.Max(first, row.Start), Math.Min(last, row.End), row.Value));
            }
        }
        return kept;
    }

    private static List<MeasureGroup> MeasureGroups(RebuiltAbcScore score)
    {
        Measure firstMeasure = score.Measures[0];
        (int Numerator, int Denominator) activeMeter = (firstMeasure.AbcNumerator, firstMeasure.AbcDenominator);
        string activeKey = score.KeyArr[firstMeasure.StartT];
        string activeStructure = "";
        List<MeasureGroup> groups = [];
        foreach (Measure measure in score.Measures)
        {
            (int Numerator, int Denominator) meter = (measure.AbcNumerator, measure.AbcDenominator);
            string key = score.KeyArr[measure.StartT];
            bool meterChanged = meter != activeMeter;
            bool keyChanged = key != activeKey;
            List<string> newLabels = [];
            foreach ((int subbeat, string label) in score.StructureEvents)
            {
                if (subbeat < measure.StartT || subbeat >= measure.EndT) continue;
                string clean = SanitizeStructureLabel(label);
                if (clean.Length > 0 && clean != activeStructure)
                {
                    newLabels.Add(clean);
                    activeStructure = clean;
                }
            }
            if (groups.Count == 0 || groups[^1].Measures.Count >= MeasuresPerGroup || meterChanged || keyChanged
                || newLabels.Count > 0)
            {
                groups.Add(new MeasureGroup([measure], newLabels, meterChanged, keyChanged));
            }
            else
            {
                groups[^1].Measures.Add(measure);
            }
            activeMeter = meter;
            // The key a bar ENDS on is what the next bar is compared against, so an inline change mid-bar does
            // not also open a new chunk on the following bar.
            activeKey = score.KeyArr[measure.EndT - 1];
        }
        return groups;
    }

    private static string RenderVoiceGroup(
        RebuiltAbcScore score, string voiceId, IReadOnlyList<Measure> measures, int unitDenominator)
    {
        string[] rendered = new string[measures.Count];
        for (int i = 0; i < measures.Count; i++)
        {
            rendered[i] = RenderVoiceMeasure(score, voiceId, measures[i], unitDenominator);
        }
        StringBuilder parts = new();
        int index = 0;
        while (index < rendered.Length)
        {
            if (!IsCompressibleFullRest(rendered[index]))
            {
                parts.Append(rendered[index]).Append('|');
                index++;
                continue;
            }
            int end = index + 1;
            while (end < rendered.Length && IsCompressibleFullRest(rendered[end]))
            {
                end++;
            }
            int count = end - index;
            parts.Append('Z');
            if (count > 1) parts.Append(count.ToString(CultureInfo.InvariantCulture));
            parts.Append('|');
            index = end;
        }
        return parts.ToString();
    }

    private static string RenderVoiceMeasure(RebuiltAbcScore score, string voiceId, Measure measure, int unitDenominator)
    {
        int[] voice = score.VoiceArrs[voiceId];
        bool showChords = voiceId == VoiceIds[0];
        Dictionary<int, int> measureAccidentals = [];
        string currentKey = score.KeyArr[measure.StartT];
        int[] keyAccidentals = AbcPitchSpelling.GetKeyAccidentals(currentKey);
        StringBuilder parts = new();
        int padding = MeasureUnits(measure.AbcNumerator, measure.AbcDenominator, unitDenominator)
            - MeasureUnits(measure.Numerator, measure.Denominator, unitDenominator);
        if (padding < 0)
        {
            throw new AbcRebuildException($"Measure {measure.Index}: notated meter is shorter than its decoded span");
        }
        int leadingPadding = measure.PadBefore ? padding : 0;
        int trailingPadding = measure.PadBefore ? 0 : padding;
        int t = measure.StartT;
        while (t < measure.EndT)
        {
            int nextT = measure.EndT;
            for (int probe = t + 1; probe < measure.EndT; probe++)
            {
                if (!SameNoteSegment(voice[t], voice[probe]))
                {
                    nextT = Math.Min(nextT, probe);
                    break;
                }
            }
            for (int probe = t + 1; probe < measure.EndT; probe++)
            {
                if (score.KeyArr[probe] != score.KeyArr[probe - 1])
                {
                    nextT = Math.Min(nextT, probe);
                    break;
                }
            }
            if (showChords)
            {
                for (int probe = t + 1; probe < measure.EndT; probe++)
                {
                    if (score.ChordArr[probe] != score.ChordArr[probe - 1])
                    {
                        nextT = Math.Min(nextT, probe);
                        break;
                    }
                }
            }

            string prefix = "";
            string key = score.KeyArr[t];
            if (t > measure.StartT && key != currentKey)
            {
                currentKey = key;
                keyAccidentals = AbcPitchSpelling.GetKeyAccidentals(currentKey);
                measureAccidentals.Clear();
                prefix += $"[K:{currentKey}]";
            }
            if (showChords && (t == measure.StartT || score.ChordArr[t] != score.ChordArr[t - 1]))
            {
                string? chordText = AbcPitchSpelling.ChordSymbolToAbc(score.ChordArr[t]);
                if (chordText is not null) prefix += $"\"{chordText}\"";
            }

            int value = voice[t];
            string noteText = value == 0
                ? "z"
                : AbcPitchSpelling.NoteToAbc((value / 2) - 1, keyAccidentals, measureAccidentals);
            int duration = DurationUnits(score, t, nextT, unitDenominator);
            if (t == measure.StartT && leadingPadding != 0)
            {
                // A bare opening rest absorbs the padding; anything else needs its own rest in front of it.
                if (value == 0 && prefix.Length == 0) duration += leadingPadding;
                else AppendDurationTokens(parts, "", "z", leadingPadding, tieOut: false);
                leadingPadding = 0;
            }
            if (value == 0 && nextT == measure.EndT && trailingPadding != 0)
            {
                duration += trailingPadding;
                trailingPadding = 0;
            }
            if (duration <= 0)
            {
                throw new AbcRebuildException($"Non-positive ABC duration at subbeats {t}:{nextT}");
            }
            bool tieOut = value > 0 && nextT < voice.Length && ContinuesPitch(value, voice[nextT]);
            AppendDurationTokens(parts, prefix, noteText, duration, tieOut);
            t = nextT;
        }
        if (leadingPadding != 0)
        {
            throw new AbcRebuildException($"Measure {measure.Index}: leading rest padding was not serialized");
        }
        if (trailingPadding != 0) AppendDurationTokens(parts, "", "z", trailingPadding, tieOut: false);
        return parts.ToString();
    }

    private static void AppendDurationTokens(
        StringBuilder parts, string prefix, string noteText, int duration, bool tieOut)
    {
        List<int> chunks = SplitDurationUnits(duration);
        for (int index = 0; index < chunks.Count; index++)
        {
            bool continues = noteText != "z" && (index + 1 < chunks.Count || tieOut);
            if (index == 0) parts.Append(prefix);
            parts.Append(noteText);
            if (chunks[index] != 1) parts.Append(chunks[index].ToString(CultureInfo.InvariantCulture));
            if (continues) parts.Append('-');
        }
    }

    /// <summary>Splits a length into values a strict parser accepts, largest first.</summary>
    internal static List<int> SplitDurationUnits(int duration)
    {
        if (duration <= 0)
        {
            throw new AbcRebuildException($"Cannot serialize non-positive duration {duration}");
        }
        List<int> result = [];
        int remaining = duration;
        while (remaining != 0)
        {
            if (Array.IndexOf(_supportedDurationUnits, remaining) >= 0)
            {
                result.Add(remaining);
                break;
            }
            int chunk = 0;
            foreach (int value in _supportedDurationUnits)
            {
                if (value < remaining) chunk = value;
            }
            if (chunk == 0)
            {
                throw new AbcRebuildException($"Duration {duration} cannot be split into representable ABC values");
            }
            result.Add(chunk);
            remaining -= chunk;
        }
        return result;
    }

    private static int DurationUnits(RebuiltAbcScore score, int startT, int endT, int unitDenominator)
    {
        int units = 0;
        for (int t = startT; t < endT; t++)
        {
            int divisor = score.SubbeatDenominators[t] * score.SubbeatDiv;
            if (unitDenominator % divisor != 0)
            {
                throw new AbcRebuildException(
                    $"ABC L:1/{unitDenominator} cannot express a 1/{divisor} subbeat exactly");
            }
            units += unitDenominator / divisor;
        }
        return units;
    }

    /// <summary>Whether a rendered bar is nothing but plain rests, so ABC's <c>Z</c> can replace it losslessly.</summary>
    internal static bool IsCompressibleFullRest(string renderedMeasure)
    {
        int cursor = 0;
        bool sawNote = false;
        foreach (Match match in MusicElementRegex().Matches(renderedMeasure))
        {
            if (match.Index > cursor) return false;
            cursor = match.Index + match.Length;
            // A chord symbol or an inline key change has to stay on its own bar to keep its position.
            if (match.Groups["quoted"].Success || match.Groups["key"].Success) return false;
            sawNote = true;
            if (match.Groups["note"].Value != "z" || match.Groups["tie"].Length > 0) return false;
        }
        return sawNote && cursor == renderedMeasure.Length;
    }

    /// <summary>Whether the next subbeat sustains this one's pitch, which is what a tie is written for.</summary>
    private static bool ContinuesPitch(int value, int nextValue)
        => value > 0 && nextValue == (((value / 2) - 1) * 2) + 2;

    /// <summary>Whether two subbeats belong to the same written note.</summary>
    private static bool SameNoteSegment(int value, int nextValue)
        => value == 0 ? nextValue == 0 : nextValue == (((value / 2) - 1) * 2) + 2;

    private static int MeasureUnits(int numerator, int denominator, int unitDenominator)
        => numerator * unitDenominator / denominator;

    private static string SanitizeStructureLabel(string value)
        => string.Join(' ', value.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));

    /// <summary>The most common value, breaking ties towards the one that appears first.</summary>
    private static int ModeWithFirstTiebreak(int[] values)
    {
        int best = values[0];
        int bestCount = 0;
        foreach (int candidate in values)
        {
            int count = 0;
            foreach (int value in values)
            {
                if (value == candidate) count++;
            }
            if (count > bestCount)
            {
                bestCount = count;
                best = candidate;
            }
        }
        return best;
    }

    private static string VoiceIdFor(int track)
    {
        // The reference indexes a two-element tuple, so track 2 raises and track -1 silently lands on "Ins".
        if (track < 0 || track >= VoiceIds.Length)
        {
            throw new MelodyVoiceException($"SheetSage2 decoded a melody note on track {track}; only "
                + $"0..{VoiceIds.Length - 1} exist.");
        }
        return VoiceIds[track];
    }

    private static long Lcm(long left, long right)
    {
        long a = left;
        long b = right;
        while (b != 0)
        {
            (a, b) = (b, a % b);
        }
        return left / a * right;
    }

    private static string Seconds(double value) => value.ToString("F6", CultureInfo.InvariantCulture);

    /// <summary>Renders a list the way Python's repr does, because the diagnostics embed one.</summary>
    private static string PythonListText(int[] values)
    {
        StringBuilder text = new("[");
        for (int i = 0; i < values.Length; i++)
        {
            if (i > 0) text.Append(", ");
            text.Append(values[i].ToString(CultureInfo.InvariantCulture));
        }
        return text.Append(']').ToString();
    }

    /// <summary>One value's span on the clip's timeline, before it is quantized onto the subbeat grid.</summary>
    private readonly record struct IntervalRow(double Start, double End, string Value);

    /// <summary>One melody note's span on the clip's timeline.</summary>
    private readonly record struct NoteSpan(double Start, double End, int Pitch);

    /// <summary>Bars written as one parallel chunk, with whatever opens it.</summary>
    private sealed class MeasureGroup(List<Measure> measures, List<string> structureLabels, bool meterChanged, bool keyChanged)
    {
        public List<Measure> Measures { get; } = measures;

        public List<string> StructureLabels { get; } = structureLabels;

        public bool MeterChanged { get; } = meterChanged;

        public bool KeyChanged { get; } = keyChanged;
    }
}
