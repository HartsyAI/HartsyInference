namespace HartsyInference.Audio.Models.SheetSage2;

/// <summary>SheetSage2's symbolic vocabulary: the token ids a transcription is written in.
///
/// <para>A transcription is a stream of events, each a subset of six fields in a fixed order — timestamp,
/// rhythm, structure, key, chord, melody — separated by subbeat shifts. Every field owns a contiguous id range,
/// and the ranges are laid out in one pass from offset 260, so an id's meaning is decided purely by which range
/// it falls in. All of it is checkpoint fact reproduced from the released <c>ScoreTokenizer</c>: the counts, the
/// order and the starting offset are what the decoder's output projection was trained against.</para></summary>
public sealed class ScoreTokenizer
{
    /// <summary>Padding.</summary>
    public const int PadToken = 0;

    /// <summary>Start of sequence; opens the prompt prefix.</summary>
    public const int SosToken = 1;

    /// <summary>End of sequence; the decode loop's stop.</summary>
    public const int EosToken = 2;

    /// <summary>Closes the prompt prefix and opens the transcription proper.</summary>
    public const int OutToken = 3;

    /// <summary>Timestamps are counted at 100 Hz.</summary>
    public const int TimeHz = 100;

    /// <summary>Where the field ranges begin. Everything below is a special or a prompt token.</summary>
    private const int RangeOffset = 260;

    /// <summary>Pitch-class spelling used to build the chord vocabulary.</summary>
    public static readonly string[] ChromaticSharps =
        ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

    /// <summary>Section labels, in id order.</summary>
    public static readonly string[] StructureLabels =
    [
        "silence", "intro", "outro", "verse", "chorus", "bridge", "pre-chorus", "post-chorus", "interlude",
        "fade-out", "loop", "rap", "preshot", "irregular", "instrumental", "intro and verse",
        "pre-chorus and chorus", "verse and pre-chorus", "solo", "theme", "development", "variation", "pre-outro",
    ];

    /// <summary>Note lengths, in subbeats, that a duration token can name.</summary>
    public static readonly int[] DurationTemplates =
        [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096];

    /// <summary>The six event fields, in the order the grammar requires them.</summary>
    public static readonly string[] EventFields = ["timestamp", "rhythm", "structure", "key", "chord", "melody"];

    /// <summary>Index of a field in <see cref="EventFields"/>; the grammar orders by this.</summary>
    public static int FieldIndex(string field) => Array.IndexOf(EventFields, field);

    /// <summary>Time-signature pairs, in id order: numerator 1..32 against six denominators.</summary>
    public IReadOnlyList<(int Numerator, int Denominator)> MeterPairs { get; }

    /// <summary>Chord labels, in id order. "N" is no-chord; the rest are <c>root:quality[/inversion]</c>.</summary>
    public IReadOnlyList<string> FullChordLabels { get; }

    /// <summary>Total vocabulary size — the decoder's output width.</summary>
    public int TokenCount { get; }

    /// <summary>Each field's half-open id range, in layout order.</summary>
    public IReadOnlyList<(string Name, int Start, int End)> Ranges { get; }

    public int SubbeatShiftStart { get; }
    public int SubbeatShiftEnd { get; }
    public int TimeStart { get; }
    public int TimeEnd { get; }
    public int MeterStart { get; }
    public int MeterEnd { get; }
    public int EighthPositionStart { get; }
    public int EighthPositionEnd { get; }
    public int StructureStart { get; }
    public int StructureEnd { get; }
    public int KeyStart { get; }
    public int KeyEnd { get; }
    public int MajMinChordStart { get; }
    public int MajMinChordEnd { get; }
    public int FullChordStart { get; }
    public int FullChordEnd { get; }
    public int PitchStart { get; }
    public int PitchEnd { get; }
    public int DurationStart { get; }
    public int DurationEnd { get; }

    public ScoreTokenizer()
    {
        List<(int, int)> meters = new(192);
        for (int numerator = 1; numerator <= 32; numerator++)
        {
            foreach (int denominator in (int[])[1, 2, 4, 8, 16, 32])
            {
                meters.Add((numerator, denominator));
            }
        }
        MeterPairs = meters;

        // Five qualities carry their inversions; the rest are root position only.
        Dictionary<string, string[]> inversions = new(StringComparer.Ordinal)
        {
            ["maj"] = ["/2", "/3", "/5"],
            ["min"] = ["/2", "/b3", "/5"],
            ["maj7"] = ["/3", "/5", "/7"],
            ["min7"] = ["/b3", "/5", "/b7"],
            ["7"] = ["/3", "/5", "/b7"],
        };
        List<string> chords = ["N"];
        foreach (string quality in (string[])["maj", "min", "dim", "aug", "maj7", "min7", "7", "hdim7", "dim7",
            "minmaj7", "sus2", "sus4", "sus4(b7)", "maj6", "min6"])
        {
            foreach (string root in ChromaticSharps)
            {
                foreach (string inversion in inversions.TryGetValue(quality, out string[]? inv) ? [.. inv, ""] : (string[])[""])
                {
                    chords.Add($"{root}:{quality}{inversion}");
                }
            }
        }
        FullChordLabels = chords;

        int offset = RangeOffset;
        List<(string, int, int)> ranges = [];
        // The public range name differs from the field name for the two chord vocabularies, because token_type
        // reports what KIND of chord an id is, not which attribute holds it.
        (string Field, string Name, int Count)[] layout =
        [
            ("subbeat_shift", "subbeat_shift", 257),
            ("time", "time", 30_000),
            ("meter", "meter", 192),
            ("eighth_position", "eighth_position", 256),
            ("structure", "structure", StructureLabels.Length),
            ("key", "key", 24),
            ("majmin_chord", "chord_majmin", 25),
            ("full_chord", "chord_full", chords.Count),
            ("pitch", "pitch", 256),
            ("duration", "duration", DurationTemplates.Length),
        ];
        Dictionary<string, (int Start, int End)> bounds = new(StringComparer.Ordinal);
        foreach ((string field, string name, int count) in layout)
        {
            bounds[field] = (offset, offset + count);
            ranges.Add((name, offset, offset + count));
            offset += count;
        }
        Ranges = ranges;
        TokenCount = offset;

        (SubbeatShiftStart, SubbeatShiftEnd) = bounds["subbeat_shift"];
        (TimeStart, TimeEnd) = bounds["time"];
        (MeterStart, MeterEnd) = bounds["meter"];
        (EighthPositionStart, EighthPositionEnd) = bounds["eighth_position"];
        (StructureStart, StructureEnd) = bounds["structure"];
        (KeyStart, KeyEnd) = bounds["key"];
        (MajMinChordStart, MajMinChordEnd) = bounds["majmin_chord"];
        (FullChordStart, FullChordEnd) = bounds["full_chord"];
        (PitchStart, PitchEnd) = bounds["pitch"];
        (DurationStart, DurationEnd) = bounds["duration"];
    }

    /// <summary>The fixed prefix every decode starts from. The literal ids are task-selection tokens the
    /// checkpoint was trained with; they are not derived from anything.</summary>
    public static int[] PromptPrefix() => [SosToken, 4, 5, 6, 7, 9, 11, OutToken];

    /// <summary>Which vocabulary an id belongs to.</summary>
    public string TokenType(int token)
    {
        foreach ((string name, int start, int end) in Ranges)
        {
            if (token >= start && token < end) return name;
        }
        return token switch
        {
            PadToken => "pad",
            SosToken => "sos",
            EosToken => "eos",
            OutToken => "out",
            _ => "prompt",
        };
    }

    /// <summary>Hundredths of a second a time token names.</summary>
    public int TokenToTimeId(int token) => token - TimeStart;

    /// <summary>Seconds a time token names.</summary>
    public double TokenToSeconds(int token) => TokenToTimeId(token) / (double)TimeHz;
}
