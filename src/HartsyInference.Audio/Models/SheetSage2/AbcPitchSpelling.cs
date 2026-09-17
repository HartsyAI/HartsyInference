using System.Globalization;
using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Models.SheetSage2;

/// <summary>How SheetSage2's chord, key and pitch labels are spelled in ABC.
///
/// <para>Spelling is chosen against the key signature rather than from a fixed table, because the two target
/// parsers (abc2midi and SymMusic) propagate an accidental to the same note letter in every octave until the next
/// barline. Writing every pitch with its own accidental would still sound right, but it would not survive a
/// round trip through either parser's own output.</para></summary>
public static partial class AbcPitchSpelling
{
    /// <summary>Note letters in scale order, which is also the index order of a key-accidental vector.</summary>
    public const string Letters = "CDEFGAB";

    /// <summary>MIDI note number of middle C, the octave the ABC bare capital letters sit in.</summary>
    private const int MiddleC = 60;

    /// <summary>Chord labels meaning "no chord", which are written as nothing at all.</summary>
    public static readonly IReadOnlySet<string> NoChords =
        new HashSet<string>(StringComparer.Ordinal) { "N", "X", "?" };

    private static readonly int[] _naturalPitchClass = [0, 2, 4, 5, 7, 9, 11];

    private static readonly string[] _sharpPitchNames =
        ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

    private static readonly string[] _flatPitchNames =
        ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];

    /// <summary>Semitones above the root for each degree of the major scale, which is what a slash-bass degree
    /// is measured against.</summary>
    private static readonly int[] _majorScaleSemitones = [0, 2, 4, 5, 7, 9, 11];

    private static readonly Dictionary<string, string> _qualityToAbc = new(StringComparer.Ordinal)
    {
        ["maj"] = "",
        ["min"] = "m",
        ["dim"] = "dim",
        ["aug"] = "aug",
        ["7"] = "7",
        ["maj7"] = "maj7",
        ["min7"] = "m7",
        ["dim7"] = "dim7",
        ["hdim7"] = "m7b5",
        ["sus4"] = "sus4",
        ["sus2"] = "sus2",
        ["maj6"] = "6",
        ["min6"] = "m6",
        ["sus4(b7)"] = "7sus4",
        // abc2midi and SymMusic both accept the parenthesized major seventh; the common aliases mmaj7 and mM7
        // make abc2midi emit a diagnostic instead.
        ["minmaj7"] = "m(maj7)",
    };

    /// <summary>Signed accidental count of every key signature ABC can name: positive is sharps, negative flats.</summary>
    private static readonly Dictionary<string, int> _keySignatureAccidentals = new(StringComparer.Ordinal)
    {
        ["C"] = 0, ["G"] = 1, ["D"] = 2, ["A"] = 3, ["E"] = 4, ["B"] = 5, ["F#"] = 6, ["C#"] = 7,
        ["F"] = -1, ["Bb"] = -2, ["Eb"] = -3, ["Ab"] = -4, ["Db"] = -5, ["Gb"] = -6, ["Cb"] = -7,
        ["Am"] = 0, ["Em"] = 1, ["Bm"] = 2, ["F#m"] = 3, ["C#m"] = 4, ["G#m"] = 5, ["D#m"] = 6, ["A#m"] = 7,
        ["Dm"] = -1, ["Gm"] = -2, ["Cm"] = -3, ["Fm"] = -4, ["Bbm"] = -5, ["Ebm"] = -6, ["Abm"] = -7,
    };

    /// <summary>How each pitch class is spelled under a key signature of the given accidental count. Remote keys
    /// need double accidentals — MIDI G is F## in G# minor — which is why this is a table per signature rather
    /// than one sharp and one flat row.</summary>
    private static readonly Dictionary<int, string[]> _keyRelativePitchNames = new()
    {
        [7] = ["B#", "C#", "C##", "D#", "D##", "E#", "F#", "F##", "G#", "G##", "A#", "B"],
        [6] = ["B#", "C#", "C##", "D#", "E", "E#", "F#", "F##", "G#", "G##", "A#", "B"],
        [5] = ["B#", "C#", "C##", "D#", "E", "E#", "F#", "F##", "G#", "A", "A#", "B"],
        [4] = ["B#", "C#", "D", "D#", "E", "E#", "F#", "F##", "G#", "A", "A#", "B"],
        [3] = ["B#", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B"],
        [2] = ["C", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B"],
        [1] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
        [0] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B"],
        [-1] = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"],
        [-2] = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"],
        [-3] = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"],
        [-4] = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"],
        [-5] = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "Cb"],
        [-6] = ["C", "Db", "D", "Eb", "Fb", "F", "Gb", "G", "Ab", "A", "Bb", "Cb"],
        [-7] = ["C", "Db", "D", "Eb", "Fb", "F", "Gb", "G", "Ab", "Bbb", "Bb", "Cb"],
    };

    // \A and \z rather than ^ and $: these mirror Python's fullmatch, which does not tolerate a trailing newline.
    [GeneratedRegex(@"\A([A-G])(#{0,2}|b{0,2})\z")]
    private static partial Regex RootRegex();

    [GeneratedRegex(@"\A(#{0,2}|b{0,2})([1-9]|1[0-3])\z")]
    private static partial Regex BassDegreeRegex();

    /// <summary>Respells a root that ABC's chord-symbol syntax cannot carry.</summary>
    /// <param name="preserveDouble">Keeps a double accidental, which chord roots need and key signatures do not.</param>
    public static string PortablePitchName(string root, bool preserveDouble = false)
    {
        (int pitchClass, _, string accidental) = PitchClassOf(root);
        if (preserveDouble || accidental.Length <= 1) return root;
        return (accidental[0] == '#' ? _sharpPitchNames : _flatPitchNames)[pitchClass];
    }

    /// <summary>Writes a decoded <c>root:quality[/bass]</c> chord label as an ABC chord symbol.</summary>
    /// <returns>The symbol, or null when the label means "no chord".</returns>
    /// <exception cref="ChordSymbolException">The label has no separator, or a quality with no ABC spelling.</exception>
    public static string? ChordSymbolToAbc(string chord)
    {
        ArgumentNullException.ThrowIfNull(chord);
        chord = chord.Trim();
        if (NoChords.Contains(chord)) return null;
        int separator = chord.IndexOf(':');
        if (separator < 0)
        {
            throw new ChordSymbolException($"Chord '{chord}' is missing the ':' quality separator.");
        }
        string root = chord[..separator];
        string descriptor = chord[(separator + 1)..];
        int slash = descriptor.IndexOf('/');
        string quality = slash < 0 ? descriptor : descriptor[..slash];
        string? bassDegree = slash < 0 ? null : descriptor[(slash + 1)..];
        if (!_qualityToAbc.TryGetValue(quality, out string? suffix))
        {
            throw new ChordSymbolException(
                $"Unsupported chord quality '{quality}' in '{chord}'; refusing to rewrite it as major.");
        }
        string text = PortablePitchName(root, preserveDouble: true) + suffix;
        // An empty bass after the slash is dropped rather than refused, as in the reference.
        return bassDegree is { Length: > 0 } ? text + "/" + BassDegreeToPitch(root, bassDegree) : text;
    }

    /// <summary>Writes a decoded <c>root:major|minor</c> key label as an ABC key signature.</summary>
    /// <exception cref="AbcRebuildException">The mode is neither major nor minor, or no enharmonic spelling of the
    /// root has an ABC key signature.</exception>
    public static string KeySymbolToAbc(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        key = key.Trim();
        string root;
        string mode;
        int separator = key.IndexOf(':');
        if (separator >= 0)
        {
            root = key[..separator];
            mode = key[(separator + 1)..];
            if (mode != "major" && mode != "minor")
            {
                throw new AbcRebuildException($"Unsupported key mode '{mode}' in '{key}'.");
            }
        }
        else if (key.EndsWith('m'))
        {
            root = key[..^1];
            mode = "minor";
        }
        else
        {
            root = key;
            mode = "major";
        }
        string suffix = mode == "minor" ? "m" : "";
        (int rootPitchClass, _, string accidental) = PitchClassOf(root);
        string candidate = PortablePitchName(root) + suffix;
        if (_keySignatureAccidentals.ContainsKey(candidate)) return candidate;
        bool preferFlats = accidental.Contains('b');
        candidate = (preferFlats ? _flatPitchNames : _sharpPitchNames)[rootPitchClass] + suffix;
        if (!_keySignatureAccidentals.ContainsKey(candidate))
        {
            candidate = (preferFlats ? _sharpPitchNames : _flatPitchNames)[rootPitchClass] + suffix;
        }
        if (!_keySignatureAccidentals.ContainsKey(candidate))
        {
            throw new AbcRebuildException($"Cannot encode portable ABC key for '{key}'.");
        }
        return candidate;
    }

    /// <summary>Accidental per note letter for an ABC key signature, indexed as <see cref="Letters"/>.</summary>
    /// <exception cref="AbcRebuildException">The key is not one ABC can name.</exception>
    public static int[] GetKeyAccidentals(string key)
    {
        if (!_keySignatureAccidentals.TryGetValue(key, out int count))
        {
            throw new AbcRebuildException($"Unsupported ABC key signature '{key}'.");
        }
        int[] accidentals = new int[Letters.Length];
        string order = count > 0 ? "FCGDAEB" : "BEADGCF";
        for (int i = 0; i < Math.Abs(count); i++)
        {
            accidentals[Letters.IndexOf(order[i])] = count > 0 ? 1 : -1;
        }
        return accidentals;
    }

    /// <summary>Writes one MIDI pitch, marking only the accidentals the bar has not already established.</summary>
    /// <param name="measureAccidentals">Accidental per note letter written so far in this bar. The caller clears
    /// it at every barline and after an inline key change, because both parsers reset there.</param>
    /// <exception cref="AbcRebuildException">The key signature has more accidentals than ABC can spell against.</exception>
    public static string NoteToAbc(int note, IReadOnlyList<int> keyAccidentals, Dictionary<int, int> measureAccidentals)
    {
        ArgumentNullException.ThrowIfNull(keyAccidentals);
        ArgumentNullException.ThrowIfNull(measureAccidentals);
        int accidentalCount = 0;
        for (int i = 0; i < keyAccidentals.Count; i++)
        {
            accidentalCount += keyAccidentals[i];
        }
        if (!_keyRelativePitchNames.TryGetValue(accidentalCount, out string[]? names))
        {
            throw new AbcRebuildException($"Unsupported key signature accidental count {accidentalCount}.");
        }
        int pitchClass = Mod(note, 12);
        string pitchName = names[pitchClass];
        char letter = pitchName[0];
        int accidentalNumber = AccidentalNumber(pitchName.AsSpan(1));
        int octave = FloorDiv(note - MiddleC, 12);
        // Cb and B# cross the MIDI octave boundary even though their written note letter does not.
        if (pitchClass == 11 && accidentalNumber == -1) octave++;
        else if (pitchClass == 0 && accidentalNumber == 1) octave--;

        int scaleIndex = Letters.IndexOf(letter);
        int current = measureAccidentals.TryGetValue(scaleIndex, out int written) ? written : keyAccidentals[scaleIndex];
        string accidentalText = "";
        if (current != accidentalNumber)
        {
            measureAccidentals[scaleIndex] = accidentalNumber;
            accidentalText = AccidentalText(accidentalNumber);
        }
        if (octave > 0)
        {
            return accidentalText + char.ToLowerInvariant(letter) + new string('\'', Math.Max(0, octave - 1));
        }
        return accidentalText + letter + new string(',', Math.Max(0, -octave));
    }

    /// <summary>Pitch class, note letter and accidental of a spelled root.</summary>
    /// <exception cref="ChordSymbolException">The text is not a note letter with at most two matching accidentals.</exception>
    private static (int PitchClass, char Letter, string Accidental) PitchClassOf(string root)
    {
        Match match = RootRegex().Match(root);
        if (!match.Success) throw new ChordSymbolException($"Invalid pitch spelling '{root}'.");
        char letter = match.Groups[1].Value[0];
        string accidental = match.Groups[2].Value;
        int offset = Count(accidental, '#') - Count(accidental, 'b');
        return (Mod(_naturalPitchClass[Letters.IndexOf(letter)] + offset, 12), letter, accidental);
    }

    /// <summary>Resolves a slash-bass, which the vocabulary writes as a scale degree such as <c>/b7</c>, to a
    /// spelled pitch. A degree that already names a pitch is kept as written.</summary>
    /// <exception cref="ChordSymbolException">The text is neither a pitch nor a degree 1-13 with an accidental.</exception>
    private static string BassDegreeToPitch(string root, string degreeText)
    {
        if (RootRegex().IsMatch(degreeText)) return PortablePitchName(degreeText, preserveDouble: true);
        Match match = BassDegreeRegex().Match(degreeText);
        if (!match.Success) throw new ChordSymbolException($"Invalid chord bass degree '{degreeText}'.");
        (int rootPitchClass, char rootLetter, string rootAccidental) = PitchClassOf(root);
        string degreeAccidental = match.Groups[1].Value;
        int degree = int.Parse(match.Groups[2].Value, CultureInfo.InvariantCulture);
        int interval = _majorScaleSemitones[(degree - 1) % 7] + 12 * ((degree - 1) / 7);
        interval += Count(degreeAccidental, '#') - Count(degreeAccidental, 'b');
        int targetPitchClass = Mod(rootPitchClass + interval, 12);

        int targetLetterIndex = Mod(Letters.IndexOf(rootLetter) + degree - 1, 7);
        char targetLetter = Letters[targetLetterIndex];
        int difference = Mod(targetPitchClass - _naturalPitchClass[targetLetterIndex] + 6, 12) - 6;
        if (difference >= -2 && difference <= 2) return targetLetter + AccidentalSpelling(difference);
        bool sharpSide = rootAccidental.Contains('#')
            || degreeAccidental.Contains('#');
        return (sharpSide ? _sharpPitchNames : _flatPitchNames)[targetPitchClass];
    }

    private static string AccidentalSpelling(int accidental) => accidental switch
    {
        -2 => "bb",
        -1 => "b",
        0 => "",
        1 => "#",
        _ => "##",
    };

    private static string AccidentalText(int accidental) => accidental switch
    {
        -2 => "__",
        -1 => "_",
        0 => "=",
        1 => "^",
        _ => "^^",
    };

    private static int AccidentalNumber(ReadOnlySpan<char> accidental) => accidental.Length switch
    {
        0 => 0,
        1 => accidental[0] == '#' ? 1 : -1,
        _ => accidental[0] == '#' ? 2 : -2,
    };

    private static int Count(string text, char value)
    {
        int count = 0;
        foreach (char c in text)
        {
            if (c == value) count++;
        }
        return count;
    }

    /// <summary>Python's modulo, which is never negative.</summary>
    private static int Mod(int value, int modulus)
    {
        int result = value % modulus;
        return result < 0 ? result + modulus : result;
    }

    /// <summary>Python's floor division, which rounds towards negative infinity.</summary>
    private static int FloorDiv(int value, int divisor)
    {
        int quotient = value / divisor;
        return value % divisor != 0 && (value < 0) != (divisor < 0) ? quotient - 1 : quotient;
    }
}
