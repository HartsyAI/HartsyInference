using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Frontends;

/// <summary>Port of pypinyin 0.55.0's <c>lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)</c>
/// (mozillazg/python-pinyin, MIT) over its <c>phrases_dict.json</c> and <c>pinyin_dict.json</c>: text splits into Han
/// and non-Han runs, Han runs into dictionary phrases by pypinyin's strict forward maximum matching, and every
/// character takes its phrase reading or else its first single-character reading, converted to tone-number style with
/// 5 for the neutral tone. A character with no reading is kept as written, as <c>errors='default'</c> keeps it.</summary>
internal sealed partial class PinyinConverter
{
    // pypinyin phonetic_symbol: tone-marked letters to tone2 spellings (ü alone becomes v).
    private static readonly Dictionary<char, string> ToneMarks = new()
    {
        ['ā'] = "a1", ['á'] = "a2", ['ǎ'] = "a3", ['à'] = "a4", ['ē'] = "e1", ['é'] = "e2", ['ě'] = "e3", ['è'] = "e4",
        ['ō'] = "o1", ['ó'] = "o2", ['ǒ'] = "o3", ['ò'] = "o4", ['ī'] = "i1", ['í'] = "i2", ['ǐ'] = "i3", ['ì'] = "i4",
        ['ū'] = "u1", ['ú'] = "u2", ['ǔ'] = "u3", ['ù'] = "u4", ['ü'] = "v", ['ǖ'] = "v1", ['ǘ'] = "v2", ['ǚ'] = "v3",
        ['ǜ'] = "v4", ['ń'] = "n2", ['ň'] = "n3", ['ǹ'] = "n4", ['ḿ'] = "m2", ['ế'] = "ê2", ['ề'] = "ê4",
    };
    private static readonly (string From, string To)[] MultiCharToneMarks =
        [("m̄", "m1"), ("m̀", "m4"), ("ê̄", "ê1"), ("ê̌", "ê3")];

    private readonly Dictionary<string, string[]> _phrases;
    private readonly HashSet<string> _prefixes;
    private readonly Dictionary<int, string> _single;

    private PinyinConverter(Dictionary<string, string[]> phrases, Dictionary<int, string> single)
    {
        _phrases = phrases;
        _single = single;
        _prefixes = new HashSet<string>(StringComparer.Ordinal);
        foreach (string phrase in phrases.Keys)
        {
            for (int i = 1; i <= phrase.Length; i++) _prefixes.Add(phrase[..i]);
        }
    }

    /// <summary>Loads <c>phrases_dict.json</c> (<c>{phrase: [[reading, ...], ...]}</c>) and <c>pinyin_dict.json</c>
    /// (<c>{codepoint: "reading,reading"}</c>).</summary>
    public static PinyinConverter FromStreams(Stream phrases, Stream singles)
    {
        ArgumentNullException.ThrowIfNull(phrases);
        ArgumentNullException.ThrowIfNull(singles);
        Dictionary<string, string[]> phraseReadings = new(50_000, StringComparer.Ordinal);
        using (JsonDocument doc = JsonDocument.Parse(phrases))
        {
            foreach (JsonProperty phrase in doc.RootElement.EnumerateObject())
            {
                List<string> readings = [];
                foreach (JsonElement item in phrase.Value.EnumerateArray())
                {
                    // heteronym=False: the first reading of each character, converted alone.
                    readings.Add(ToTone3(item.GetArrayLength() == 0 ? "" : item[0].GetString() ?? ""));
                }
                phraseReadings[phrase.Name] = [.. readings];
            }
        }
        Dictionary<int, string> single = new(45_000);
        using (JsonDocument doc = JsonDocument.Parse(singles))
        {
            foreach (JsonProperty entry in doc.RootElement.EnumerateObject())
            {
                int codePoint = int.Parse(entry.Name, NumberStyles.None, CultureInfo.InvariantCulture);
                string readings = entry.Value.GetString() ?? "";
                int comma = readings.IndexOf(',', StringComparison.Ordinal);
                single[codePoint] = ToTone3(comma < 0 ? readings : readings[..comma]);
            }
        }
        return new PinyinConverter(phraseReadings, single);
    }

    /// <summary>Loads both files from disk.</summary>
    public static PinyinConverter FromFiles(string phrasesPath, string singlesPath)
    {
        using FileStream phrases = File.OpenRead(phrasesPath);
        using FileStream singles = File.OpenRead(singlesPath);
        return FromStreams(phrases, singles);
    }

    /// <summary><c>lazy_pinyin(text, style=TONE3, neutral_tone_with_five=True)</c>: one entry per character of a Han
    /// run, one per non-Han run (kept as written).</summary>
    public List<string> LazyPinyin(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        List<string> result = [];
        int i = 0;
        while (i < text.Length)
        {
            int start = i;
            bool han = IsHans(text, i);
            while (i < text.Length && IsHans(text, i) == han) i += CharLength(text, i);
            string run = text[start..i];
            if (han) Convert(run, result);
            else result.Add(run);
        }
        return result;
    }

    // simpleseg.seg -> mmseg.seg.cut (no_non_phrases=True), then each piece's phrase or single-character readings.
    private void Convert(string han, List<string> result)
    {
        string remain = han;
        while (remain.Length != 0)
        {
            string lastValid = "";
            bool brokeOut = false;
            int index = 0;
            while (index < remain.Length)
            {
                index += CharLength(remain, index);
                string word = remain[..index];
                if (!_prefixes.Contains(word))
                {
                    string piece = lastValid.Length != 0 ? lastValid : remain[..CharLength(remain, 0)];
                    AppendPiece(piece, result);
                    remain = remain[piece.Length..];
                    brokeOut = true;
                    break;
                }
                if (_phrases.ContainsKey(word)) lastValid = word;
            }
            if (brokeOut) continue;
            if (lastValid.Length != 0)
            {
                AppendPiece(lastValid, result);
                remain = remain[lastValid.Length..];
                continue;
            }
            // The whole remainder is a phrase prefix but holds no phrase: every character stands alone.
            for (int j = 0; j < remain.Length; j += CharLength(remain, j))
            {
                AppendPiece(remain.Substring(j, CharLength(remain, j)), result);
            }
            break;
        }
    }

    private void AppendPiece(string piece, List<string> result)
    {
        if (_phrases.TryGetValue(piece, out string[]? readings))
        {
            result.AddRange(readings);
            return;
        }
        for (int i = 0; i < piece.Length; i += CharLength(piece, i))
        {
            int cp = char.ConvertToUtf32(piece, i);
            result.Add(_single.TryGetValue(cp, out string? reading) ? reading
                : ToTone3(piece.Substring(i, CharLength(piece, i))));
        }
    }

    /// <summary>pypinyin's TONE3 style with <c>neutral_tone_with_five</c>: tone marks become a trailing tone number,
    /// <c>ü</c> becomes <c>v</c>, and a reading with no tone number gets 5.</summary>
    internal static string ToTone3(string pinyin)
    {
        StringBuilder sb = new(pinyin.Length + 1);
        foreach (char c in pinyin)
        {
            if (ToneMarks.TryGetValue(c, out string? to)) sb.Append(to);
            else sb.Append(c);
        }
        string tone2 = sb.ToString();
        foreach ((string from, string to) in MultiCharToneMarks) tone2 = tone2.Replace(from, to, StringComparison.Ordinal);
        string tone3 = Tone3Regex().Replace(tone2, "$1$3$2");
        if (tone3.Length == 0 || DigitRegex().IsMatch(tone3)) return tone3;
        return tone3 + "5";
    }

    // pypinyin RE_HANS: the CJK blocks it holds readings for.
    private static bool IsHans(string text, int i)
    {
        int cp = char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1])
            ? char.ConvertToUtf32(text[i], text[i + 1]) : text[i];
        return cp == 0x3007 || (cp >= 0xe815 && cp <= 0xe864) || cp == 0xfa18 || (cp >= 0x3400 && cp <= 0x4dbf)
            || (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0x20000 && cp <= 0x2a6df)
            || (cp >= 0x2a703 && cp <= 0x2b73f) || (cp >= 0x2b740 && cp <= 0x2b81d) || (cp >= 0x2b825 && cp <= 0x2bf6e)
            || (cp >= 0x2c029 && cp <= 0x2ce93) || cp == 0x2d016 || (cp >= 0x2d11b && cp <= 0x2ebd9)
            || (cp >= 0x2f80a && cp <= 0x2fa1f) || (cp >= 0x30000 && cp <= 0x3134a) || (cp >= 0x31350 && cp <= 0x32389);
    }

    private static int CharLength(string text, int i) =>
        char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]) ? 2 : 1;

    [GeneratedRegex("^([a-zêü]+)([1-5])([a-zêü]*)$")]
    private static partial Regex Tone3Regex();

    [GeneratedRegex(@"\d")]
    private static partial Regex DigitRegex();
}
