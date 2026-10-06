using System.Text;

namespace HartsyInference.Audio.Frontends;

/// <summary>Port of misaki's <c>transcription.pinyin_to_ipa</c> (adapted by misaki, Apache-2.0, from stefantaubert's
/// pinyin-to-ipa, MIT) as <c>ZHG2P.py2ipa</c> uses it: the first IPA variant of a tone-number syllable, split into
/// pypinyin's strict initial and final, with <c>ZHG2P.retone</c>'s tone arrows (→ ↗ ↓ ↘, nothing for the neutral
/// tone) and its <c>ɨ</c> for the apical vowels of zi/ci/si and zhi/chi/shi/ri.</summary>
internal static class PinyinToIpa
{
    // Tone 1..5 as retone writes the Chao letters ˥ ˧˥ ˧˩˧ ˥˩ and the empty neutral tone.
    private static readonly string[] Tones = ["", "→", "↗", "↓", "↘", ""];
    // pypinyin _INITIALS, in its match order.
    private static readonly string[] StrictInitials =
        ["b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s"];
    private static readonly HashSet<string> Finals = new(StringComparer.Ordinal)
    {
        "i", "u", "ü", "a", "ia", "ua", "o", "uo", "e", "ie", "üe", "ai", "uai", "ei", "uei", "ao", "iao", "ou", "iou",
        "an", "ian", "uan", "üan", "en", "in", "uen", "ün", "ang", "iang", "uang", "eng", "ing", "ueng", "ong", "iong",
        "er", "ê",
    };
    // transcription.py INITIAL_MAPPING, first variant.
    private static readonly Dictionary<string, string> InitialIpa = new(StringComparer.Ordinal)
    {
        ["b"] = "p", ["c"] = "ʦʰ", ["ch"] = "ꭧʰ", ["d"] = "t", ["f"] = "f", ["g"] = "k", ["h"] = "x", ["j"] = "ʨ",
        ["k"] = "kʰ", ["l"] = "l", ["m"] = "m", ["n"] = "n", ["p"] = "pʰ", ["q"] = "ʨʰ", ["r"] = "ɻ", ["s"] = "s",
        ["sh"] = "ʂ", ["t"] = "tʰ", ["x"] = "ɕ", ["z"] = "ʦ", ["zh"] = "ꭧ",
    };
    // SYLLABIC_CONSONANT_MAPPINGS and INTERJECTION_MAPPINGS, first variant; 0 marks where the tone goes.
    private static readonly Dictionary<string, string[]> WholeSyllables = new(StringComparer.Ordinal)
    {
        ["hm"] = ["h", "m0"], ["hng"] = ["h", "ŋ0"], ["m"] = ["m0"], ["n"] = ["n0"], ["ng"] = ["ŋ0"],
        ["io"] = ["j", "ɔ0"], ["ê"] = ["ɛ0"], ["er"] = ["ɚ0"], ["o"] = ["ɔ0"],
    };
    // FINAL_MAPPING, first variant.
    private static readonly Dictionary<string, string[]> FinalIpa = new(StringComparer.Ordinal)
    {
        ["a"] = ["a0"], ["ai"] = ["ai̯0"], ["an"] = ["a0", "n"], ["ang"] = ["a0", "ŋ"], ["ao"] = ["au̯0"],
        ["e"] = ["ɤ0"], ["ei"] = ["ei̯0"], ["en"] = ["ə0", "n"], ["eng"] = ["ə0", "ŋ"], ["i"] = ["i0"],
        ["ia"] = ["j", "a0"], ["ian"] = ["j", "ɛ0", "n"], ["iang"] = ["j", "a0", "ŋ"], ["iao"] = ["j", "au̯0"],
        ["ie"] = ["j", "e0"], ["in"] = ["i0", "n"], ["iou"] = ["j", "ou̯0"], ["ing"] = ["i0", "ŋ"],
        ["iong"] = ["j", "ʊ0", "ŋ"], ["ong"] = ["ʊ0", "ŋ"], ["ou"] = ["ou̯0"], ["u"] = ["u0"], ["uei"] = ["w", "ei̯0"],
        ["ua"] = ["w", "a0"], ["uai"] = ["w", "ai̯0"], ["uan"] = ["w", "a0", "n"], ["uen"] = ["w", "ə0", "n"],
        ["uang"] = ["w", "a0", "ŋ"], ["ueng"] = ["w", "ə0", "ŋ"], ["uo"] = ["w", "o0"], ["o"] = ["w", "o0"],
        ["ü"] = ["y0"], ["üe"] = ["ɥ", "e0"], ["üan"] = ["ɥ", "ɛ0", "n"], ["ün"] = ["y0", "n"],
    };
    private static readonly HashSet<char> UTones = ['u', 'ū', 'ú', 'ǔ', 'ù'];
    private static readonly HashSet<char> ITones = ['i', 'ī', 'í', 'ǐ', 'ì'];

    /// <summary><c>ZHG2P.py2ipa</c> for one tone-number syllable (<c>zhong1</c>, <c>lv4</c>, <c>ma5</c>); null where
    /// misaki raises because the syllable has no initial/final reading.</summary>
    public static string? Convert(string tone3)
    {
        ArgumentNullException.ThrowIfNull(tone3);
        int tone = 5;
        StringBuilder normal = new(tone3.Length);
        bool toneFound = false;
        foreach (char c in tone3.Replace("5", "", StringComparison.Ordinal))
        {
            if (char.IsDigit(c))
            {
                if (!toneFound)
                {
                    tone = c - '0';
                    toneFound = true;
                }
                continue;
            }
            normal.Append(c == 'ü' ? 'v' : c);
        }
        if (tone is < 1 or > 5) return null;
        string py = normal.ToString();
        string[]? phonemes;
        string initial = "";
        if (!WholeSyllables.TryGetValue(py, out phonemes))
        {
            initial = Initial(py, StrictInitials);
            string final = StrictFinal(py.Replace('v', 'ü'));
            if (final.Length == 0) return null;
            if (final == "i" && initial is "zh" or "ch" or "sh" or "r") phonemes = ["ɻ̩0"];
            else if (final == "i" && initial is "z" or "c" or "s") phonemes = ["ɹ̩0"];
            else if (!FinalIpa.TryGetValue(final, out phonemes)) return null;
        }
        StringBuilder sb = new();
        if (initial.Length != 0) sb.Append(InitialIpa[initial]);
        foreach (string p in phonemes) sb.Append(p.Replace("0", Tones[tone], StringComparison.Ordinal));
        return sb.Replace("ɻ̩", "ɨ").Replace("ɹ̩", "ɨ").ToString();
    }

    private static string Initial(string pinyin, string[] initials)
    {
        foreach (string i in initials)
        {
            if (pinyin.StartsWith(i, StringComparison.Ordinal)) return i;
        }
        return "";
    }

    // pypinyin get_finals(strict=True) after standard.convert_finals.
    private static string StrictFinal(string pinyin)
    {
        pinyin = ConvertFinals(pinyin);
        string initial = Initial(pinyin, StrictInitials);
        string final = pinyin[initial.Length..];
        if (Finals.Contains(final)) return final;
        // y and w read as initials, so "yo" still finds its final.
        initial = pinyin.StartsWith('y') || pinyin.StartsWith('w') ? pinyin[..1] : initial;
        final = pinyin[initial.Length..];
        return Finals.Contains(final) ? final : "";
    }

    // pypinyin standard.convert_finals on a toneless syllable: zero-consonant y/w, ü after j/q/x, and the iu/ui/un
    // abbreviations written out.
    private static string ConvertFinals(string pinyin)
    {
        string raw = pinyin;
        if (raw.StartsWith('y'))
        {
            string rest = pinyin[1..];
            if (rest.Length > 0 && UTones.Contains(rest[0])) pinyin = "ü" + pinyin[2..];
            else if (rest.Length > 0 && ITones.Contains(rest[0])) pinyin = rest;
            else pinyin = "i" + rest;
        }
        if (raw.StartsWith('w'))
        {
            string rest = pinyin[1..];
            pinyin = rest.Length > 0 && UTones.Contains(rest[0]) ? rest : "u" + rest;
        }
        if (!Finals.Contains(pinyin)) pinyin = raw;
        if (pinyin.Length >= 2 && pinyin[0] is 'j' or 'q' or 'x' && pinyin[1] == 'u') pinyin = pinyin[0] + "ü" + pinyin[2..];
        if (pinyin.EndsWith("iu", StringComparison.Ordinal) && pinyin.Length > 2 && IsAsciiLower(pinyin[..^2], whole: true))
            pinyin = pinyin[..^2] + "iou";
        if (pinyin.EndsWith("ui", StringComparison.Ordinal) && pinyin.Length > 2 && IsAsciiLower(pinyin[..^2], whole: false))
            pinyin = pinyin[..^2] + "uei";
        if (pinyin.EndsWith("un", StringComparison.Ordinal) && pinyin.Length > 2 && IsAsciiLower(pinyin[..^2], whole: false))
            pinyin = pinyin[..^2] + "uen";
        return pinyin;
    }

    // IU_RE is anchored at the start (^[a-z]+iu$); UI_RE and UN_RE only need one [a-z] right before the ending.
    private static bool IsAsciiLower(string head, bool whole)
    {
        if (!whole) return char.IsAsciiLetterLower(head[^1]);
        foreach (char c in head)
        {
            if (!char.IsAsciiLetterLower(c)) return false;
        }
        return true;
    }
}
