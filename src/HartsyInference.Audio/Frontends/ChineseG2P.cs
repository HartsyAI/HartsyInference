using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text;

namespace HartsyInference.Audio.Frontends;

/// <summary>GPT-SoVITS Chinese (zh) G2P: maps Han text to the model's phonemes plus <c>word2ph</c>. Each
/// character's default pinyin (initial + tone-3 final, from the embedded pypinyin-derived dictionary) feeds the
/// validated <see cref="PinyinPhonemizer"/>; punctuation is normalized to the model's symbol set.
///
/// <para><b>Coverage:</b> this is the per-character default-pronunciation path (faithful to pypinyin's default
/// reading). Polyphone disambiguation (g2pW), tone sandhi, phrase pinyin, and erhua are accuracy refinements
/// that layer on top by supplying a different (initial, final) per character — they plug into the same
/// <see cref="PinyinPhonemizer"/> assembly. Number/date text-normalization (zh_normalization) is applied by the
/// caller / a future normalizer.</para></summary>
public static class ChineseG2P
{
    private const string DictResource = "HartsyInference.Audio.Frontends.Resources.pinyin_char_dict.txt";

    // chinese2.rep_map: normalize punctuation to the model's symbol set.
    private static readonly Dictionary<char, string> _punctMap = new()
    {
        ['：'] = ",", ['；'] = ",", ['，'] = ",", ['。'] = ".", ['！'] = "!", ['？'] = "?",
        ['\n'] = ".", ['·'] = ",", ['、'] = ",", ['$'] = ".", ['/'] = ",", ['—'] = "-",
        ['~'] = "…", ['～'] = "…", [','] = ",", ['.'] = ".", ['!'] = "!", ['?'] = "?", ['-'] = "-",
    };

    private static readonly Dictionary<int, (string Initial, string Final)> _charDict = LoadDict();

    private static Dictionary<int, (string, string)> LoadDict()
    {
        Dictionary<int, (string, string)> map = new(42_000);
        Assembly asm = typeof(ChineseG2P).Assembly;
        using Stream? s = asm.GetManifestResourceStream(DictResource);
        if (s is null)
            throw new InvalidOperationException($"Embedded resource '{DictResource}' is missing.");
        using StreamReader reader = new(s, Encoding.UTF8);
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            if (line.Length == 0) continue;
            int t1 = line.IndexOf('\t');
            int t2 = line.IndexOf('\t', t1 + 1);
            if (t1 <= 0 || t2 <= t1) continue;
            int codepoint = char.ConvertToUtf32(line, 0);
            string ini = line[(t1 + 1)..t2];
            string fin = line[(t2 + 1)..];
            map[codepoint] = (ini, fin);
        }
        return map;
    }

    /// <summary>Converts normalized Han text to (phonemes, word2ph). Han characters use their default pinyin;
    /// recognized punctuation becomes a single symbol; unknown characters are skipped.</summary>
    public static (List<string> Phones, List<int> Word2Ph) G2P(string text)
    {
        // "..." → "…" before per-character mapping (the only multi-char punctuation rule).
        text = text.Replace("...", "…");

        List<string> initials = [];
        List<string> finals = [];
        foreach (System.Text.Rune rune in text.EnumerateRunes())
        {
            int cp = rune.Value;
            if (cp <= char.MaxValue && _punctMap.TryGetValue((char)cp, out string? punct))
            {
                initials.Add(punct);
                finals.Add(punct);
            }
            else if (cp == '…')
            {
                initials.Add("…");
                finals.Add("…");
            }
            else if (_charDict.TryGetValue(cp, out (string Initial, string Final) py))
            {
                initials.Add(py.Initial);
                finals.Add(py.Final);
            }
            // else: unknown character (e.g. latin in a zh segment) — skipped, matching the upstream eng skip.
        }
        return PinyinPhonemizer.Assemble(initials, finals);
    }

    /// <summary>Default-pinyin lookup for a single Han character (initial + tone-3 final), or null if unknown.</summary>
    public static (string Initial, string Final)? CharPinyin(int codepoint)
        => _charDict.TryGetValue(codepoint, out (string, string) py) ? py : null;
}
