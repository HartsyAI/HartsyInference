using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text;

namespace HartsyInference.Audio.Frontends;

/// <summary>GPT-SoVITS English (en) G2P: maps English text to the model's ARPAbet phoneme symbols (the
/// stress-marked set in <see cref="GptSoVitsSymbols"/>). Words are looked up in the embedded CMUdict
/// (<c>cmudict.rep</c>); short out-of-vocabulary tokens fall back to letter-by-letter spelling and possessive
/// <c>'s</c> splitting (the deterministic part of upstream <c>en_G2p.qryword</c>).
///
/// <para><b>Coverage:</b> CMUdict (~125k words) covers the vast majority of text. Longer OOV words that
/// upstream resolves with the g2p_en neural LTS, and POS-based homograph disambiguation, are accuracy
/// refinements not yet ported — such words are letter-spelled or skipped, never producing a non-symbol.</para></summary>
public static class GptSoVitsEnglishG2P
{
    private const string DictResource = "HartsyInference.Audio.Frontends.Resources.cmudict.rep";
    private static readonly Dictionary<string, string[]> _cmu = LoadDict();

    private static Dictionary<string, string[]> LoadDict()
    {
        Dictionary<string, string[]> dict = new(130_000);
        Assembly asm = typeof(GptSoVitsEnglishG2P).Assembly;
        using Stream? s = asm.GetManifestResourceStream(DictResource)
            ?? throw new InvalidOperationException($"Embedded resource '{DictResource}' is missing.");
        using StreamReader reader = new(s, Encoding.UTF8);
        string? line;
        int idx = 0;
        while ((line = reader.ReadLine()) is not null)
        {
            idx++;
            if (idx < 57) continue;     // skip the CMUdict license header (read_dict_new)
            int sep = line.IndexOf("  ", System.StringComparison.Ordinal);
            if (sep <= 0) continue;
            string word = line[..sep].ToLowerInvariant();
            string[] phones = line[(sep + 2)..].Split(' ', System.StringSplitOptions.RemoveEmptyEntries);
            dict.TryAdd(word, phones);
        }
        return dict;
    }

    /// <summary>Converts English text to ARPAbet phoneme symbols. Tokenizes into words and punctuation; every
    /// emitted phone is guaranteed to be in the model symbol set.</summary>
    public static List<string> G2P(string text)
    {
        List<string> phones = [];
        foreach (string token in Tokenize(text))
        {
            if (token.Length == 1 && PinyinPhonemizer.IsPunctuation(token))
            {
                phones.Add(token);
                continue;
            }
            foreach (string ph in QueryWord(token))
                if (GptSoVitsSymbols.IdOf(ph) >= 0) phones.Add(ph);
        }
        // Upstream: pad short utterances so the model has enough context.
        if (phones.Count < 4) phones.Insert(0, ",");
        return phones;
    }

    /// <summary>Word → ARPAbet phones: CMUdict lookup, then the deterministic OOV fallbacks (possessive split,
    /// letter-spelling for short tokens).</summary>
    private static IEnumerable<string> QueryWord(string oWord)
    {
        string word = oWord.ToLowerInvariant();
        if (word.Length > 1 && _cmu.TryGetValue(word, out string[]? p)) return p;

        if (word.Length <= 3)
        {
            List<string> phones = [];
            foreach (char w in word)
            {
                if (w == 'a') phones.Add("EY1");
                else if (!char.IsLetter(w)) phones.Add(w.ToString());
                else if (_cmu.TryGetValue(w.ToString(), out string[]? lp)) phones.AddRange(lp);
            }
            return phones;
        }
        if (word.EndsWith("'s", System.StringComparison.Ordinal))
        {
            List<string> phones = [.. QueryWord(word[..^2])];
            // 's voicing follows the final phone (S after unvoiced, Z otherwise) — approximate with Z.
            if (phones.Count > 0) phones.Add(phones[^1] is "T" or "P" or "K" or "F" or "TH" ? "S" : "Z");
            return phones;
        }
        return [];     // long OOV: would use the g2p_en neural LTS (not yet ported)
    }

    /// <summary>Splits text into lowercase word tokens and single-character punctuation (normalized to the
    /// symbol set). Whitespace and unsupported characters are dropped.</summary>
    private static IEnumerable<string> Tokenize(string text)
    {
        StringBuilder word = new();
        foreach (char ch in text)
        {
            if (char.IsLetter(ch) || ch == '\'')
            {
                word.Append(ch);
                continue;
            }
            if (word.Length > 0) { yield return word.ToString(); word.Clear(); }
            string? punct = ch switch
            {
                '.' or ';' => ".",
                ',' => ",",
                '!' => "!",
                '?' => "?",
                '-' => "-",
                _ => null,
            };
            if (punct is not null) yield return punct;
        }
        if (word.Length > 0) yield return word.ToString();
    }
}
