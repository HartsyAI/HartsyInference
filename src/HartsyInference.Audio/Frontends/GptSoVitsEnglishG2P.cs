using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Frontends;

/// <summary>GPT-SoVITS English (en) G2P: upstream <c>text/english.py</c>. Text is normalized
/// (<see cref="GptSoVitsEnglishNormalizer"/>), split with NLTK's TweetTokenizer rules and POS-tagged
/// (<see cref="PerceptronPosTagger"/>); each word then takes the first of: punctuation as itself, single letters,
/// g2p_en homographs by tag, GPT-SoVITS's merged CMU dictionary (<c>cmudict.rep</c> + <c>cmudict-fast.rep</c> +
/// <c>engdict-hot.rep</c>), the name dictionary for title-case words, letter spelling up to three letters, the
/// possessive rule, <see cref="WordSegmenter"/> compounds, and the g2p_en neural letter-to-sound model
/// (<see cref="G2pEnLetterToSound"/>). Output phones are filtered to the v2 symbol set as upstream does.</summary>
public static partial class GptSoVitsEnglishG2P
{
    private static readonly Lazy<Dictionary<string, string[]>> Cmu = new(() => LoadDict("gptsovits_en_dict.txt.gz"));
    private static readonly Lazy<Dictionary<string, string[]>> Names = new(() => LoadDict("gptsovits_en_names.txt.gz"));
    private static readonly Lazy<Dictionary<string, Homograph>> Homographs = new(LoadHomographs);
    private static readonly HashSet<string> DroppedPhones = new(StringComparer.Ordinal) { " ", "<pad>", "UW", "</s>", "<s>" };

    /// <summary>Upstream <c>clean_text(text, "en")</c> phones: normalize, phonemize, and prefix a comma when fewer than
    /// four phones result.</summary>
    public static List<string> G2P(string text)
    {
        List<string> phones = Phonemize(Normalize(text));
        if (phones.Count < 4) phones.Insert(0, ",");
        return phones;
    }

    /// <summary>Upstream <c>english.text_normalize</c>; the text the phones are read from.</summary>
    public static string Normalize(string text) => GptSoVitsEnglishNormalizer.TextNormalize(text);

    /// <summary>Upstream <c>english.g2p</c> on already-normalized text: every phone is a v2 symbol.</summary>
    public static List<string> Phonemize(string normalizedText)
    {
        ArgumentNullException.ThrowIfNull(normalizedText);
        List<string> words = Tokenize(normalizedText);
        string[] tags = words.Count == 0 ? [] : PerceptronPosTagger.English.Tag(words);
        List<string> phones = [];
        for (int i = 0; i < words.Count; i++)
        {
            foreach (string ph in WordPhones(words[i], tags[i]))
            {
                if (DroppedPhones.Contains(ph)) continue;
                string p = ph == "<unk>" ? "UNK" : ph;
                if (GptSoVitsSymbols.IdOf(p) >= 0) phones.Add(p);
                else if (p == "'") phones.Add("-");
            }
        }
        return phones;
    }

    /// <summary>NLTK <c>TweetTokenizer().tokenize</c> restricted to what survives normalization (letters, space and
    /// <c>'.,?!-</c>): bare domains ("example.com"), words with inner apostrophes or hyphens, other word runs, and
    /// single punctuation marks.</summary>
    internal static List<string> Tokenize(string text)
    {
        List<string> tokens = [];
        foreach (Match m in TweetTokenRegex().Matches(text)) tokens.Add(m.Value);
        return tokens;
    }

    // en_G2p.__call__ per token.
    private static IReadOnlyList<string> WordPhones(string oWord, string pos)
    {
        string word = oWord.ToLowerInvariant();
        if (!HasAsciiLower(word)) return [word];
        if (word.Length == 1) return oWord == "A" ? ["EY1"] : Cmu.Value[word];
        if (Homographs.Value.TryGetValue(word, out Homograph h))
        {
            if (pos.StartsWith(h.Pos, StringComparison.Ordinal)) return h.First;
            if (pos.Length < h.Pos.Length && h.Pos.StartsWith(pos, StringComparison.Ordinal)) return h.First;
            return h.Second;
        }
        return QueryWord(oWord);
    }

    // en_G2p.qryword.
    private static IReadOnlyList<string> QueryWord(string oWord)
    {
        string word = oWord.ToLowerInvariant();
        if (word.Length > 1 && Cmu.Value.TryGetValue(word, out string[]? known)) return known;
        if (IsTitle(oWord) && Names.Value.TryGetValue(word, out string[]? name)) return name;
        if (word.Length <= 3)
        {
            List<string> letters = [];
            foreach (char w in word)
            {
                if (w == 'a') letters.Add("EY1");
                else if (!char.IsLetter(w)) letters.Add(w.ToString());
                else letters.AddRange(Cmu.Value[w.ToString()]);
            }
            return letters;
        }
        if (PossessiveRegex().IsMatch(word))
        {
            List<string> phones = [.. QueryWord(word[..^2])];
            string last = phones.Count > 0 ? phones[^1] : "";
            if (last is "P" or "T" or "K" or "F" or "TH" or "HH") phones.Add("S");
            else if (last is "S" or "Z" or "SH" or "ZH" or "CH" or "JH") phones.AddRange(["AH0", "Z"]);
            else phones.Add("Z");
            return phones;
        }
        List<string> parts = WordSegmenter.English.Segment(word);
        if (parts.Count == 1) return G2pEnLetterToSound.English.Predict(word);
        List<string> joined = [];
        foreach (string part in parts) joined.AddRange(QueryWord(part));
        return joined;
    }

    private static bool HasAsciiLower(string word)
    {
        foreach (char ch in word)
            if (ch is >= 'a' and <= 'z') return true;
        return false;
    }

    // Python str.istitle(): upper/title-case letters only after uncased characters, lower-case only after cased ones.
    private static bool IsTitle(string s)
    {
        bool cased = false, previousCased = false;
        foreach (char ch in s)
        {
            if (char.IsUpper(ch))
            {
                if (previousCased) return false;
                previousCased = cased = true;
            }
            else if (char.IsLower(ch))
            {
                if (!previousCased) return false;
                previousCased = cased = true;
            }
            else previousCased = false;
        }
        return cased;
    }

    private static Dictionary<string, string[]> LoadDict(string resource)
    {
        Dictionary<string, string[]> dict = new(StringComparer.Ordinal);
        foreach (string line in EmbeddedTextResource.ReadLines(resource))
        {
            int tab = line.IndexOf('\t', StringComparison.Ordinal);
            dict[line[..tab]] = line[(tab + 1)..].Split(' ');
        }
        return dict;
    }

    private static Dictionary<string, Homograph> LoadHomographs()
    {
        Dictionary<string, Homograph> dict = new(StringComparer.Ordinal);
        foreach (string line in EmbeddedTextResource.ReadLines("gptsovits_en_homographs.txt.gz"))
        {
            string[] f = line.Split('|');
            dict[f[0]] = new Homograph(f[1].Split(' '), f[2].Split(' '), f[3]);
        }
        return dict;
    }

    [GeneratedRegex(@"^([a-z]+)('s)$")]
    private static partial Regex PossessiveRegex();

    [GeneratedRegex(@"(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+){0,126}[.](?:[a-z]{2,13})\b/?(?!@)|[^\W\d_](?:[^\W\d_]|['\-_])+[^\W\d_]|[\w_]+|\.(?:\s*\.)+|\S",
        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex TweetTokenRegex();

    private readonly record struct Homograph(string[] First, string[] Second, string Pos);
}
