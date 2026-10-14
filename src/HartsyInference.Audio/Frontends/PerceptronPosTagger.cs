using System.Globalization;

namespace HartsyInference.Audio.Frontends;

/// <summary>NLTK's averaged-perceptron part-of-speech tagger (<c>nltk.pos_tag</c>, English, Penn Treebank tags):
/// the tag dictionary shortcut, then the greedy left-to-right perceptron over NLTK's feature templates, with the
/// pretrained <c>averaged_perceptron_tagger_eng</c> weights. g2p_en-style frontends use it to pick homograph
/// readings ("record" as noun or verb).</summary>
internal sealed class PerceptronPosTagger
{
    private static readonly Lazy<PerceptronPosTagger> Shared = new(Load);

    private readonly string[] _classes;
    private readonly Dictionary<string, (int Class, double Weight)[]> _weights;
    private readonly Dictionary<string, string> _tagDict;

    private PerceptronPosTagger(string[] classes, Dictionary<string, (int, double)[]> weights,
        Dictionary<string, string> tagDict)
    {
        _classes = classes;
        _weights = weights;
        _tagDict = tagDict;
    }

    /// <summary>The tagger over the embedded English model (loaded on first use).</summary>
    public static PerceptronPosTagger English => Shared.Value;

    /// <summary>Tags <paramref name="tokens"/>, one Penn Treebank tag per token.</summary>
    public string[] Tag(IReadOnlyList<string> tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        string[] context = new string[tokens.Count + 4];
        context[0] = "-START-";
        context[1] = "-START2-";
        for (int i = 0; i < tokens.Count; i++) context[i + 2] = NormalizeWord(tokens[i]);
        context[^2] = "-END-";
        context[^1] = "-END2-";

        string[] tags = new string[tokens.Count];
        string prev = "-START-", prev2 = "-START2-";
        double[] scores = new double[_classes.Length];
        for (int i = 0; i < tokens.Count; i++)
        {
            string word = tokens[i];
            if (!_tagDict.TryGetValue(word, out string? tag) || tag.Length == 0) tag = Predict(word, i + 2, context, prev, prev2, scores);
            tags[i] = tag;
            prev2 = prev;
            prev = tag;
        }
        return tags;
    }

    private string Predict(string word, int i, string[] context, string prev, string prev2, double[] scores)
    {
        Array.Clear(scores);
        // Feature order matters: scores are float sums and NLTK adds them in this order.
        Accumulate("bias", scores);
        Accumulate("i suffix " + Suffix(word), scores);
        Accumulate("i pref1 " + (word.Length > 0 ? word[..1] : ""), scores);
        Accumulate("i-1 tag " + prev, scores);
        Accumulate("i-2 tag " + prev2, scores);
        Accumulate("i tag+i-2 tag " + prev + " " + prev2, scores);
        Accumulate("i word " + context[i], scores);
        Accumulate("i-1 tag+i word " + prev + " " + context[i], scores);
        Accumulate("i-1 word " + context[i - 1], scores);
        Accumulate("i-1 suffix " + Suffix(context[i - 1]), scores);
        Accumulate("i-2 word " + context[i - 2], scores);
        Accumulate("i+1 word " + context[i + 1], scores);
        Accumulate("i+1 suffix " + Suffix(context[i + 1]), scores);
        Accumulate("i+2 word " + context[i + 2], scores);

        // max(classes, key=(score, label)): ties go to the larger label.
        int best = 0;
        for (int c = 1; c < _classes.Length; c++)
        {
            if (scores[c] > scores[best]
                || (scores[c] == scores[best] && string.CompareOrdinal(_classes[c], _classes[best]) > 0)) best = c;
        }
        return _classes[best];
    }

    private void Accumulate(string feature, double[] scores)
    {
        if (!_weights.TryGetValue(feature, out (int Class, double Weight)[]? row)) return;
        foreach ((int c, double w) in row) scores[c] += w;
    }

    private static string Suffix(string word) => word.Length > 3 ? word[^3..] : word;

    private static string NormalizeWord(string word)
    {
        if (word.Contains('-', StringComparison.Ordinal) && word[0] != '-') return "!HYPHEN";
        if (word.Length == 4 && IsAllDigits(word)) return "!YEAR";
        if (word.Length > 0 && char.IsDigit(word[0])) return "!DIGITS";
        return word.ToLowerInvariant();
    }

    private static bool IsAllDigits(string word)
    {
        foreach (char ch in word)
            if (!char.IsDigit(ch)) return false;
        return word.Length > 0;
    }

    private static PerceptronPosTagger Load()
    {
        List<string> classList = [.. EmbeddedTextResource.ReadLines("nltk_perceptron_classes.txt.gz")];
        classList.Sort(StringComparer.Ordinal);
        string[] classes = [.. classList];
        Dictionary<string, int> classIndex = new(StringComparer.Ordinal);
        for (int i = 0; i < classes.Length; i++) classIndex[classes[i]] = i;

        Dictionary<string, List<(int, double)>> rows = new(StringComparer.Ordinal);
        foreach (string line in EmbeddedTextResource.ReadLines("nltk_perceptron_weights.txt.gz"))
        {
            int t2 = line.LastIndexOf('\t');
            int t1 = line.LastIndexOf('\t', t2 - 1);
            string feature = line[..t1];
            int c = classIndex[line[(t1 + 1)..t2]];
            double w = double.Parse(line.AsSpan(t2 + 1), NumberStyles.Float, CultureInfo.InvariantCulture);
            if (!rows.TryGetValue(feature, out List<(int, double)>? list)) rows[feature] = list = [];
            list.Add((c, w));
        }
        Dictionary<string, (int, double)[]> weights = new(rows.Count, StringComparer.Ordinal);
        foreach (KeyValuePair<string, List<(int, double)>> kv in rows) weights[kv.Key] = [.. kv.Value];

        Dictionary<string, string> tagDict = new(StringComparer.Ordinal);
        foreach (string line in EmbeddedTextResource.ReadLines("nltk_perceptron_tagdict.txt.gz"))
        {
            int tab = line.LastIndexOf('\t');
            tagDict[line[..tab]] = line[(tab + 1)..];
        }
        return new PerceptronPosTagger(classes, weights, tagDict);
    }
}
