using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Frontends;

/// <summary>Port of jieba 0.42.1's precise mode, <c>jieba.lcut(text, cut_all=False, HMM=True)</c> (fxsjy/jieba, MIT):
/// the prefix dictionary from <c>dict.txt</c>, the word DAG and its maximum-probability route, and the
/// <c>finalseg</c> BMES hidden Markov model whose Viterbi path groups the single characters the route leaves
/// unmatched. Blocks outside jieba's word characters split on whitespace and then fall apart character by character,
/// as jieba's do. Probabilities are summed in the order jieba sums them so ties break the same way.</summary>
internal sealed partial class JiebaSegmenter
{
    private const double MinFloat = -3.14e100;
    private const int B = 0, E = 1, M = 2, S = 3;
    // finalseg/prob_start.py and prob_trans.py, states in jieba's comparison order B < E < M < S.
    private static readonly double[] StartP = [-0.26268660809250016, MinFloat, MinFloat, -1.4652633398537678];
    private static readonly double[,] TransP =
    {
        { MinFloat, -0.510825623765990, -0.916290731874155, MinFloat },
        { -0.5897149736854513, MinFloat, MinFloat, -0.8085250474669937 },
        { MinFloat, -0.33344856811948514, -1.2603623820268226, MinFloat },
        { -0.7211965654669841, MinFloat, MinFloat, -0.6658631448798212 },
    };
    // finalseg PrevStatus: the states each state may follow.
    private static readonly int[][] PrevStatus = [[E, S], [B, M], [M, B], [S, E]];

    private readonly Dictionary<string, int> _freq;
    private readonly double _logTotal;
    private readonly Dictionary<char, double>[] _emit;

    private JiebaSegmenter(Dictionary<string, int> freq, long total, Dictionary<char, double>[] emit)
    {
        _freq = freq;
        _logTotal = Math.Log(total);
        _emit = emit;
    }

    /// <summary>Loads jieba's <c>dict.txt</c> (<c>word freq [tag]</c> per line) and <c>finalseg/prob_emit.py</c>.</summary>
    public static JiebaSegmenter FromStreams(Stream dictionary, Stream probEmit)
    {
        ArgumentNullException.ThrowIfNull(dictionary);
        ArgumentNullException.ThrowIfNull(probEmit);
        Dictionary<string, int> freq = new(700_000, StringComparer.Ordinal);
        long total = 0;
        using (StreamReader reader = new(dictionary, new UTF8Encoding(false), detectEncodingFromByteOrderMarks: false))
        {
            int lineNo = 0;
            while (reader.ReadLine() is { } line)
            {
                lineNo++;
                string[] fields = line.Trim().Split(' ');
                if (fields.Length < 2 || !int.TryParse(fields[1], NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture,
                    out int count))
                    throw new InvalidDataException($"invalid jieba dictionary entry at line {lineNo}: {line}");
                string word = fields[0];
                freq[word] = count;
                total += count;
                for (int i = 1; i < word.Length; i++) freq.TryAdd(word[..i], 0);
            }
        }
        return new JiebaSegmenter(freq, total, ParseEmit(probEmit));
    }

    /// <summary>Loads both files from disk.</summary>
    public static JiebaSegmenter FromFiles(string dictionaryPath, string probEmitPath)
    {
        using FileStream dictionary = File.OpenRead(dictionaryPath);
        using FileStream emit = File.OpenRead(probEmitPath);
        return FromStreams(dictionary, emit);
    }

    /// <summary><c>jieba.lcut(sentence)</c>: the words of <paramref name="sentence"/>, in order.</summary>
    public List<string> Cut(string sentence)
    {
        ArgumentNullException.ThrowIfNull(sentence);
        List<string> words = [];
        int i = 0;
        while (i < sentence.Length)
        {
            int start = i;
            bool han = IsHanDefault(sentence[i]);
            while (i < sentence.Length && IsHanDefault(sentence[i]) == han) i++;
            string block = sentence[start..i];
            if (han)
            {
                CutDag(block, words);
                continue;
            }
            // re_skip_default "(\r\n|\s)": each whitespace (or CRLF) is a word; anything else falls apart by character.
            int j = 0;
            while (j < block.Length)
            {
                if (block[j] == '\r' && j + 1 < block.Length && block[j + 1] == '\n')
                {
                    words.Add("\r\n");
                    j += 2;
                }
                else if (IsPythonSpace(block[j]))
                {
                    words.Add(block[j].ToString());
                    j++;
                }
                else
                {
                    int len = char.IsHighSurrogate(block[j]) && j + 1 < block.Length && char.IsLowSurrogate(block[j + 1])
                        ? 2 : 1;
                    words.Add(block.Substring(j, len));
                    j += len;
                }
            }
        }
        return words;
    }

    // Tokenizer.__cut_DAG: the max-probability route, with runs of single characters handed to the HMM unless the
    // run is itself a dictionary word.
    private void CutDag(string sentence, List<string> words)
    {
        int n = sentence.Length;
        int[] route = Route(sentence);
        StringBuilder buf = new();
        int x = 0;
        while (x < n)
        {
            int y = route[x] + 1;
            if (y - x == 1)
            {
                buf.Append(sentence[x]);
            }
            else
            {
                FlushBuffer(buf, words);
                words.Add(sentence[x..y]);
            }
            x = y;
        }
        FlushBuffer(buf, words);
    }

    private void FlushBuffer(StringBuilder buf, List<string> words)
    {
        if (buf.Length == 0) return;
        string run = buf.ToString();
        buf.Clear();
        if (run.Length == 1)
        {
            words.Add(run);
        }
        else if (_freq.GetValueOrDefault(run) == 0)
        {
            FinalSegCut(run, words);
        }
        else
        {
            foreach (char c in run) words.Add(c.ToString());
        }
    }

    // Tokenizer.get_DAG + calc: for each start, the end that maximizes log(freq) - log(total) + best(rest), the later
    // end winning a tie as Python's tuple max picks it.
    private int[] Route(string sentence)
    {
        int n = sentence.Length;
        List<int>[] dag = new List<int>[n];
        for (int k = 0; k < n; k++)
        {
            List<int> ends = [];
            int i = k;
            while (i < n && _freq.TryGetValue(sentence[k..(i + 1)], out int f))
            {
                if (f != 0) ends.Add(i);
                i++;
            }
            if (ends.Count == 0) ends.Add(k);
            dag[k] = ends;
        }
        double[] score = new double[n + 1];
        int[] route = new int[n + 1];
        for (int idx = n - 1; idx >= 0; idx--)
        {
            double best = 0;
            int bestEnd = -1;
            foreach (int x in dag[idx])
            {
                int f = _freq.GetValueOrDefault(sentence[idx..(x + 1)]);
                double v = Math.Log(f == 0 ? 1 : f) - _logTotal + score[x + 1];
                if (bestEnd < 0 || v > best || (v == best && x > bestEnd))
                {
                    best = v;
                    bestEnd = x;
                }
            }
            score[idx] = best;
            route[idx] = bestEnd;
        }
        return route;
    }

    // finalseg.cut: Han runs go through the Viterbi; anything else splits around letter/digit/decimal/percent runs.
    private void FinalSegCut(string sentence, List<string> words)
    {
        int i = 0;
        while (i < sentence.Length)
        {
            int start = i;
            bool han = IsHanFinalSeg(sentence[i]);
            while (i < sentence.Length && IsHanFinalSeg(sentence[i]) == han) i++;
            string block = sentence[start..i];
            if (han)
            {
                Viterbi(block, words);
                continue;
            }
            foreach (string piece in SkipFinalSegRegex().Split(block))
            {
                if (piece.Length != 0) words.Add(piece);
            }
        }
    }

    private void Viterbi(string obs, List<string> words)
    {
        int n = obs.Length;
        double[] v = new double[4], next = new double[4];
        int[,] back = new int[n, 4];
        for (int y = 0; y < 4; y++) v[y] = StartP[y] + Emit(y, obs[0]);
        for (int t = 1; t < n; t++)
        {
            for (int y = 0; y < 4; y++)
            {
                double em = Emit(y, obs[t]);
                double best = 0;
                int from = -1;
                foreach (int y0 in PrevStatus[y])
                {
                    double p = v[y0] + TransP[y0, y] + em;
                    if (from < 0 || p > best || (p == best && y0 > from))
                    {
                        best = p;
                        from = y0;
                    }
                }
                next[y] = best;
                back[t, y] = from;
            }
            (v, next) = (next, v);
        }
        int state = v[S] >= v[E] ? S : E;
        int[] states = new int[n];
        for (int t = n - 1; t >= 0; t--)
        {
            states[t] = state;
            if (t > 0) state = back[t, state];
        }
        int begin = 0, nexti = 0;
        for (int i = 0; i < n; i++)
        {
            switch (states[i])
            {
                case B:
                    begin = i;
                    break;
                case E:
                    words.Add(obs[begin..(i + 1)]);
                    nexti = i + 1;
                    break;
                case S:
                    words.Add(obs[i].ToString());
                    nexti = i + 1;
                    break;
            }
        }
        if (nexti < n) words.Add(obs[nexti..]);
    }

    private double Emit(int state, char c) => _emit[state].TryGetValue(c, out double p) ? p : MinFloat;

    // jieba re_han_default: [\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-].
    private static bool IsHanDefault(char c) => (c >= '\u4e00' && c <= '\u9fd5') || char.IsAsciiLetterOrDigit(c)
        || c is '+' or '#' or '&' or '.' or '_' or '%' or '-';

    private static bool IsHanFinalSeg(char c) => c >= '\u4e00' && c <= '\u9fd5';

    // Python str.isspace, which also counts the ASCII separators U+001C..U+001F.
    internal static bool IsPythonSpace(char c) => char.IsWhiteSpace(c) || (c >= '\u001c' && c <= '\u001f');

    // prob_emit.py is a Python dict literal, P={'B': {'\u4e00': -3.65, ...}, 'E': {...}, ...}; keys may also be literal.
    private static Dictionary<char, double>[] ParseEmit(Stream stream)
    {
        Dictionary<char, double>[] emit = [new(), new(), new(), new()];
        int state = -1;
        using StreamReader reader = new(stream, Encoding.UTF8);
        while (reader.ReadLine() is { } line)
        {
            foreach (Match m in EmitEntryRegex().Matches(line))
            {
                if (m.Groups[1].Success)
                {
                    state = "BEMS".IndexOf(m.Groups[1].Value[0], StringComparison.Ordinal);
                    continue;
                }
                if (state < 0) throw new InvalidDataException("prob_emit.py: emission entry before any state");
                string key = m.Groups[2].Value;
                char c = key.Length == 1 ? key[0]
                    : (char)int.Parse(key.AsSpan(2), NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                emit[state][c] = double.Parse(m.Groups[3].Value, NumberStyles.Float, CultureInfo.InvariantCulture);
            }
        }
        if (emit.Any(d => d.Count == 0)) throw new InvalidDataException("prob_emit.py: a BMES state has no emissions");
        return emit;
    }

    [GeneratedRegex(@"'([BEMS])':\s*\{|'(\\u[0-9a-fA-F]{4}|[^'\\])':\s*(-?[0-9.]+(?:[eE][-+]?[0-9]+)?)")]
    private static partial Regex EmitEntryRegex();

    [GeneratedRegex(@"([a-zA-Z0-9]+(?:\.\d+)?%?)")]
    private static partial Regex SkipFinalSegRegex();
}
