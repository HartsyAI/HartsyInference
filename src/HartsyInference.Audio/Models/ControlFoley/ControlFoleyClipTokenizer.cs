using System.Collections.Concurrent;
using System.Net;
using System.Text;
using System.Text.RegularExpressions;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>open_clip <c>SimpleTokenizer</c> as built by <c>get_tokenizer('ViT-H-14-378-quickgelu')</c>: ftfy-style text
/// cleaning, whitespace collapse, lowercase, byte-level BPE with <c>&lt;/w&gt;</c> words, <c>&lt;start_of_text&gt;</c> and
/// <c>&lt;end_of_text&gt;</c> markers, zero padding (not EOT padding) and truncation that forces the last id to EOT.</summary>
/// <remarks>The ftfy encoding-repair (mojibake) heuristics are not reproduced; every other default ftfy fix is.</remarks>
public sealed class ControlFoleyClipTokenizer
{
    /// <summary>Context length of every CLIP text tower.</summary>
    public const int ContextLength = 77;

    /// <summary>Start-of-text id.</summary>
    public const int StartOfTextId = 49406;

    /// <summary>End-of-text id.</summary>
    public const int EndOfTextId = 49407;

    private const int MergeCount = 49152 - 256 - 2;
    private const string StartMarker = "<start_of_text>";
    private const string EndMarker = "<end_of_text>";

    private static readonly Regex Pattern = new(
        StartMarker + "|" + EndMarker + @"|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
        RegexOptions.IgnoreCase | RegexOptions.Compiled | RegexOptions.CultureInvariant);

    private const string Cp1252C1 =
        "\u20AC\u0081\u201A\u0192\u201E\u2026\u2020\u2021\u02C6\u2030\u0160\u2039\u0152\u008D\u017D\u008F"
        + "\u0090\u2018\u2019\u201C\u201D\u2022\u2013\u2014\u02DC\u2122\u0161\u203A\u0153\u009D\u017E\u0178";

    private static readonly Dictionary<char, string> LatinLigatures = new()
    {
        ['\uFB00'] = "ff", ['\uFB01'] = "fi", ['\uFB02'] = "fl", ['\uFB03'] = "ffi", ['\uFB04'] = "ffl",
        ['\uFB05'] = "st", ['\uFB06'] = "st", ['\u0132'] = "IJ", ['\u0133'] = "ij", ['\u0149'] = "\u02BCn",
        ['\u01C7'] = "LJ", ['\u01C8'] = "Lj", ['\u01C9'] = "lj", ['\u01CA'] = "NJ", ['\u01CB'] = "Nj", ['\u01CC'] = "nj",
        ['\u01F1'] = "DZ", ['\u01F2'] = "Dz", ['\u01F3'] = "dz",
    };

    private readonly Dictionary<string, int> _encoder;
    private readonly Dictionary<(string, string), int> _ranks;
    private readonly ConcurrentDictionary<string, string[]> _cache = new();
    private readonly Dictionary<int, char> _byteToChar;

    /// <summary>Builds the tokenizer from the CLIP merges embedded in <c>HartsyInference.ModelAssets</c>.</summary>
    public ControlFoleyClipTokenizer()
    {
        _byteToChar = BuildByteToChar();
        // Ids follow open_clip's dict insertion order (printable bytes first), which Dictionary preserves without removals.
        List<string> vocab = [.. _byteToChar.Select(kv => kv.Value.ToString())];
        vocab.AddRange(vocab.Select(v => v + "</w>").ToList());
        _ranks = new Dictionary<(string, string), int>(MergeCount);
        using Stream stream = EmbeddedTokenizerResources.OpenClipMerges();
        using StreamReader reader = new(stream, Encoding.UTF8);
        reader.ReadLine();
        for (int i = 0; i < MergeCount; i++)
        {
            string line = reader.ReadLine() ?? throw new InvalidDataException("CLIP merges resource is truncated.");
            string[] parts = line.Split(' ');
            _ranks[(parts[0], parts[1])] = i;
            vocab.Add(parts[0] + parts[1]);
        }

        vocab.Add(StartMarker);
        vocab.Add(EndMarker);
        _encoder = new Dictionary<string, int>(vocab.Count);
        for (int i = 0; i < vocab.Count; i++)
        {
            _encoder[vocab[i]] = i;
        }
    }

    /// <summary>Tokenizes <paramref name="text"/> to exactly <see cref="ContextLength"/> ids.</summary>
    public int[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        List<int> ids = [StartOfTextId];
        foreach (Match match in Pattern.Matches(WhitespaceClean(BasicClean(text)).ToLowerInvariant()))
        {
            StringBuilder mapped = new();
            foreach (byte b in Encoding.UTF8.GetBytes(match.Value))
            {
                mapped.Append(_byteToChar[b]);
            }

            foreach (string piece in Bpe(mapped.ToString()))
            {
                ids.Add(_encoder[piece]);
            }
        }

        ids.Add(EndOfTextId);
        int[] result = new int[ContextLength];
        int count = Math.Min(ids.Count, ContextLength);
        for (int i = 0; i < count; i++)
        {
            result[i] = ids[i];
        }

        if (ids.Count > ContextLength)
        {
            result[ContextLength - 1] = EndOfTextId;
        }

        return result;
    }

    /// <summary>Tokenizes every prompt; row <c>i</c> is <c>Encode(prompts[i])</c>.</summary>
    public int[][] EncodeBatch(IReadOnlyList<string> prompts)
    {
        ArgumentNullException.ThrowIfNull(prompts);
        int[][] rows = new int[prompts.Count][];
        for (int i = 0; i < rows.Length; i++)
        {
            rows[i] = Encode(prompts[i]);
        }

        return rows;
    }

    private string[] Bpe(string token)
    {
        if (token == StartMarker || token == EndMarker)
        {
            return [token];
        }

        if (_cache.TryGetValue(token, out string[]? cached))
        {
            return cached;
        }

        List<string> word = [];
        for (int i = 0; i < token.Length - 1; i++)
        {
            word.Add(token[i].ToString());
        }

        word.Add(token[^1] + "</w>");
        while (word.Count > 1)
        {
            int best = int.MaxValue;
            (string, string) bigram = default;
            for (int i = 0; i < word.Count - 1; i++)
            {
                if (_ranks.TryGetValue((word[i], word[i + 1]), out int rank) && rank < best)
                {
                    best = rank;
                    bigram = (word[i], word[i + 1]);
                }
            }

            if (best == int.MaxValue)
            {
                break;
            }

            List<string> merged = [];
            int pos = 0;
            while (pos < word.Count)
            {
                if (pos < word.Count - 1 && word[pos] == bigram.Item1 && word[pos + 1] == bigram.Item2)
                {
                    merged.Add(bigram.Item1 + bigram.Item2);
                    pos += 2;
                }
                else
                {
                    merged.Add(word[pos]);
                    pos++;
                }
            }

            word = merged;
        }

        string[] result = [.. word];
        _cache[token] = result;
        return result;
    }

    private static string BasicClean(string text)
    {
        StringBuilder fixedText = new();
        int pos = 0;
        while (pos < text.Length)
        {
            int end = text.IndexOf('\n', pos) + 1;
            if (end == 0)
            {
                end = text.Length;
            }

            fixedText.Append(FtfyFixSegment(text.Substring(pos, end - pos)));
            pos = end;
        }

        string unescaped = WebUtility.HtmlDecode(WebUtility.HtmlDecode(fixedText.ToString()));
        return unescaped.Trim(PythonWhitespace);
    }

    private static string FtfyFixSegment(string segment)
    {
        if (!segment.Contains('<'))
        {
            segment = WebUtility.HtmlDecode(segment);
        }

        segment = FixC1Controls(segment);
        StringBuilder sb = new(segment.Length);
        for (int i = 0; i < segment.Length; i++)
        {
            char c = segment[i];
            if (c == '\r')
            {
                sb.Append('\n');
                if (i + 1 < segment.Length && segment[i + 1] == '\n')
                {
                    i++;
                }
            }
            else if (c is '\u2028' or '\u2029')
            {
                sb.Append('\n');
            }
            else if (LatinLigatures.TryGetValue(c, out string? replacement))
            {
                sb.Append(replacement);
            }
            else if (c is >= '\uFF01' and <= '\uFF5E')
            {
                sb.Append((char)(c - 0xFEE0));
            }
            else if (c == '\u3000')
            {
                sb.Append(' ');
            }
            else if (c is '\u2018' or '\u2019' or '\u201A' or '\u201B' or '\u2032' or '\u2035')
            {
                sb.Append('\'');
            }
            else if (c is '\u201C' or '\u201D' or '\u201E' or '\u201F' or '\u2033' or '\u2036')
            {
                sb.Append('"');
            }
            else if (!IsFtfyControl(c))
            {
                sb.Append(c);
            }
        }

        return sb.ToString().Normalize(NormalizationForm.FormC);
    }

    private static string FixC1Controls(string text)
    {
        if (!text.Any(c => c is >= '\u0080' and <= '\u009F'))
        {
            return text;
        }

        StringBuilder sb = new(text.Length);
        foreach (char c in text)
        {
            sb.Append(c is >= '\u0080' and <= '\u009F' ? Cp1252C1[c - 0x80] : c);
        }

        return sb.ToString();
    }

    private static bool IsFtfyControl(char c) =>
        c is <= '\u0008' or '\u000B' or (>= '\u000E' and <= '\u001F') or '\u007F' or (>= '\u206A' and <= '\u206F')
            or '\uFEFF' or (>= '\uFFF9' and <= '\uFFFB');

    private static readonly char[] PythonWhitespace = BuildPythonWhitespace();

    private static char[] BuildPythonWhitespace()
    {
        List<char> list = [];
        for (int c = 0; c < 0x10000; c++)
        {
            if (char.IsWhiteSpace((char)c) || c is >= 0x1C and <= 0x1F)
            {
                list.Add((char)c);
            }
        }

        return [.. list];
    }

    private static string WhitespaceClean(string text) =>
        string.Join(' ', text.Split(PythonWhitespace, StringSplitOptions.RemoveEmptyEntries));

    private static Dictionary<int, char> BuildByteToChar()
    {
        List<int> bs = [];
        bs.AddRange(Enumerable.Range('!', '~' - '!' + 1));
        bs.AddRange(Enumerable.Range(0xA1, 0xAC - 0xA1 + 1));
        bs.AddRange(Enumerable.Range(0xAE, 0xFF - 0xAE + 1));
        Dictionary<int, char> map = [];
        foreach (int b in bs)
        {
            map[b] = (char)b;
        }

        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!map.ContainsKey(b))
            {
                map[b] = (char)(256 + n);
                n++;
            }
        }

        return map;
    }
}
