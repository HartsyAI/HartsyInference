using System.Globalization;
using System.Text.Json;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>Ordered HF pre-tokenizer stages (<c>Split</c> with behavior/invert, then <c>ByteLevel</c>) that turn text into byte-level BPE pre-tokens.</summary>
public sealed class PreTokenizerPipeline
{
    private const string Gpt2Pattern = @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

    // One BMP stand-in per Unicode category so astral code points match category classes as one unit, like onig.
    private static readonly char[] ProxyByCategory = BuildProxyTable();
    private static readonly Lazy<SplitStage> Gpt2Stage = new(() => new SplitStage(Gpt2Pattern, SplitBehavior.Isolated, false));

    private readonly SplitStage[] _stages;
    private readonly bool _addPrefixSpace;
    private readonly bool _byteLevelGpt2Split;

    private PreTokenizerPipeline(SplitStage[] stages, bool addPrefixSpace, bool byteLevelGpt2Split)
    {
        _stages = stages;
        _addPrefixSpace = addPrefixSpace;
        _byteLevelGpt2Split = byteLevelGpt2Split;
    }

    /// <summary>The GPT-2 pre-tokenization used when a tokenizer declares none.</summary>
    public static PreTokenizerPipeline Gpt2 { get; } = new([], false, true);

    /// <summary>Builds the pipeline from a <c>pre_tokenizer</c> JSON object; null or missing yields <see cref="Gpt2"/>.</summary>
    public static PreTokenizerPipeline FromJson(JsonElement? preTokenizer)
    {
        if (preTokenizer is not JsonElement root || root.ValueKind != JsonValueKind.Object) return Gpt2;
        List<SplitStage> stages = [];
        bool addPrefixSpace = false;
        bool gpt2Split = false;
        Collect(root, stages, ref addPrefixSpace, ref gpt2Split);
        return new PreTokenizerPipeline([.. stages], addPrefixSpace, gpt2Split);
    }

    /// <summary>Splits <paramref name="text"/> into pre-tokens, in order, covering everything the stages keep.</summary>
    public List<string> Split(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        List<string> result = [];
        if (text.Length == 0) return result;

        string work = ToProxy(text, out int[]? map);
        List<(int Start, int End)> pieces = [(0, work.Length)];
        List<(int Start, int End)> next = [];
        foreach (SplitStage stage in _stages)
        {
            stage.Apply(work, pieces, next);
            (pieces, next) = (next, pieces);
            next.Clear();
        }

        foreach ((int start, int end) in pieces)
        {
            int origStart = map is null ? start : map[start];
            int origEnd = map is null ? end : map[end];
            string piece = text.Substring(origStart, origEnd - origStart);
            if (_addPrefixSpace && piece.Length > 0 && piece[0] != ' ') piece = " " + piece;
            if (_byteLevelGpt2Split) result.AddRange(Gpt2Split(piece));
            else result.Add(piece);
        }
        return result;
    }

    private static List<string> Gpt2Split(string piece)
    {
        SplitStage stage = Gpt2Stage.Value;
        List<(int Start, int End)> input = [(0, piece.Length)];
        List<(int Start, int End)> output = [];
        stage.Apply(piece, input, output);
        List<string> parts = new(output.Count);
        foreach ((int s, int e) in output) parts.Add(piece.Substring(s, e - s));
        return parts;
    }

    private static void Collect(JsonElement el, List<SplitStage> stages, ref bool addPrefixSpace, ref bool gpt2Split)
    {
        string type = el.TryGetProperty("type", out JsonElement t) ? t.GetString() ?? string.Empty : string.Empty;
        switch (type)
        {
            case "Sequence":
                foreach (JsonElement child in el.GetProperty("pretokenizers").EnumerateArray())
                    Collect(child, stages, ref addPrefixSpace, ref gpt2Split);
                break;
            case "Split":
                stages.Add(ReadSplit(el));
                break;
            case "ByteLevel":
                addPrefixSpace |= el.TryGetProperty("add_prefix_space", out JsonElement a) && a.ValueKind == JsonValueKind.True;
                // HF defaults use_regex to true when absent.
                bool useRegex = !el.TryGetProperty("use_regex", out JsonElement u) || u.ValueKind != JsonValueKind.False;
                gpt2Split |= useRegex;
                break;
            default:
                throw new NotSupportedException($"Pre-tokenizer stage type '{type}' is not supported.");
        }
    }

    private static SplitStage ReadSplit(JsonElement el)
    {
        JsonElement pattern = el.GetProperty("pattern");
        if (!pattern.TryGetProperty("Regex", out JsonElement rx) || rx.GetString() is not string regex)
            throw new NotSupportedException("Split pre-tokenizer patterns other than Regex are not supported.");
        string behaviorName = el.GetProperty("behavior").GetString() ?? string.Empty;
        SplitBehavior behavior = behaviorName switch
        {
            "Removed" => SplitBehavior.Removed,
            "Isolated" => SplitBehavior.Isolated,
            "MergedWithPrevious" => SplitBehavior.MergedWithPrevious,
            "MergedWithNext" => SplitBehavior.MergedWithNext,
            "Contiguous" => SplitBehavior.Contiguous,
            _ => throw new NotSupportedException($"Split behavior '{behaviorName}' is not supported."),
        };
        bool invert = el.TryGetProperty("invert", out JsonElement inv) && inv.ValueKind == JsonValueKind.True;
        return new SplitStage(regex, behavior, invert);
    }

    // Returns the text with each surrogate pair collapsed to a same-category BMP proxy; map[i] is the original offset of proxy index i.
    private static string ToProxy(string text, out int[]? map)
    {
        bool hasPair = false;
        for (int i = 0; i + 1 < text.Length; i++)
        {
            if (char.IsHighSurrogate(text[i]) && char.IsLowSurrogate(text[i + 1])) { hasPair = true; break; }
        }
        if (!hasPair) { map = null; return text; }

        char[] chars = new char[text.Length];
        int[] offsets = new int[text.Length + 1];
        int n = 0;
        for (int i = 0; i < text.Length; i++)
        {
            offsets[n] = i;
            if (i + 1 < text.Length && char.IsHighSurrogate(text[i]) && char.IsLowSurrogate(text[i + 1]))
            {
                int cp = char.ConvertToUtf32(text[i], text[i + 1]);
                chars[n++] = ProxyByCategory[(int)CharUnicodeInfo.GetUnicodeCategory(cp)];
                i++;
            }
            else chars[n++] = text[i];
        }
        offsets[n] = text.Length;
        map = new int[n + 1];
        Array.Copy(offsets, map, n + 1);
        return new string(chars, 0, n);
    }

    private static char[] BuildProxyTable()
    {
        char[] table = new char[Enum.GetValues<UnicodeCategory>().Length];
        // A pattern naming an explicit non-ASCII range above U+00FF (beyond these CJK ones) could match a proxy by accident.
        for (int c = 0x0100; c < 0xD800; c++)
        {
            if (c is >= 0x3040 and <= 0x30FF or >= 0x4E00 and <= 0x9FFF or >= 0x3400 and <= 0x4DBF) continue;
            char ch = (char)c;
            if (char.IsWhiteSpace(ch)) continue;
            int cat = (int)CharUnicodeInfo.GetUnicodeCategory(ch);
            if (table[cat] == 0) table[cat] = ch;
        }
        for (int c = 0xE000; c <= 0xFFFF; c++)
        {
            char ch = (char)c;
            if (char.IsWhiteSpace(ch)) continue;
            int cat = (int)CharUnicodeInfo.GetUnicodeCategory(ch);
            if (table[cat] == 0) table[cat] = ch;
        }
        // Categories with no BMP representative fall back to a private-use char (category Co).
        for (int i = 0; i < table.Length; i++) if (table[i] == 0) table[i] = '';
        return table;
    }
}
