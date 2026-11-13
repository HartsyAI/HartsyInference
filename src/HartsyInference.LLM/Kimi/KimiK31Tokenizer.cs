using System.Globalization;
using System.Text.Json;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Kimi;

/// <summary>The Kimi K3.1 text tokenizer: the checkpoint's <c>tiktoken.model</c> with the pre-tokenizer and special-token table of its
/// <c>tokenization_kimi.py</c> (<c>TikTokenTokenizer</c>). Used when the directory has no <c>tokenizer.json</c>.</summary>
public static class KimiK31Tokenizer
{
    /// <summary>The rank file the Moonshot repo ships.</summary>
    public const string TiktokenFile = "tiktoken.model";

    /// <summary>Reserved special ids above the ranks, as <c>num_reserved_special_tokens</c> in <c>tokenization_kimi.py</c>.</summary>
    public const int ReservedSpecialCount = 256;

    // tokenization_kimi.py's pat_str, rewritten for .NET, which has no \p{Han} script class and no && set intersection. Han is its BMP blocks plus the
    // astral ideograph planes as surrogate pairs; "letter minus Han" is a negative lookahead. The astral ranges cover planes 2 and 3 as a whole, which
    // holds no non-Han letters, so the approximation only differs on unassigned code points.
    private const string Han = @"(?:[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〻㐀-䶿一-鿿豈-龎]"
        + @"|[\uD840-\uD88C][\uDC00-\uDFFF])";
    private const string UpperLetter = @"(?:(?!" + Han + @")[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}])";
    private const string LowerLetter = @"(?:(?!" + Han + @")[\p{Ll}\p{Lm}\p{Lo}\p{M}])";
    private const string Contraction = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)?";

    /// <summary>The pre-tokenizer regex: Han runs, cased words, digit runs of at most three, punctuation runs, and whitespace.</summary>
    public static readonly string PreTokenPattern = string.Join("|",
        Han + "+",
        @"[^\r\n\p{L}\p{N}]?" + UpperLetter + "*" + LowerLetter + "+" + Contraction,
        @"[^\r\n\p{L}\p{N}]?" + UpperLetter + "+" + LowerLetter + "*" + Contraction,
        @"\p{N}{1,3}",
        @" ?[^\s\p{L}\p{N}]+[\r\n]*",
        @"\s*[\r\n]+",
        @"\s+(?!\S)",
        @"\s+");

    /// <summary>Loads the directory's tokenizer. The ranks come from <c>tiktoken.model</c>; the named specials come from <c>tokenizer_config.json</c>, and
    /// every other special id is <c>&lt;|reserved_token_N|&gt;</c>, as the Python encoding names them.</summary>
    /// <param name="directory">A checkpoint directory that holds <c>tiktoken.model</c> and, for its special names, <c>tokenizer_config.json</c>.</param>
    /// <param name="stopIds">Ids that end generation.</param>
    /// <param name="bosId">The begin-of-sequence id from <c>config.json</c>, or null.</param>
    /// <param name="eosId">The end-of-sequence id from <c>config.json</c>, or null. It is the config's id, not <c>tokenizer_config.json</c>'s
    /// <c>eos_token</c>, because the Moonshot config and generation config both name <c>&lt;|end_of_msg|&gt;</c>.</param>
    /// <exception cref="InvalidDataException">The tiktoken file or the special-token table is malformed, or a given id names no special token.</exception>
    public static ILlmTokenizer Load(string directory, IReadOnlyList<int> stopIds, int? bosId, int? eosId)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        ArgumentNullException.ThrowIfNull(stopIds);
        string tiktokenPath = Path.Combine(directory, TiktokenFile);
        int baseCount = TiktokenFileTokenizer.CountRanks(tiktokenPath);
        IReadOnlyList<(string Token, int Id)> specials = TiktokenFileTokenizer.WithReservedRange(
            baseCount, ReservedSpecialCount, ReadAddedTokens(Path.Combine(directory, "tokenizer_config.json")));
        return TiktokenFileTokenizer.Load(tiktokenPath, PreTokenPattern, specials, bosId, eosId, stopIds);
    }

    /// <summary>The <c>added_tokens_decoder</c> of <c>tokenizer_config.json</c> as id to literal, empty when the file is absent.</summary>
    private static Dictionary<int, string> ReadAddedTokens(string path)
    {
        Dictionary<int, string> named = [];
        if (!File.Exists(path))
            return named;
        try
        {
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
            if (!document.RootElement.TryGetProperty("added_tokens_decoder", out JsonElement decoder))
                return named;
            foreach (JsonProperty entry in decoder.EnumerateObject())
            {
                if (!int.TryParse(entry.Name, NumberStyles.None, CultureInfo.InvariantCulture, out int id))
                    throw new InvalidDataException($"'{path}' names a non-integer added token id '{entry.Name}'.");
                if (!entry.Value.TryGetProperty("content", out JsonElement content) || content.ValueKind != JsonValueKind.String)
                    throw new InvalidDataException($"'{path}' gives added token {id} no string content.");
                named[id] = content.GetString()!;
            }
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"'{path}' is not valid JSON: {ex.Message}", ex);
        }
        return named;
    }
}
