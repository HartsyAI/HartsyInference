namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>Loads a tiktoken rank file (<c>tiktoken.model</c>: a base64 token and its rank per line) as the engine's byte-level BPE, with the pre-tokenizer
/// regex and special tokens its Python <c>tiktoken.Encoding</c> is built from. Built on <see cref="TiktokenConverter"/> and <see cref="HfTokenizerJson"/>,
/// so any family that ships a tiktoken file loads through here rather than keeping its own copy.</summary>
public static class TiktokenFileTokenizer
{
    /// <summary>Loads <paramref name="tiktokenPath"/> with <paramref name="pattern"/> as its pre-tokenizer and <paramref name="specials"/> above its ranks.</summary>
    /// <param name="pattern">The pre-tokenizer regex, in .NET syntax (tiktoken's <c>pat_str</c>).</param>
    /// <param name="specials">Special tokens with their ids. Ids and literals must be unique and must not fall inside the rank range.</param>
    /// <param name="bosId">The begin-of-sequence id, which must name one of <paramref name="specials"/>, or null.</param>
    /// <param name="eosId">The end-of-sequence id, which must name one of <paramref name="specials"/>, or null.</param>
    /// <param name="stopIds">Extra ids that end generation.</param>
    /// <exception cref="FileNotFoundException">The rank file does not exist.</exception>
    /// <exception cref="InvalidDataException">The ranks are not dense from zero, or a special id is duplicated or not given a literal.</exception>
    public static GgufTokenizer Load(string tiktokenPath, string pattern, IReadOnlyList<(string Token, int Id)> specials,
        int? bosId = null, int? eosId = null, IReadOnlyList<int>? stopIds = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(tiktokenPath);
        ArgumentException.ThrowIfNullOrEmpty(pattern);
        ArgumentNullException.ThrowIfNull(specials);
        if (!File.Exists(tiktokenPath))
            throw new FileNotFoundException("tiktoken rank file not found.", tiktokenPath);
        ValidateSpecials(specials);
        string? bos = LiteralOf(specials, bosId, nameof(bosId));
        string? eos = LiteralOf(specials, eosId, nameof(eosId));

        byte[] json = TiktokenConverter.ToHuggingFaceJson(tiktokenPath, pattern, normalizer: null, addedTokens: specials);
        using MemoryStream stream = new(json);
        return HfTokenizerJson.LoadByteLevelBpe(stream, stopIds, bos, eos);
    }

    /// <summary>Counts the rank lines of a tiktoken file, which is the first special id a tiktoken encoding assigns.</summary>
    public static int CountRanks(string tiktokenPath)
    {
        ArgumentException.ThrowIfNullOrEmpty(tiktokenPath);
        int count = 0;
        foreach (string line in File.ReadLines(tiktokenPath))
            if (!string.IsNullOrWhiteSpace(line))
                count++;
        return count;
    }

    /// <summary>Returns the specials of a tiktoken encoding's reserved range: the id names in <paramref name="named"/>, and every other id in
    /// <c>[firstId, firstId + count)</c> as <c>&lt;|reserved_token_N|&gt;</c>, in ascending id order. Ids outside the range are not included.</summary>
    public static IReadOnlyList<(string Token, int Id)> WithReservedRange(int firstId, int count, IReadOnlyDictionary<int, string> named)
    {
        ArgumentNullException.ThrowIfNull(named);
        if (firstId < 0 || count < 0)
            throw new ArgumentOutOfRangeException(nameof(firstId), "The reserved range must start at a non-negative id and have a non-negative length.");
        List<(string Token, int Id)> result = new(count);
        for (int id = firstId; id < firstId + count; id++)
            result.Add((named.TryGetValue(id, out string? name) ? name : $"<|reserved_token_{id}|>", id));
        return result;
    }

    private static void ValidateSpecials(IReadOnlyList<(string Token, int Id)> specials)
    {
        HashSet<int> ids = [];
        HashSet<string> literals = new(StringComparer.Ordinal);
        foreach ((string token, int id) in specials)
        {
            if (string.IsNullOrEmpty(token))
                throw new InvalidDataException($"Special token id {id} has no literal.");
            if (!ids.Add(id))
                throw new InvalidDataException($"Special token id {id} is given twice.");
            if (!literals.Add(token))
                throw new InvalidDataException($"Special token literal '{token}' is given twice.");
        }
    }

    private static string? LiteralOf(IReadOnlyList<(string Token, int Id)> specials, int? id, string parameter)
    {
        if (id is not { } value)
            return null;
        foreach ((string token, int specialId) in specials)
            if (specialId == value)
                return token;
        throw new InvalidDataException($"{parameter} {value} names no special token of the tiktoken encoding.");
    }
}
