using System.Text;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>The ByT5 tokenizer (HF <c>ByT5Tokenizer</c>): every UTF-8 byte becomes id <c>byte + 3</c> after the three
/// specials (<c>&lt;pad&gt;</c> 0, <c>&lt;/s&gt;</c> 1, <c>&lt;unk&gt;</c> 2), the sentinels <c>&lt;extra_id_N&gt;</c>
/// (N 0–124) are ids 259 + N, and <see cref="Encode"/> appends <c>&lt;/s&gt;</c>. Special and sentinel tokens written in
/// the text are matched as whole tokens before the bytes, as the HF added-token pass does. There is no vocabulary file:
/// the mapping is the byte table itself.</summary>
public static class ByT5Tokenizer
{
    /// <summary>The padding id.</summary>
    public const int PadId = 0;

    /// <summary>The end-of-sequence id, appended by <see cref="Encode"/>.</summary>
    public const int EosId = 1;

    /// <summary>The unknown-token id.</summary>
    public const int UnkId = 2;

    /// <summary>How many ids precede the byte table.</summary>
    public const int ByteOffset = 3;

    /// <summary>The first sentinel id (<c>&lt;extra_id_0&gt;</c>).</summary>
    public const int FirstSentinelId = 259;

    /// <summary>How many <c>&lt;extra_id_N&gt;</c> sentinels exist.</summary>
    public const int SentinelCount = 125;

    /// <summary>Token ids for <paramref name="text"/>, ending with <see cref="EosId"/>.</summary>
    public static int[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        List<int> ids = new List<int>(Encoding.UTF8.GetMaxByteCount(text.Length) + 1);
        int start = 0;
        for (int i = 0; i < text.Length; i++)
        {
            if (text[i] != '<' || MatchSpecial(text, i) is not (int id, int length))
                continue;
            AppendBytes(ids, text.AsSpan(start, i - start));
            ids.Add(id);
            i += length - 1;
            start = i + 1;
        }
        AppendBytes(ids, text.AsSpan(start));
        ids.Add(EosId);
        return ids.ToArray();
    }

    private static void AppendBytes(List<int> ids, ReadOnlySpan<char> chars)
    {
        if (chars.IsEmpty)
            return;
        byte[] bytes = Encoding.UTF8.GetBytes(chars.ToArray());
        foreach (byte b in bytes)
            ids.Add(b + ByteOffset);
    }

    /// <summary>The special or sentinel token starting at <paramref name="at"/>, if any.</summary>
    private static (int Id, int Length)? MatchSpecial(string text, int at)
    {
        ReadOnlySpan<char> rest = text.AsSpan(at);
        if (rest.StartsWith("<pad>", StringComparison.Ordinal))
            return (PadId, 5);
        if (rest.StartsWith("</s>", StringComparison.Ordinal))
            return (EosId, 4);
        if (rest.StartsWith("<unk>", StringComparison.Ordinal))
            return (UnkId, 5);
        const string Sentinel = "<extra_id_";
        if (!rest.StartsWith(Sentinel, StringComparison.Ordinal))
            return null;
        int digits = 0;
        while (Sentinel.Length + digits < rest.Length && char.IsAsciiDigit(rest[Sentinel.Length + digits]))
            digits++;
        if (digits == 0 || Sentinel.Length + digits >= rest.Length || rest[Sentinel.Length + digits] != '>')
            return null;
        if (!int.TryParse(rest.Slice(Sentinel.Length, digits), out int n) || n >= SentinelCount)
            return null;
        return (FirstSentinelId + n, Sentinel.Length + digits + 1);
    }
}
