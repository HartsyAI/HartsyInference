using System.Text;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Tests.OutputParsing;

/// <summary>Test tokenizer whose ids are handed out per piece of bytes, so a test controls exactly where a completion is split; the five DeepSeek control literals are single special ids.</summary>
internal sealed class PieceTokenizer : ILlmTokenizer
{
    public const string Bos = "<｜begin▁of▁sentence｜>";
    public const string Eos = "<｜end▁of▁sentence｜>";
    public const string Think = "<think>";
    public const string ThinkEnd = "</think>";
    public const string Dsml = "｜DSML｜";
    public static readonly string[] Specials = [Bos, Eos, Think, ThinkEnd, Dsml];

    private readonly List<byte[]> _pieces = [];

    public int Add(byte[] bytes)
    {
        _pieces.Add(bytes);
        return _pieces.Count - 1 + Specials.Length;
    }

    public int SpecialIdOf(string literal) => Array.IndexOf(Specials, literal);

    /// <summary>Splits the UTF-8 bytes at <paramref name="points"/> (ascending byte offsets) into pieces, control literals kept as plain text bytes.</summary>
    public int[] SplitBytes(string text, params int[] points)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(text);
        List<int> ids = [];
        int start = 0;
        foreach (int requested in points.Append(bytes.Length))
        {
            int point = Math.Min(requested, bytes.Length);
            if (point > start) ids.Add(Add(bytes[start..point]));
            start = Math.Max(start, point);
        }
        return [.. ids];
    }

    /// <summary>Tokenizes like the real vocabulary does: control literals become their special id, the text between them is cut into random 1..<paramref name="maxPiece"/>-byte pieces.</summary>
    public int[] RandomWithSpecials(string text, Random random, int maxPiece)
    {
        List<int> ids = [];
        int i = 0;
        while (i < text.Length)
        {
            string? literal = Specials.FirstOrDefault(s => string.CompareOrdinal(text, i, s, 0, s.Length) == 0);
            if (literal is not null)
            {
                ids.Add(SpecialIdOf(literal));
                i += literal.Length;
                continue;
            }
            int next = text.Length;
            foreach (string s in Specials)
            {
                int at = text.IndexOf(s, i, StringComparison.Ordinal);
                if (at >= 0 && at < next) next = at;
            }
            byte[] bytes = Encoding.UTF8.GetBytes(text[i..next]);
            int pos = 0;
            while (pos < bytes.Length)
            {
                int len = Math.Min(bytes.Length - pos, random.Next(1, maxPiece + 1));
                ids.Add(Add(bytes[pos..(pos + len)]));
                pos += len;
            }
            i = next;
        }
        return [.. ids];
    }

    public byte[]? TokenBytes(int id, bool includeSpecial)
    {
        if (id < Specials.Length) return includeSpecial ? Encoding.UTF8.GetBytes(Specials[id]) : [];
        return _pieces[id - Specials.Length];
    }

    public string Decode(IReadOnlyList<int> ids)
    {
        List<byte> all = [];
        foreach (int id in ids) all.AddRange(TokenBytes(id, false)!);
        return Encoding.UTF8.GetString([.. all]);
    }

    public int? SpecialId(string token) => Array.IndexOf(Specials, token) is int i and >= 0 ? i : null;

    public int[] Encode(string text, bool addSpecial) => throw new NotSupportedException();

    public int[] EncodeOrdinary(string text) => throw new NotSupportedException();

    public int? BosId => 0;

    public int? EosId => 1;

    public IReadOnlyList<int> StopIds => [1];

    public string? BosToken => Bos;

    public string? EosToken => Eos;
}
