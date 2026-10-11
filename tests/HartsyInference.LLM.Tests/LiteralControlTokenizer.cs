using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Tests;

/// <summary>A tokenizer with a few control literals (ids 1000+) and one id per ASCII character, so a test can count control tokens in the encoded prompt.</summary>
internal sealed class LiteralControlTokenizer : ILlmTokenizer
{
    public const string ImStart = "<|im_start|>";
    public const string ImEnd = "<|im_end|>";
    public const string ToolCall = "<tool_call>";
    public const string Bos = "<s>";
    public const string ThinkOpen = "<think>";
    public const string ThinkClose = "</think>";

    private static readonly string[] Specials = [ImStart, ImEnd, ToolCall, Bos, ThinkOpen, ThinkClose];

    public static int IdOf(string literal) => 1000 + Array.IndexOf(Specials, literal);

    public IReadOnlyList<string> SpecialLiterals { get; } = [.. Specials.OrderByDescending(s => s.Length)];

    public int[] Encode(string text, bool addSpecial)
    {
        if (!addSpecial) return EncodeOrdinary(text);
        List<int> ids = [];
        int i = 0;
        while (i < text.Length)
        {
            string? literal = SpecialLiterals.FirstOrDefault(l => string.CompareOrdinal(text, i, l, 0, l.Length) == 0 && l.Length <= text.Length - i);
            if (literal is not null)
            {
                ids.Add(IdOf(literal));
                i += literal.Length;
                continue;
            }
            ids.Add(text[i]);
            i++;
        }
        return [.. ids];
    }

    public int[] EncodeOrdinary(string text) => [.. text.Select(c => (int)c)];

    public string Decode(IReadOnlyList<int> ids) => throw new NotSupportedException();

    public int? SpecialId(string token) => Array.IndexOf(Specials, token) is int i and >= 0 ? 1000 + i : null;

    public int? BosId => null;

    public int? EosId => null;

    public IReadOnlyList<int> StopIds => [];

    public string? BosToken => Bos;

    public string? EosToken => null;
}
