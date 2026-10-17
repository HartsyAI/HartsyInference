using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Tests;

/// <summary>One id per character, and negative ids for the two ChatML control tokens.</summary>
internal sealed class CharTokenizer : ILlmTokenizer
{
    private const int ImStart = -1;
    private const int ImEnd = -2;

    public static string Text(int[] ids) => string.Concat(ids.Select(id => id switch { ImStart => "<|im_start|>", ImEnd => "<|im_end|>", _ => ((char)id).ToString() }));

    public int[] Encode(string text, bool addSpecial) => [.. text.Select(c => (int)c)];

    public int[] EncodeOrdinary(string text) => Encode(text, addSpecial: false);

    public string Decode(IReadOnlyList<int> ids) => Text([.. ids]);

    public int? SpecialId(string token) => token switch { "<|im_start|>" => ImStart, "<|im_end|>" => ImEnd, _ => null };

    public int? BosId => null;

    public int? EosId => ImEnd;

    public IReadOnlyList<int> StopIds => [ImEnd];

    public string? BosToken => null;

    public string? EosToken => "<|im_end|>";
}
