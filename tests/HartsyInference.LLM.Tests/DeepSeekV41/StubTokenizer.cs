using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>Deterministic stand-in tokenizer: the image placeholder literal maps to id 129264 and every other character maps to its UTF-16 code.</summary>
internal sealed class StubTokenizer : ILlmTokenizer
{
    public const string Placeholder = "<｜deepseek_image｜>";
    public const int PlaceholderId = 129264;

    public string? LastText { get; private set; }

    public int[] Encode(string text, bool addSpecial)
    {
        LastText = text;
        List<int> ids = [];
        int i = 0;
        while (i < text.Length)
        {
            if (string.CompareOrdinal(text, i, Placeholder, 0, Placeholder.Length) == 0)
            {
                ids.Add(PlaceholderId);
                i += Placeholder.Length;
            }
            else ids.Add(text[i++]);
        }
        return [.. ids];
    }

    public int[] EncodeOrdinary(string text) => Encode(text, false);

    public string Decode(IReadOnlyList<int> ids) => throw new NotSupportedException();

    public int? SpecialId(string token) => token == Placeholder ? PlaceholderId : null;

    public int? BosId => null;

    public int? EosId => null;

    public IReadOnlyList<int> StopIds => [];

    public string? BosToken => null;

    public string? EosToken => null;
}
