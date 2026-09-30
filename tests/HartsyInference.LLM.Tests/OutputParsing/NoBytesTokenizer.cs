using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Tests.OutputParsing;

/// <summary>Minimal tokenizer without per-token bytes, so <see cref="ILlmTokenizer.TokenBytes"/> keeps its null default.</summary>
internal abstract class NoBytesTokenizer : ILlmTokenizer
{
    public abstract string Decode(IReadOnlyList<int> ids);

    public int? SpecialId(string token) => null;

    public int[] Encode(string text, bool addSpecial) => [];

    public int[] EncodeOrdinary(string text) => [];

    public int? BosId => null;

    public int? EosId => null;

    public IReadOnlyList<int> StopIds => [];

    public string? BosToken => null;

    public string? EosToken => null;
}
