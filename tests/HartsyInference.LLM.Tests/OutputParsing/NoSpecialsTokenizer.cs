namespace HartsyInference.LLM.Tests.OutputParsing;

/// <summary>A vocabulary that has none of the DeepSeek control tokens.</summary>
internal sealed class NoSpecialsTokenizer : NoBytesTokenizer
{
    public override string Decode(IReadOnlyList<int> ids) => "";
}
