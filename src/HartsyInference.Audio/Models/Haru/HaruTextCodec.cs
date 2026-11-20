using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Audio.Models.Haru;

/// <summary>The two text conversions a conversation needs: a prompt to ids (special literals parsed) and ids to text.</summary>
internal sealed record HaruTextCodec(Func<string, int[]> Encode, Func<IReadOnlyList<int>, string> Decode)
{
    internal static HaruTextCodec From(GgufTokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        return new HaruTextCodec(text => tokenizer.Encode(text, addSpecial: true), tokenizer.Decode);
    }
}
