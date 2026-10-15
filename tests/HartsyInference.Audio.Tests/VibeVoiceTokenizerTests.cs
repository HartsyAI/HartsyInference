using HartsyInference.Audio.Models.VibeVoice;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Audio.Tests;

public sealed class VibeVoiceTokenizerTests
{
    [Fact]
    public void AngleBracketLiteral_EncodesAsOrdinaryText()
    {
        using VibeVoiceTokenizer tokenizer = new();
        using Qwen2Tokenizer exact = Qwen2Tokenizer.CreateBaseVocabulary();

        int[] ids = tokenizer.Encode("<think>");

        // Upstream Qwen2.5 has no <think> added token, so the characters encode as ordinary text (HF ids: 13708 766 29).
        int[] ordinary = [13708, 766, 29];
        Assert.Equal(ordinary, ids);
        Assert.Equal(exact.EncodeExact("<think>"), ids);
    }
}
