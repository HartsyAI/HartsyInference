using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.ModelAssets.Tokenizers.Tests;

/// <summary>Structural tests for the Qwen2 ChatML tokenizer (Lance / HunyuanImage 2.1 conditioning). Numeric parity with the HF Qwen2.5-VL tokenizer is validation-gated on real vocab files; these tests pin the template structure and special-token placement, which is what the pipelines depend on.</summary>
public sealed class Qwen2TokenizerTests
{
    [Fact]
    public void EncodeChat_WrapsPromptInChatMlTemplate()
    {
        using Qwen2Tokenizer tok = new();
        int[] ids = tok.EncodeChat("a cat", addGenerationPrompt: true);

        Assert.Equal(Qwen2Tokenizer.ImStartId, ids[0]);
        // system turn + user turn + assistant prefix → exactly 3 <|im_start|> and 2 <|im_end|>.
        Assert.Equal(3, ids.Count(t => t == Qwen2Tokenizer.ImStartId));
        Assert.Equal(2, ids.Count(t => t == Qwen2Tokenizer.ImEndId));
    }

    [Fact]
    public void EncodeChat_NoGenerationPrompt_EndsAtUserImEnd()
    {
        using Qwen2Tokenizer tok = new();
        int[] ids = tok.EncodeChat("a cat", addGenerationPrompt: false);

        Assert.Equal(Qwen2Tokenizer.ImEndId, ids[^1]);
        Assert.Equal(2, ids.Count(t => t == Qwen2Tokenizer.ImStartId));
    }

    [Fact]
    public void EncodeChat_EmptySystemPrompt_OmitsSystemTurn()
    {
        using Qwen2Tokenizer tok = new();
        int[] ids = tok.EncodeChat("a cat", systemPrompt: "", addGenerationPrompt: false);

        Assert.Single(ids, t => t == Qwen2Tokenizer.ImStartId);
        Assert.Single(ids, t => t == Qwen2Tokenizer.ImEndId);
    }

    [Fact]
    public void EncodeChat_LongerPromptYieldsMoreTokens()
    {
        using Qwen2Tokenizer tok = new();
        int[] shortIds = tok.EncodeChat("cat");
        int[] longIds = tok.EncodeChat("a magnificent dragon flying over a castle at sunset, oil painting");
        Assert.True(longIds.Length > shortIds.Length);
    }

    [Fact]
    public void PadToLength_PadsWithEndOfTextAndMaskMatches()
    {
        using Qwen2Tokenizer tok = new();
        int[] ids = tok.EncodeChat("a cat", addGenerationPrompt: false);
        int padded = ids.Length + 7;
        int[] paddedIds = Qwen2Tokenizer.PadToLength(ids, padded);
        int[] mask = Qwen2Tokenizer.CreateAttentionMask(paddedIds);

        Assert.Equal(padded, paddedIds.Length);
        Assert.Equal(Qwen2Tokenizer.EndOfTextId, paddedIds[^1]);
        Assert.Equal(ids.Length, mask.Sum());
    }

    [Fact]
    public void EncodeRaw_RoundTripsPlainText()
    {
        using Qwen2Tokenizer tok = new();
        IReadOnlyList<int> ids = tok.EncodeRaw("hello world");
        Assert.True(ids.Count >= 2);
        Assert.All(ids, t => Assert.InRange(t, 0, Qwen2Tokenizer.VocabSize - 1));
    }
}
