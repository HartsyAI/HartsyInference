using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.ModelAssets.Tokenizers.Tests;

/// <summary>Numeric parity checks for the Qwen3 tokenizer used by image prompt conditioning. Goldens were
/// produced by Hugging Face tokenizers 0.22.2 from the embedded canonical qwen3_tokenizer.json with
/// <c>add_special_tokens=False</c>.</summary>
public sealed class Qwen3TokenizerTests
{
    [Theory]
    [InlineData(" leading space", new[] { 6388, 3550 })]
    [InlineData("café — π\nline 2", new[] { 924, 58858, 1959, 51745, 198, 1056, 220, 17 })]
    [InlineData("2024, 12345 lemons", new[] { 17, 15, 17, 19, 11, 220, 16, 17, 18, 19, 20, 512, 23570 })]
    public void EncodeRaw_MatchesHuggingFaceByteLevelIds(string text, int[] expected)
    {
        using Qwen3Tokenizer tokenizer = new();
        Assert.Equal(expected, tokenizer.EncodeRaw(text));
    }

    [Fact]
    public void EncodeChat_DisabledThinkingTemplate_MatchesHuggingFaceIds()
    {
        using Qwen3Tokenizer tokenizer = new(maxLength: 32);

        int[] actual = tokenizer.EncodeChat(" leading space");
        int[] expectedPrefix =
        [
            Qwen3Tokenizer.ImStartId, 872, 198, 6388, 3550, Qwen3Tokenizer.ImEndId, 198,
            Qwen3Tokenizer.ImStartId, 77091, 198, Qwen3Tokenizer.ThinkStartId, 271,
            Qwen3Tokenizer.ThinkEndId, 271,
        ];

        Assert.Equal(expectedPrefix, actual[..expectedPrefix.Length]);
        Assert.All(actual[expectedPrefix.Length..], id => Assert.Equal(Qwen3Tokenizer.BosTokenId, id));
    }

    [Fact]
    public void EncodeRaw_AppliesCanonicalNfcNormalization()
    {
        using Qwen3Tokenizer tokenizer = new();

        Assert.Equal(new[] { 924, 58858 }, tokenizer.EncodeRaw("cafe\u0301"));
    }

    [Fact]
    public void EncodeRaw_RecognizesQwenAddedTokenLiterals()
    {
        using Qwen3Tokenizer tokenizer = new();

        Assert.Equal(new[] { Qwen3Tokenizer.ThinkStartId, 271, Qwen3Tokenizer.ThinkEndId },
            tokenizer.EncodeRaw("<think>\n\n</think>"));
    }

    [Fact]
    public void Encode_FixedLengthUsesExactBpeThenEosAndPadding()
    {
        using Qwen3Tokenizer tokenizer = new(maxLength: 6);

        Assert.Equal(new[]
        {
            6388, 3550, Qwen3Tokenizer.EosTokenId,
            Qwen3Tokenizer.BosTokenId, Qwen3Tokenizer.BosTokenId, Qwen3Tokenizer.BosTokenId,
        }, tokenizer.Encode(" leading space"));
    }

    [Fact]
    public void EncodeChat_EnabledThinkingOrVlTemplate_MatchesHuggingFaceIds()
    {
        using Qwen3Tokenizer tokenizer = new(maxLength: 24);

        int[] actual = tokenizer.EncodeChat(" leading space", includeThinkBlock: false);
        int[] expectedPrefix =
        [
            Qwen3Tokenizer.ImStartId, 872, 198, 6388, 3550, Qwen3Tokenizer.ImEndId, 198,
            Qwen3Tokenizer.ImStartId, 77091, 198,
        ];

        Assert.Equal(expectedPrefix, actual[..expectedPrefix.Length]);
        Assert.All(actual[expectedPrefix.Length..], id => Assert.Equal(Qwen3Tokenizer.BosTokenId, id));
    }

    [Fact]
    public void EncodeChat_LeadingNewlineMergesAcrossTemplateAndPromptBoundary()
    {
        using Qwen3Tokenizer tokenizer = new(maxLength: 16);

        int[] actual = tokenizer.EncodeChat("\ncat", includeThinkBlock: false);
        int[] expectedPrefix =
        [
            Qwen3Tokenizer.ImStartId, 872, 271, 4616, Qwen3Tokenizer.ImEndId, 198,
            Qwen3Tokenizer.ImStartId, 77091, 198,
        ];

        Assert.Equal(expectedPrefix, actual[..expectedPrefix.Length]);
        Assert.All(actual[expectedPrefix.Length..], id => Assert.Equal(Qwen3Tokenizer.BosTokenId, id));
    }

    [Fact]
    public void EncodeChatWithLength_PreservesRealTrailingEndOfTextToken()
    {
        using Qwen3Tokenizer tokenizer = new(maxLength: 4);

        (int[] tokenIds, int realLength) = tokenizer.EncodeChatWithLength(
            "<|endoftext|>", includeThinkBlock: false);

        Assert.Equal(new[] { Qwen3Tokenizer.ImStartId, 872, 198, Qwen3Tokenizer.BosTokenId }, tokenIds);
        Assert.Equal(tokenIds.Length, realLength);
        // This demonstrates why ID-only inference is not a safe substitute for returned metadata.
        Assert.Equal(3, Qwen3Tokenizer.CreateAttentionMask(tokenIds).Sum());
    }
}
