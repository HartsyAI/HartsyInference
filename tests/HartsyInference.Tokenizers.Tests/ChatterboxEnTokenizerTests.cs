using HartsyInference.Tokenizers;
using Xunit;

namespace HartsyInference.Tokenizers.Tests;

/// <summary>Bit-exact parity tests for <see cref="ChatterboxEnTokenizer"/>. The expected ids were produced by the
/// reference HuggingFace <c>tokenizers</c> library loading the same embedded <c>chatterbox_tokenizer.json</c> and
/// encoding <c>text.replace(' ', '[SPACE]')</c> (the upstream <c>EnTokenizer.encode</c>).</summary>
public sealed class ChatterboxEnTokenizerTests
{
    private static readonly ChatterboxEnTokenizer Tok = new();

    [Theory]
    [InlineData("hello world", new[] { 62, 84, 28, 2, 179, 79 })]
    [InlineData("the cat sat on the mat", new[] { 42, 2, 16, 48, 2, 32, 48, 2, 47, 2, 42, 2, 26, 48 })]
    [InlineData("it's a test, really?", new[] { 60, 4, 32, 2, 14, 2, 33, 218, 7, 2, 46, 195, 13 })]
    [InlineData("Ask not what your country can do.",
        new[] { 277, 32, 24, 2, 149, 2, 193, 2, 223, 2, 16, 146, 144, 38, 2, 201, 2, 134, 9 })]
    public void Encode_MatchesReferenceTokenizer(string text, int[] expected)
    {
        Assert.Equal(expected, Tok.Encode(text));
    }

    [Fact]
    public void SpecialIds_AreTheExpectedConstants()
    {
        Assert.Equal(0, Tok.StopId);
        Assert.Equal(1, Tok.UnkId);
        Assert.Equal(2, Tok.SpaceId);
        Assert.Equal(255, Tok.StartId);
    }

    [Fact]
    public void EncodeWithStartStop_WrapsSequence()
    {
        int[] ids = Tok.EncodeWithStartStop("hello world");
        Assert.Equal(Tok.StartId, ids[0]);
        Assert.Equal(Tok.StopId, ids[^1]);
        Assert.Equal(new[] { 255, 62, 84, 28, 2, 179, 79, 0 }, ids);
    }

    [Fact]
    public void SoundEffectTag_IsMatchedWhole()
    {
        // [laughter] is added-token id 607; it must map to a single id, not be BPE-split.
        int[] ids = Tok.Encode("[laughter]");
        Assert.Equal(new[] { 607 }, ids);
    }
}
