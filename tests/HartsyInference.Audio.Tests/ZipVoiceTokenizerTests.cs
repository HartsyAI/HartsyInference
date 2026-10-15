using HartsyInference.Audio.Models.ZipVoice;
using HartsyInference.Audio.Tests.Parity;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>The parts of <see cref="ZipVoiceTokenizer"/> the upstream fixture does not reach: bracketed tags, the languages
/// it rejects, newlines and the vocabulary lookup.</summary>
public sealed class ZipVoiceTokenizerTests
{
    private static readonly ZipVoiceFixture.LexiconPhonemizer Phonemizer = new(new Dictionary<string, string>
    {
        ["cat"] = "kˈæt",
        ["dog"] = "dˈɑːɡ",
    });

    [Fact]
    public void ATag_IsOneTokenBetweenTheSpeech()
    {
        ZipVoiceTokenizer tokenizer = ZipVoiceFixture.PadOnlyTokenizer();
        string[] tokens = tokenizer.Tokenize("cat [S1] dog.", Phonemizer).ToArray();
        Assert.Contains("[S1]", tokens);
        Assert.Equal("kˈæt[S1]dˈɑːɡ.", string.Concat(tokens));
    }

    [Fact]
    public void Newlines_ReadAsSpaces()
    {
        ZipVoiceTokenizer tokenizer = ZipVoiceFixture.PadOnlyTokenizer();
        Assert.Equal("kˈæt dˈɑːɡ.", string.Concat(tokenizer.Tokenize("cat\ndog.", Phonemizer)));
    }

    [Theory]
    [InlineData("cat \u4f60\u597d dog.")]
    [InlineData("cat \u3400 dog.")]
    [InlineData("cat \ud840\udc00 dog.")]
    [InlineData("cat \uf900 dog.")]
    public void HanText_IsRejectedInEveryBlock(string text)
    {
        ZipVoiceTokenizer tokenizer = ZipVoiceFixture.PadOnlyTokenizer();
        Assert.Throws<NotSupportedException>(() => tokenizer.Tokenize(text, Phonemizer));
    }

    [Fact]
    public void TextWithoutLetters_HasNoTokens()
    {
        ZipVoiceTokenizer tokenizer = ZipVoiceFixture.PadOnlyTokenizer();
        Assert.Empty(tokenizer.Tokenize("42.", Phonemizer));
    }

    [Fact]
    public void ToIds_SkipsSymbolsMissingFromTheVocabulary()
    {
        string path = Path.Combine(Path.GetTempPath(), $"zipvoice_tokens_{Guid.NewGuid():N}.txt");
        File.WriteAllText(path, "_\t0\nk\t7\nˈ\t8\nt\t9\n");
        try
        {
            ZipVoiceTokenizer tokenizer = ZipVoiceTokenizer.FromFile(path);
            Assert.Equal([7, 8, 9], tokenizer.Encode("cat", Phonemizer));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
