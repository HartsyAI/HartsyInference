using System.IO;
using System.Text;
using HartsyInference.Audio.Frontends;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tests the English G2P front-end: ARPAbet→IPA mapping, CMUdict lookup (via a tiny in-memory dict),
/// number normalization, and the OOV letter-to-sound fallback. No large asset required.</summary>
public sealed class EnglishG2PTests
{
    [Fact]
    public void ArpabetToIpa_MapsPhonesAndStress()
    {
        Assert.Equal("həlˈoʊ", ArpabetToIpa.ConvertWord(["HH", "AH0", "L", "OW1"]));
        Assert.Equal("kˈæt", ArpabetToIpa.ConvertWord(["K", "AE1", "T"]));
        Assert.Equal("ə", ArpabetToIpa.ConvertWord(["AH0"]));   // reduced schwa
        Assert.Equal("ˈɝ", ArpabetToIpa.ConvertWord(["ER1"]));  // stressed r-colored vowel
        Assert.Equal("ˌɛ", ArpabetToIpa.ConvertWord(["EH2"]));  // secondary stress
    }

    private static EnglishG2P Dict(string entries)
        => new(new MemoryStream(Encoding.UTF8.GetBytes(entries)));

    [Fact]
    public void ToIpa_KnownWords_FromDict_WithPunctuation()
    {
        EnglishG2P g2p = Dict(";;; header\nhello HH AH0 L OW1\nworld W ER1 L D\n");
        Assert.Equal("həlˈoʊ wˈɝld", g2p.ToIpa("Hello, world!"));
        Assert.Equal(2, g2p.WordCount);
    }

    [Fact]
    public void ToIpa_FirstPronunciationWins_ForVariants()
    {
        EnglishG2P g2p = Dict("read R IY1 D\nread(2) R EH1 D\n");
        Assert.Equal("ɹˈid", g2p.ToIpa("read"));
        Assert.Equal(1, g2p.WordCount);
    }

    [Fact]
    public void ToIpa_ExpandsIntegers_ToWords()
    {
        // No dict entries → numbers expand to words, then fall through letter-to-sound.
        EnglishG2P g2p = Dict("");
        string expected = $"{EnglishG2P.LetterToSound("forty")} {EnglishG2P.LetterToSound("two")}";
        Assert.Equal(expected, g2p.ToIpa("42"));
    }

    [Fact]
    public void ToIpa_OovWord_UsesLetterToSound()
    {
        EnglishG2P g2p = Dict("");
        Assert.Equal(EnglishG2P.LetterToSound("zylophone"), g2p.ToIpa("zylophone"));
        Assert.NotEmpty(g2p.ToIpa("zylophone"));
    }

    [Fact]
    public void LetterToSound_HandlesDigraphs()
    {
        Assert.Equal("ʃ", EnglishG2P.LetterToSound("sh"));
        Assert.Equal("tʃ", EnglishG2P.LetterToSound("ch"));
        Assert.StartsWith("θ", EnglishG2P.LetterToSound("the"));
    }
}
