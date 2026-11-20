using System.Text;
using HartsyInference.Audio.Models.Haru;
using Xunit;

namespace HartsyInference.Audio.Tests.Haru;

public sealed class HaruStreamDetokenizerTests
{
    private const int Pad = 100;
    private const int Eop = 101;

    // Byte tokens: each id below 256 is one UTF-8 byte, so a multi-byte character spans several tokens.
    private static string Decode(IReadOnlyList<int> ids) => Encoding.UTF8.GetString([.. ids.Select(static i => (byte)i)]);

    [Fact]
    public void ARunOfSilentFramesReadsAsOneSpace()
    {
        HaruStreamDetokenizer d = new(Decode, Pad, Eop);
        Assert.Equal(" ", d.Push(Pad));
        Assert.Equal(string.Empty, d.Push(Pad));
        Assert.Equal(string.Empty, d.Push(Eop));
        Assert.Equal("a", d.Push('a'));
        Assert.Equal(" ", d.Push(Eop));
    }

    [Fact]
    public void AGapGivesUpACharacterThatNeverCompleted()
    {
        HaruStreamDetokenizer d = new(Decode, Pad, Eop);
        byte[] bytes = Encoding.UTF8.GetBytes("元");
        Assert.Equal(string.Empty, d.Push(bytes[0]));
        Assert.Equal("\uFFFD ", d.Push(Pad));
        Assert.Equal("b", d.Push('b'));
    }

    [Fact]
    public void HeldBytesAreGivenUpAfterTheLongestCharacter()
    {
        HaruStreamDetokenizer d = new(Decode, Pad, Eop);
        for (int i = 1; i < HaruStreamDetokenizer.MaxHeldTokens; i++)
        {
            Assert.Equal(string.Empty, d.Push(0xE5));
        }
        string text = d.Push(0xE5);
        Assert.StartsWith("\uFFFD", text);
        Assert.Equal("a", d.Push('a'));
    }

    [Fact]
    public void ACharacterSplitOverTokensIsHeldBackUntilItIsWhole()
    {
        HaruStreamDetokenizer d = new(Decode, Pad, Eop);
        byte[] bytes = Encoding.UTF8.GetBytes("元");
        Assert.Equal(3, bytes.Length);
        Assert.Equal(string.Empty, d.Push(bytes[0]));
        Assert.Equal(string.Empty, d.Push(bytes[1]));
        Assert.Equal("元", d.Push(bytes[2]));
        Assert.Equal("b", d.Push('b'));
    }
}
