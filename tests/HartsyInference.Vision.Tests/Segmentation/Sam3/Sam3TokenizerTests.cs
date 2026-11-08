using System.Text.Json;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Vision.Segmentation.Sam3;
using Xunit;

namespace HartsyInference.Vision.Tests.Segmentation.Sam3;

/// <summary>Behavior of the SAM 3 prompt tokenizer: framing, padding, normalization and truncation. Expected token ids are read from the
/// embedded CLIP vocabulary at test time rather than written out.</summary>
public sealed class Sam3TokenizerTests : IDisposable
{
    private const int ContextLength = 32;
    private const int StartOfTextId = 49406;
    private const int EndOfTextId = 49407;

    private readonly Sam3Tokenizer _tokenizer = new Sam3Tokenizer();
    private readonly Dictionary<string, int> _vocab = LoadVocab();

    /// <summary>Releases the tokenizer created for this test.</summary>
    public void Dispose() => _tokenizer.Dispose();

    [Fact]
    public void Encode_SamePrompt_ReturnsIdenticalIdsOnRepeatedCalls()
    {
        int[] first = _tokenizer.Encode("a red car on the road");
        int[] second = _tokenizer.Encode("a red car on the road");

        Assert.Equal(first, second);
    }

    [Fact]
    public void Encode_SamePromptOnTwoInstances_ReturnsIdenticalIds()
    {
        using Sam3Tokenizer other = new Sam3Tokenizer();

        Assert.Equal(_tokenizer.Encode("a yellow bus"), other.Encode("a yellow bus"));
    }

    [Fact]
    public void Encode_AnyPrompt_ReturnsContextLengthIds()
    {
        int[] ids = _tokenizer.Encode("a cat");

        Assert.Equal(ContextLength, ids.Length);
    }

    [Fact]
    public void Encode_AnyPrompt_StartsWithStartOfTextId()
    {
        int[] ids = _tokenizer.Encode("a cat");

        Assert.Equal(StartOfTextId, ids[0]);
    }

    [Fact]
    public void Encode_ShortPrompt_PlacesEndOfTextAfterLastContentTokenAndPadsWithZero()
    {
        int[] ids = _tokenizer.Encode("a cat");

        Assert.Equal(new[] { StartOfTextId, _vocab["a</w>"], _vocab["cat</w>"], EndOfTextId }, ids[..4]);
        Assert.All(ids[4..], id => Assert.Equal(0, id));
    }

    [Fact]
    public void Encode_MixedCase_MatchesLowercasePrompt()
    {
        Assert.Equal(_tokenizer.Encode("a cat"), _tokenizer.Encode("A Cat"));
    }

    [Fact]
    public void Encode_RepeatedAndTrailingWhitespace_DoesNotChangeIds()
    {
        Assert.Equal(_tokenizer.Encode("a cat"), _tokenizer.Encode("a    cat  "));
    }

    [Fact]
    public void Encode_EmptyPrompt_ReturnsStartEndAndPadding()
    {
        int[] ids = _tokenizer.Encode(string.Empty);

        Assert.Equal(new[] { StartOfTextId, EndOfTextId }, ids[..2]);
        Assert.All(ids[2..], id => Assert.Equal(0, id));
    }

    [Fact]
    public void Encode_OverlongPrompt_TruncatesToContextLengthWithEndOfTextLast()
    {
        string[] words = new string[100];
        Array.Fill(words, "cat");
        string prompt = string.Join(' ', words);

        int[] ids = _tokenizer.Encode(prompt);

        Assert.Equal(ContextLength, ids.Length);
        Assert.Equal(StartOfTextId, ids[0]);
        Assert.Equal(EndOfTextId, ids[ContextLength - 1]);
        // SOT, 30 content tokens, EOT: the content is cut to what fits, and no padding remains.
        Assert.All(ids[1..(ContextLength - 1)], id => Assert.Equal(_vocab["cat</w>"], id));
        Assert.DoesNotContain(0, ids);
    }

    [Fact]
    public void Encode_AccentedWord_UsesUtf8ByteLevelTokens()
    {
        // Upstream maps the UTF-8 bytes of "é" (C3 A9) to the chars "Ã" "©" before BPE; the merged piece is "Ã©</w>".
        int[] ids = _tokenizer.Encode("é");

        Assert.Equal(new[] { StartOfTextId, _vocab["Ã©</w>"], EndOfTextId }, ids[..3]);
    }

    [Fact]
    public void Encode_NullPrompt_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => _tokenizer.Encode(null!));
    }

    [Fact]
    public void Encode_AfterDispose_ThrowsObjectDisposedException()
    {
        Sam3Tokenizer disposed = new Sam3Tokenizer();
        disposed.Dispose();

        Assert.Throws<ObjectDisposedException>(() => disposed.Encode("a cat"));
    }

    private static Dictionary<string, int> LoadVocab()
    {
        using Stream stream = EmbeddedTokenizerResources.OpenClipVocab();
        Dictionary<string, int>? vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(stream);
        return vocab ?? throw new InvalidOperationException("The embedded CLIP vocabulary did not parse.");
    }
}
