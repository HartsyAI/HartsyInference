using SharpInference.Tests.Common;
using SharpInference.Tokenizers;
using Xunit;

namespace SharpInference.Tokenizers.Tests;

/// <summary>Tests for CLIP BPE tokenizer using real OpenAI CLIP vocabulary and merges files.</summary>
public sealed class ClipTokenizerTests : IDisposable
{
    private static string VocabPath => TestPaths.Tokenizers.ClipVocab;
    private static string MergesPath => TestPaths.Tokenizers.ClipMerges;

    private readonly ClipTokenizer? _tokenizer;
    private readonly bool _modelsAvailable;

    public ClipTokenizerTests()
    {
        _modelsAvailable = File.Exists(VocabPath) && File.Exists(MergesPath);
        if (_modelsAvailable)
        {
            _tokenizer = new ClipTokenizer(VocabPath, MergesPath);
        }
    }

    public void Dispose()
    {
        _tokenizer?.Dispose();
    }

    // ── Construction Tests ─────────────────────────────────────────────

    [Fact]
    public void Construction_WithValidFiles_Succeeds()
    {
        if (!_modelsAvailable) return;
        Assert.NotNull(_tokenizer);
    }

    [Fact]
    public void Construction_WithInvalidPath_ThrowsFileNotFound()
    {
        Assert.Throws<FileNotFoundException>(() => new ClipTokenizer("nonexistent.json", "nonexistent.txt"));
    }

    // ── Encode Tests ───────────────────────────────────────────────────

    [Fact]
    public void Encode_OutputLength_Is77()
    {
        if (!_modelsAvailable) return;

        int[] tokens = _tokenizer!.Encode("hello world");

        Assert.Equal(ClipTokenizer.MaxLength, tokens.Length);
    }

    [Fact]
    public void Encode_StartsWithSOT()
    {
        if (!_modelsAvailable) return;

        int[] tokens = _tokenizer!.Encode("hello world");

        Assert.Equal(ClipTokenizer.StartOfTextId, tokens[0]);
    }

    [Fact]
    public void Encode_EndsWithEOTFollowedByPadding()
    {
        if (!_modelsAvailable) return;

        int[] tokens = _tokenizer!.Encode("hello");

        // Find EOT position — should be right after content tokens
        int eotPos = -1;
        for (int i = 1; i < tokens.Length; i++)
        {
            if (tokens[i] == ClipTokenizer.EndOfTextId)
            {
                eotPos = i;
                break;
            }
        }

        Assert.True(eotPos > 0, "EOT token not found");

        // All positions after EOT should be zero (padding)
        for (int i = eotPos + 1; i < tokens.Length; i++)
        {
            Assert.Equal(0, tokens[i]);
        }
    }

    [Fact]
    public void Encode_APhotoOfACat_MatchesPythonReference()
    {
        if (!_modelsAvailable) return;

        // Verified output from Microsoft.ML.Tokenizers BpeTokenizer with CLIP vocab/merges
        // [49406, 64, 1153, 684, 64, 1481, 49407, 0, 0, ...]
        int[] tokens = _tokenizer!.Encode("a photo of a cat");

        Assert.Equal(ClipTokenizer.StartOfTextId, tokens[0]);

        int[] expectedContent = [64, 1153, 684, 64, 1481];
        for (int i = 0; i < expectedContent.Length; i++)
        {
            Assert.Equal(expectedContent[i], tokens[i + 1]);
        }

        Assert.Equal(ClipTokenizer.EndOfTextId, tokens[expectedContent.Length + 1]);
    }

    [Fact]
    public void Encode_EmptyString_HasSOTAndEOTOnly()
    {
        if (!_modelsAvailable) return;

        int[] tokens = _tokenizer!.Encode("");

        Assert.Equal(ClipTokenizer.StartOfTextId, tokens[0]);
        Assert.Equal(ClipTokenizer.EndOfTextId, tokens[1]);

        for (int i = 2; i < tokens.Length; i++)
        {
            Assert.Equal(0, tokens[i]);
        }
    }

    [Fact]
    public void Encode_Lowercases_Input()
    {
        if (!_modelsAvailable) return;

        int[] upper = _tokenizer!.Encode("A PHOTO OF A CAT");
        int[] lower = _tokenizer!.Encode("a photo of a cat");

        // CLIP lowercases all input, so these should produce identical tokens
        Assert.Equal(lower, upper);
    }

    [Fact]
    public void Encode_Numbers_AreSingleDigitTokens()
    {
        if (!_modelsAvailable) return;

        int[] tokens = _tokenizer!.Encode("123");

        // SOT + digit tokens + EOT — digits tokenized as [16, 17, 18]
        Assert.Equal(ClipTokenizer.StartOfTextId, tokens[0]);
        Assert.Equal(16, tokens[1]);
        Assert.Equal(17, tokens[2]);
        Assert.Equal(18, tokens[3]);
        Assert.Equal(ClipTokenizer.EndOfTextId, tokens[4]);
    }

    [Fact]
    public void Encode_Punctuation_IsTokenized()
    {
        if (!_modelsAvailable) return;

        int[] tokens = _tokenizer!.Encode("hello, world!");

        // Should have more tokens than just "hello world" due to punctuation
        int eotPos = Array.IndexOf(tokens, ClipTokenizer.EndOfTextId);
        Assert.True(eotPos > 3, "Punctuation should produce additional tokens");
    }

    [Fact]
    public void Encode_LongText_TruncatesTo77()
    {
        if (!_modelsAvailable) return;

        // Generate text that will exceed 75 content tokens
        string longText = string.Join(" ", Enumerable.Repeat("magnificent", 80));
        int[] tokens = _tokenizer!.Encode(longText);

        Assert.Equal(ClipTokenizer.MaxLength, tokens.Length);
        Assert.Equal(ClipTokenizer.StartOfTextId, tokens[0]);
        // Last position should be EOT when truncated
        Assert.Equal(ClipTokenizer.EndOfTextId, tokens[ClipTokenizer.MaxLength - 1]);
    }

    // ── EncodeRaw Tests ────────────────────────────────────────────────

    [Fact]
    public void EncodeRaw_NoSOTOrEOT()
    {
        if (!_modelsAvailable) return;

        IReadOnlyList<int> raw = _tokenizer!.EncodeRaw("hello");

        Assert.DoesNotContain(ClipTokenizer.StartOfTextId, raw);
        Assert.DoesNotContain(ClipTokenizer.EndOfTextId, raw);
    }

    // ── Decode Tests ───────────────────────────────────────────────────

    [Fact]
    public void Decode_FiltersSpecialTokens()
    {
        if (!_modelsAvailable) return;

        int[] tokens = _tokenizer!.Encode("hello world");
        string decoded = _tokenizer!.Decode(tokens);

        Assert.DoesNotContain("<|startoftext|>", decoded);
        Assert.DoesNotContain("<|endoftext|>", decoded);
    }

    // ── Constants Tests ────────────────────────────────────────────────

    [Fact]
    public void Constants_MatchClipSpec()
    {
        Assert.Equal(49406, ClipTokenizer.StartOfTextId);
        Assert.Equal(49407, ClipTokenizer.EndOfTextId);
        Assert.Equal(77, ClipTokenizer.MaxLength);
        Assert.Equal(49408, ClipTokenizer.VocabSize);
    }

    // ── Dispose Tests ──────────────────────────────────────────────────

    [Fact]
    public void Encode_AfterDispose_Throws()
    {
        if (!_modelsAvailable) return;

        ClipTokenizer tokenizer = new ClipTokenizer(VocabPath, MergesPath);
        tokenizer.Dispose();

        Assert.Throws<ObjectDisposedException>(() => tokenizer.Encode("test"));
    }

    [Fact]
    public void Dispose_CalledTwice_DoesNotThrow()
    {
        if (!_modelsAvailable) return;

        ClipTokenizer tokenizer = new ClipTokenizer(VocabPath, MergesPath);
        tokenizer.Dispose();
        tokenizer.Dispose();
    }
}
