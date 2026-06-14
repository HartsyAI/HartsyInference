using HartsyInference.Tests.Common;
using HartsyInference.Tokenizers;
using Xunit;

namespace HartsyInference.Tokenizers.Tests;

/// <summary>Tests for T5 SentencePiece tokenizer using real T5 model files with protobuf bos_id patching.</summary>
public sealed class T5TokenizerTests : IDisposable
{
    private static string T5ModelPath => TestPaths.Tokenizers.T5Spiece;

    private readonly T5Tokenizer? _tokenizer;
    private readonly bool _modelAvailable;

    public T5TokenizerTests()
    {
        _modelAvailable = File.Exists(T5ModelPath);
        if (_modelAvailable)
        {
            _tokenizer = new T5Tokenizer(T5ModelPath);
        }
    }

    public void Dispose()
    {
        _tokenizer?.Dispose();
    }

    // ── Construction Tests ─────────────────────────────────────────────

    [Fact]
    public void Construction_WithValidFile_Succeeds()
    {
        if (!_modelAvailable) return;
        Assert.NotNull(_tokenizer);
    }

    [Fact]
    public void Construction_WithStream_Succeeds()
    {
        if (!_modelAvailable) return;

        using Stream stream = File.OpenRead(T5ModelPath);
        using T5Tokenizer tokenizer = new T5Tokenizer(stream);
        Assert.NotNull(tokenizer);
    }

    [Fact]
    public void Construction_WithInvalidPath_ThrowsFileNotFound()
    {
        Assert.Throws<FileNotFoundException>(() => new T5Tokenizer("nonexistent.model"));
    }

    [Fact]
    public void Construction_WithCustomMaxLength_UsesIt()
    {
        if (!_modelAvailable) return;

        using T5Tokenizer tokenizer = new T5Tokenizer(T5ModelPath, maxLength: 256);
        int[] tokens = tokenizer.Encode("hello");
        Assert.Equal(256, tokens.Length);
    }

    // ── Encode Tests ───────────────────────────────────────────────────

    [Fact]
    public void Encode_OutputLength_IsMaxLength()
    {
        if (!_modelAvailable) return;

        int[] tokens = _tokenizer!.Encode("hello world");

        Assert.Equal(T5Tokenizer.DefaultMaxLength, tokens.Length);
    }

    [Fact]
    public void Encode_HelloWorld_MatchesPythonReference()
    {
        if (!_modelAvailable) return;

        // Reference from HuggingFace T5Tokenizer: "hello world" → [21820, 296, 1]
        // Our Encode appends EOS and pads to maxLength
        int[] tokens = _tokenizer!.Encode("hello world");

        Assert.Equal(21820, tokens[0]);
        Assert.Equal(296, tokens[1]);
        Assert.Equal(T5Tokenizer.EosTokenId, tokens[2]);

        // Rest should be padding
        for (int i = 3; i < tokens.Length; i++)
        {
            Assert.Equal(T5Tokenizer.PadTokenId, tokens[i]);
        }
    }

    [Fact]
    public void Encode_APhotoOfACat_MatchesPythonReference()
    {
        if (!_modelAvailable) return;

        // Reference: "a photo of a cat" → [3, 9, 1202, 13, 3, 9, 1712, 1]
        int[] tokens = _tokenizer!.Encode("a photo of a cat");
        int[] expectedContent = [3, 9, 1202, 13, 3, 9, 1712];

        for (int i = 0; i < expectedContent.Length; i++)
        {
            Assert.Equal(expectedContent[i], tokens[i]);
        }

        Assert.Equal(T5Tokenizer.EosTokenId, tokens[expectedContent.Length]);
    }

    [Fact]
    public void Encode_ABeautifulSunset_MatchesPythonReference()
    {
        if (!_modelAvailable) return;

        // Reference: "A beautiful sunset over the ocean" → [71, 786, 13744, 147, 8, 5431, 1]
        int[] tokens = _tokenizer!.Encode("A beautiful sunset over the ocean");
        int[] expectedContent = [71, 786, 13744, 147, 8, 5431];

        for (int i = 0; i < expectedContent.Length; i++)
        {
            Assert.Equal(expectedContent[i], tokens[i]);
        }

        Assert.Equal(T5Tokenizer.EosTokenId, tokens[expectedContent.Length]);
    }

    [Fact]
    public void Encode_EmptyString_HasEosOnly()
    {
        if (!_modelAvailable) return;

        int[] tokens = _tokenizer!.Encode("");

        // First token should be EOS (no content)
        Assert.Equal(T5Tokenizer.EosTokenId, tokens[0]);

        // Rest should be padding
        for (int i = 1; i < tokens.Length; i++)
        {
            Assert.Equal(T5Tokenizer.PadTokenId, tokens[i]);
        }
    }

    [Fact]
    public void Encode_NoBosToken_FirstTokenIsContent()
    {
        if (!_modelAvailable) return;

        // T5 does NOT prepend a BOS token — first token should be content
        int[] tokens = _tokenizer!.Encode("hello");

        // First token should NOT be any special token (PAD=0, EOS=1, UNK=2)
        Assert.True(tokens[0] > 2, $"First token should be content, got {tokens[0]}");
    }

    [Fact]
    public void Encode_AlwaysEndsWithEos()
    {
        if (!_modelAvailable) return;

        string[] texts = ["hello", "a photo of a cat", "The quick brown fox"];

        foreach (string text in texts)
        {
            int[] tokens = _tokenizer!.Encode(text);

            // Find last non-padding token
            int lastContent = -1;
            for (int i = tokens.Length - 1; i >= 0; i--)
            {
                if (tokens[i] != T5Tokenizer.PadTokenId)
                {
                    lastContent = i;
                    break;
                }
            }

            Assert.Equal(T5Tokenizer.EosTokenId, tokens[lastContent]);
        }
    }

    [Fact]
    public void Encode_LongText_TruncatesToMaxLength()
    {
        if (!_modelAvailable) return;

        string longText = string.Join(" ", Enumerable.Repeat("magnificent", 200));
        int[] tokens = _tokenizer!.Encode(longText);

        Assert.Equal(T5Tokenizer.DefaultMaxLength, tokens.Length);

        // Last slot should be EOS
        Assert.Equal(T5Tokenizer.EosTokenId, tokens[T5Tokenizer.DefaultMaxLength - 1]);
    }

    // ── EncodeRaw Tests ────────────────────────────────────────────────

    [Fact]
    public void EncodeRaw_NoEosOrPadding()
    {
        if (!_modelAvailable) return;

        IReadOnlyList<int> raw = _tokenizer!.EncodeRaw("hello world");

        Assert.DoesNotContain(T5Tokenizer.EosTokenId, raw);
        Assert.DoesNotContain(T5Tokenizer.PadTokenId, raw);
    }

    [Fact]
    public void EncodeRaw_HelloWorld_Returns2Tokens()
    {
        if (!_modelAvailable) return;

        IReadOnlyList<int> raw = _tokenizer!.EncodeRaw("hello world");

        Assert.Equal(2, raw.Count);
        Assert.Equal(21820, raw[0]);
        Assert.Equal(296, raw[1]);
    }

    // ── Decode Tests ───────────────────────────────────────────────────

    [Fact]
    public void Decode_RoundTrip_PreservesText()
    {
        if (!_modelAvailable) return;

        string original = "a photo of a cat";
        int[] tokens = _tokenizer!.Encode(original);
        string decoded = _tokenizer!.Decode(tokens);

        Assert.Equal(original, decoded);
    }

    [Fact]
    public void Decode_FiltersEosAndPad()
    {
        if (!_modelAvailable) return;

        int[] tokens = _tokenizer!.Encode("hello");
        string decoded = _tokenizer!.Decode(tokens);

        Assert.DoesNotContain("</s>", decoded);
        Assert.DoesNotContain("<pad>", decoded);
    }

    // ── Attention Mask Tests ───────────────────────────────────────────

    [Fact]
    public void CreateAttentionMask_OnesForContentZerosForPad()
    {
        if (!_modelAvailable) return;

        int[] tokens = _tokenizer!.Encode("hello world");
        int[] mask = T5Tokenizer.CreateAttentionMask(tokens);

        Assert.Equal(tokens.Length, mask.Length);

        // Tokens: [21820, 296, 1, 0, 0, ...] → Mask: [1, 1, 1, 0, 0, ...]
        Assert.Equal(1, mask[0]);
        Assert.Equal(1, mask[1]);
        Assert.Equal(1, mask[2]); // EOS token gets mask=1 (it's not PAD)

        // After EOS+content, remaining should be 0
        for (int i = 3; i < mask.Length; i++)
        {
            Assert.Equal(0, mask[i]);
        }
    }

    // ── Constants Tests ────────────────────────────────────────────────

    [Fact]
    public void Constants_MatchT5Spec()
    {
        Assert.Equal(0, T5Tokenizer.PadTokenId);
        Assert.Equal(1, T5Tokenizer.EosTokenId);
        Assert.Equal(2, T5Tokenizer.UnkTokenId);
        Assert.Equal(77, T5Tokenizer.DefaultMaxLength);
    }

    // ── Dispose Tests ──────────────────────────────────────────────────

    [Fact]
    public void Encode_AfterDispose_Throws()
    {
        if (!_modelAvailable) return;

        T5Tokenizer tokenizer = new T5Tokenizer(T5ModelPath);
        tokenizer.Dispose();

        Assert.Throws<ObjectDisposedException>(() => tokenizer.Encode("test"));
    }

    [Fact]
    public void Dispose_CalledTwice_DoesNotThrow()
    {
        if (!_modelAvailable) return;

        T5Tokenizer tokenizer = new T5Tokenizer(T5ModelPath);
        tokenizer.Dispose();
        tokenizer.Dispose();
    }

    // ── SD3/Flux Max Length Tests ──────────────────────────────────────

    [Fact]
    public void Encode_Sd3MaxLength256_PadsCorrectly()
    {
        if (!_modelAvailable) return;

        using T5Tokenizer tokenizer = new T5Tokenizer(T5ModelPath, maxLength: 256);
        int[] tokens = tokenizer.Encode("hello world");

        Assert.Equal(256, tokens.Length);
        Assert.Equal(21820, tokens[0]);
        Assert.Equal(296, tokens[1]);
        Assert.Equal(T5Tokenizer.EosTokenId, tokens[2]);

        for (int i = 3; i < 256; i++)
        {
            Assert.Equal(T5Tokenizer.PadTokenId, tokens[i]);
        }
    }

    [Fact]
    public void Encode_FluxMaxLength512_PadsCorrectly()
    {
        if (!_modelAvailable) return;

        using T5Tokenizer tokenizer = new T5Tokenizer(T5ModelPath, maxLength: 512);
        int[] tokens = tokenizer.Encode("hello");

        Assert.Equal(512, tokens.Length);
    }
}
