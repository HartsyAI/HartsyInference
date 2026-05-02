using SharpInference.Tests.Common;
using SharpInference.Tokenizers;
using Xunit;

namespace SharpInference.Tokenizers.Tests;

/// <summary>Verifies the parameterless constructors that load tokenizers from
/// embedded assembly resources. The critical correctness check: an embedded-resource
/// tokenizer must produce byte-identical output to the same tokenizer constructed
/// from the original on-disk vocab/merges/spiece file. If they ever diverge (e.g.
/// because of an LF/CRLF mismatch when copying a merges file, or a truncated copy)
/// every diffusion pipeline using these defaults would silently produce wrong
/// conditioning embeddings.</summary>
public sealed class EmbeddedResourceTokenizerTests
{
    // ── CLIP ────────────────────────────────────────────────────────────

    [Fact]
    public void EmbeddedClipTokenizer_Constructs()
    {
        using ClipTokenizer tok = new();
        Assert.NotNull(tok);
    }

    [Fact]
    public void EmbeddedClipTokenizer_MatchesFileBased()
    {
        if (!File.Exists(TestPaths.Tokenizers.ClipVocab) || !File.Exists(TestPaths.Tokenizers.ClipMerges))
        {
            return; // file-based reference unavailable on this machine
        }
        using ClipTokenizer fromFile = new(TestPaths.Tokenizers.ClipVocab, TestPaths.Tokenizers.ClipMerges);
        using ClipTokenizer fromEmbed = new();

        foreach (string prompt in new[] { "a photo of a cat", "hello world", "", "magnificent dragon flying over a castle at sunset" })
        {
            int[] expected = fromFile.Encode(prompt);
            int[] actual = fromEmbed.Encode(prompt);
            Assert.Equal(expected, actual);
        }
    }

    // ── T5 ──────────────────────────────────────────────────────────────

    [Fact]
    public void EmbeddedT5Tokenizer_Constructs()
    {
        using T5Tokenizer tok = new();
        Assert.NotNull(tok);
    }

    [Fact]
    public void EmbeddedT5Tokenizer_RespectsMaxLength()
    {
        using T5Tokenizer tok = new(maxLength: 256);
        int[] tokens = tok.Encode("hello");
        Assert.Equal(256, tokens.Length);
    }

    [Fact]
    public void EmbeddedT5Tokenizer_MatchesFileBased()
    {
        if (!File.Exists(TestPaths.Tokenizers.T5XxlSpiece))
        {
            return;
        }
        using T5Tokenizer fromFile = new(TestPaths.Tokenizers.T5XxlSpiece, maxLength: 256);
        using T5Tokenizer fromEmbed = new(maxLength: 256);

        foreach (string prompt in new[] { "a photo of a cat", "hello world", "", "magnificent dragon flying over a castle at sunset" })
        {
            int[] expected = fromFile.Encode(prompt);
            int[] actual = fromEmbed.Encode(prompt);
            Assert.Equal(expected, actual);
        }
    }

    // ── Qwen3 ───────────────────────────────────────────────────────────

    [Fact]
    public void EmbeddedQwen3Tokenizer_Constructs()
    {
        using Qwen3Tokenizer tok = new();
        Assert.NotNull(tok);
    }

    [Fact]
    public void EmbeddedQwen3Tokenizer_RespectsMaxLength()
    {
        using Qwen3Tokenizer tok = new(maxLength: 64);
        int[] tokens = tok.Encode("hello");
        Assert.Equal(64, tokens.Length);
    }

    [Fact]
    public void EmbeddedQwen3Tokenizer_MatchesFileBased()
    {
        if (!File.Exists(TestPaths.Tokenizers.Qwen3Vocab) || !File.Exists(TestPaths.Tokenizers.Qwen3Merges))
        {
            return;
        }
        using Qwen3Tokenizer fromFile = new(TestPaths.Tokenizers.Qwen3Vocab, TestPaths.Tokenizers.Qwen3Merges, maxLength: 256);
        using Qwen3Tokenizer fromEmbed = new(maxLength: 256);

        foreach (string prompt in new[] { "a photo of a cat", "hello world", "", "magnificent dragon flying over a castle at sunset" })
        {
            int[] expected = fromFile.Encode(prompt);
            int[] actual = fromEmbed.Encode(prompt);
            Assert.Equal(expected, actual);
        }
    }
}
