using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Unit tests for <see cref="SpeechService.RejectUnsupportedChunking"/>: which <c>chunk_words</c> requests
/// are refused before a job is queued.</summary>
public sealed class TtsChunkedSynthesisRejectionTests
{
    private static TtsModelDescriptor Descriptor(bool supportsChunking) => new()
    {
        ResolveRepo = _ => "some/repo",
        LoadAsync = (_, _, _) => throw new NotSupportedException("not exercised by this test"),
        SupportsChunking = supportsChunking,
    };

    [Fact]
    public void NullChunkWords_IsAccepted()
    {
        SpeechRequest request = new() { Text = "hello" };

        SpeechService.RejectUnsupportedChunking(Descriptor(supportsChunking: false), request, streaming: false);
    }

    [Fact]
    public void ZeroChunkWords_ThrowsArgumentOutOfRange()
    {
        SpeechRequest request = new() { Text = "hello", ChunkWords = 0 };

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SpeechService.RejectUnsupportedChunking(Descriptor(supportsChunking: true), request, streaming: false));
    }

    [Fact]
    public void StreamingWithChunkWords_ThrowsNotSupportedMentioningStreaming()
    {
        SpeechRequest request = new() { Text = "hello", ChunkWords = 40 };

        NotSupportedException error = Assert.Throws<NotSupportedException>(() =>
            SpeechService.RejectUnsupportedChunking(Descriptor(supportsChunking: true), request, streaming: true));

        Assert.Contains("streaming", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void NonStreamingWithUnsupportedModel_ThrowsNotSupportedMentioningChunkWords()
    {
        SpeechRequest request = new() { Text = "hello", ChunkWords = 40 };

        NotSupportedException error = Assert.Throws<NotSupportedException>(() =>
            SpeechService.RejectUnsupportedChunking(Descriptor(supportsChunking: false), request, streaming: false));

        Assert.Contains("chunk_words", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NonStreamingWithSupportedModel_IsAccepted()
    {
        SpeechRequest request = new() { Text = "hello", ChunkWords = 40 };

        SpeechService.RejectUnsupportedChunking(Descriptor(supportsChunking: true), request, streaming: false);
    }
}
