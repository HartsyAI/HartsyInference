namespace SharpInference.Tokenizers;

/// <summary>Whisper multilingual BPE tokenizer with special tokens for language, task, and timestamps. Stub implementation for Phase 5 — full implementation requires Whisper-specific vocabulary and special token handling.</summary>
public sealed class WhisperTokenizer : IDisposable
{
    /// <summary>Start of transcript token ID.</summary>
    public const int SotTokenId = 50258;

    /// <summary>End of transcript token ID.</summary>
    public const int EotTokenId = 50257;

    /// <summary>Translate task token ID.</summary>
    public const int TranslateTokenId = 50358;

    /// <summary>Transcribe task token ID.</summary>
    public const int TranscribeTokenId = 50359;

    /// <summary>No timestamps token ID.</summary>
    public const int NoTimestampsTokenId = 50363;

    private int _disposed;

    /// <summary>Creates a Whisper tokenizer. Full implementation deferred to Phase 5.</summary>
    public WhisperTokenizer()
    {
    }

    /// <summary>Tokenizes text into token IDs. Phase 5 stub — throws NotImplementedException.</summary>
    public int[] Encode(string text)
    {
        ThrowIfDisposed();
        throw new NotImplementedException("WhisperTokenizer.Encode is a Phase 5 deliverable.");
    }

    /// <summary>Decodes token IDs back to text. Phase 5 stub — throws NotImplementedException.</summary>
    public string Decode(ReadOnlySpan<int> tokenIds)
    {
        ThrowIfDisposed();
        throw new NotImplementedException("WhisperTokenizer.Decode is a Phase 5 deliverable.");
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0)
            throw new ObjectDisposedException(nameof(WhisperTokenizer));
    }

    /// <summary>Disposes resources.</summary>
    public void Dispose()
    {
        Interlocked.Exchange(ref _disposed, 1);
    }
}
