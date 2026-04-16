namespace SharpInference.Core.Pipelines;

/// <summary>Interface for audio pipelines (Whisper STT, Kokoro TTS, voice conversion).</summary>
public interface IAudioPipeline : IDisposable
{
    /// <summary>Name of the model loaded in this pipeline.</summary>
    string ModelName { get; }
}

/// <summary>Speech-to-text pipeline interface.</summary>
public interface ISttPipeline : IAudioPipeline
{
    /// <summary>Transcribes audio data to text.</summary>
    Task<TranscriptionResult> TranscribeAsync(ReadOnlyMemory<byte> audioData, CancellationToken cancellationToken = default);

    /// <summary>Transcribes audio in streaming chunks for real-time processing.</summary>
    IAsyncEnumerable<TranscriptionSegment> TranscribeStreamingAsync(IAsyncEnumerable<ReadOnlyMemory<byte>> audioChunks, CancellationToken cancellationToken = default);
}

/// <summary>Text-to-speech pipeline interface.</summary>
public interface ITtsPipeline : IAudioPipeline
{
    /// <summary>Synthesizes speech from text, streaming audio chunks as they're generated.</summary>
    IAsyncEnumerable<ReadOnlyMemory<byte>> SynthesizeAsync(string text, string voice, CancellationToken cancellationToken = default);
}

/// <summary>Result of a speech-to-text transcription.</summary>
public sealed class TranscriptionResult
{
    /// <summary>The transcribed text.</summary>
    public required string Text { get; init; }

    /// <summary>Detected language code (e.g., "en", "es").</summary>
    public string? Language { get; init; }

    /// <summary>Individual segments with timestamps.</summary>
    public IReadOnlyList<TranscriptionSegment>? Segments { get; init; }
}

/// <summary>A segment of transcribed audio with timing information.</summary>
public sealed class TranscriptionSegment
{
    /// <summary>Transcribed text for this segment.</summary>
    public required string Text { get; init; }

    /// <summary>Start time in the audio.</summary>
    public TimeSpan Start { get; init; }

    /// <summary>End time in the audio.</summary>
    public TimeSpan End { get; init; }
}
