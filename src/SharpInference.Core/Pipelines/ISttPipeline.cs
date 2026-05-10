namespace SharpInference.Core.Pipelines;

/// <summary>Speech-to-text pipeline interface.</summary>
public interface ISttPipeline : IAudioPipeline
{
    /// <summary>Transcribes audio data to text.</summary>
    Task<TranscriptionResult> TranscribeAsync(ReadOnlyMemory<byte> audioData, CancellationToken cancellationToken = default);

    /// <summary>Transcribes audio in streaming chunks for real-time processing.</summary>
    IAsyncEnumerable<TranscriptionSegment> TranscribeStreamingAsync(IAsyncEnumerable<ReadOnlyMemory<byte>> audioChunks, CancellationToken cancellationToken = default);
}
