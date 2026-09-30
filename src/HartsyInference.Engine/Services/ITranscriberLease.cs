using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>A speech-to-text model held resident on its engine for direct, synchronous transcription of PCM, opened by
/// <see cref="ITranscribeService.OpenTranscriberAsync"/>.</summary>
/// <remarks>Same contract as <see cref="ISynthesizerLease"/>: one thread at a time; the caller holds
/// <see cref="DeviceGate"/> around each call because the lease takes neither it nor the engine's audio generation lock;
/// the runner is pinned through memory-pressure eviction; any engine release revokes the lease, after which calls throw
/// <see cref="ObjectDisposedException"/>; Dispose is idempotent.</remarks>
public interface ITranscriberLease : IDisposable
{
    /// <summary>Transcribes mono <paramref name="pcm"/> in [-1, 1] sampled at <paramref name="sampleRate"/> Hz on the
    /// calling thread, resampling only when that differs from the model's input rate. <paramref name="options"/>
    /// supplies the language and task exactly as <see cref="ITranscribeService.RunAsync"/> reads them; its
    /// <see cref="AudioRequest.Audio"/> is not read, and word timestamps or diarization are refused because a lease
    /// returns plain text.</summary>
    string Transcribe(ReadOnlySpan<float> pcm, int sampleRate, AudioRequest options);
}
