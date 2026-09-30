using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>A speech-to-text model held resident on its engine for direct, synchronous transcription of PCM, opened by
/// <see cref="ITranscribeService.OpenTranscriberAsync"/>.</summary>
/// <remarks>Same contract as <see cref="ISynthesizerLease"/>:
/// <list type="bullet">
/// <item>one thread at a time;</item>
/// <item>the caller holds <see cref="DeviceGate"/> around each call, because the lease takes neither it nor the engine's
/// audio generation lock;</item>
/// <item>it runs on the engine's own backend;</item>
/// <item>the runner is pinned through memory-pressure eviction;</item>
/// <item>any engine release revokes the lease, which surfaces as <see cref="ObjectDisposedException"/> on the next
/// call, and the holder disposes it and opens a new one;</item>
/// <item>Dispose is idempotent.</item>
/// </list></remarks>
public interface ITranscriberLease : IDisposable
{
    /// <summary>Transcribes mono <paramref name="pcm"/> in [-1, 1] sampled at <paramref name="sampleRate"/> Hz on the
    /// calling thread, resampling only when that differs from the model's input rate. <paramref name="options"/>
    /// supplies the language and task exactly as <see cref="ITranscribeService.RunAsync"/> reads them; its
    /// <see cref="AudioRequest.Audio"/> is not read, and word timestamps or diarization are refused because a lease
    /// returns plain text.</summary>
    string Transcribe(ReadOnlySpan<float> pcm, int sampleRate, AudioRequest options);
}
