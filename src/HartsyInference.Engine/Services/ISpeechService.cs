using HartsyInference.Audio.Streaming;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Typed text-to-speech surface, including zero-shot voice cloning from a reference clip.</summary>
public interface ISpeechService
{
    /// <summary>Synthesizes speech for <paramref name="request"/>.</summary>
    Task<AudioResult> SynthesizeAsync(ModelSpec spec, SpeechRequest request, CancellationToken cancel = default);

    /// <summary>Synthesizes speech for <paramref name="request"/>, yielding audio incrementally for models that support it (currently Kyutai TTS). Models without a streaming implementation yield exactly one chunk containing the complete synthesized buffer, so callers have a single code path regardless of which model is selected.</summary>
    IAsyncEnumerable<AudioChunk> SynthesizeStreamAsync(ModelSpec spec, SpeechRequest request, CancellationToken cancel = default);

    /// <summary>Loads the model for <paramref name="spec"/> exactly as <see cref="SynthesizeAsync"/> would (same catalog,
    /// cache key and download) and returns a lease that keeps it resident on this engine's backend until disposed; see
    /// <see cref="ISynthesizerLease"/> for the contract.</summary>
    /// <remarks>Opening runs the service path, with the engine's audio generation lock, the device gate and the
    /// memory-pressure sweep, so it waits for a generation in flight; never await it while holding
    /// <see cref="DeviceGate"/>. For a model whose voice selects its weights (Piper) the spec's variant names the
    /// voice.</remarks>
    Task<ISynthesizerLease> OpenSynthesizerAsync(ModelSpec spec, CancellationToken cancel = default) =>
        throw new NotSupportedException("This speech-service implementation does not expose synthesizer leases.");
}
