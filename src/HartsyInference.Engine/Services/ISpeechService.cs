using HartsyInference.Audio.Streaming;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Typed text-to-speech surface, including zero-shot voice cloning from a reference clip.</summary>
public interface ISpeechService
{
    /// <summary>Synthesizes speech for <paramref name="request"/>.</summary>
    Task<AudioResult> SynthesizeAsync(ModelSpec spec, SpeechRequest request, CancellationToken cancel = default);

    /// <summary>Synthesizes speech for <paramref name="request"/>, yielding audio incrementally for models that
    /// support it (currently Kyutai TTS). Models without a streaming implementation yield exactly one chunk
    /// containing the complete synthesized buffer, so callers have a single code path regardless of which model
    /// is selected.</summary>
    IAsyncEnumerable<AudioChunk> SynthesizeStreamAsync(ModelSpec spec, SpeechRequest request, CancellationToken cancel = default);
}
