using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Typed text-to-speech surface, including zero-shot voice cloning from a reference clip.</summary>
public interface ISpeechService
{
    /// <summary>Synthesizes speech for <paramref name="request"/>.</summary>
    Task<AudioResult> SynthesizeAsync(ModelSpec spec, SpeechRequest request, CancellationToken cancel = default);
}
