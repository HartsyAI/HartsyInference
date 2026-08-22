using HartsyInference.Core.Backends;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Audio;

/// <summary>A loaded speech-to-text model reduced to: transcribe a mono waveform (already at the descriptor's input rate) to text, and — where the architecture emits timestamps — to timed segments.</summary>
internal interface ISttRunner : IDisposable
{
    /// <summary>Transcribes <paramref name="audioMono"/> under the request's language/task options.</summary>
    string Transcribe(IBackend backend, float[] audioMono, AudioRequest request);

    /// <summary>Transcribes into timestamped spans, or returns null when the architecture has no timestamp mechanism (Moonshine and Kyutai emit no timestamp tokens, so they always return null here).</summary>
    IReadOnlyList<SttSegment>? TranscribeTimed(IBackend backend, float[] audioMono, AudioRequest request);
}
