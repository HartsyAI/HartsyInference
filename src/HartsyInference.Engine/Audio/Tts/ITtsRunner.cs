using HartsyInference.Core.Backends;

namespace HartsyInference.Engine.Audio;

/// <summary>A loaded text-to-speech model reduced to: text (plus an optional voice reference) → mono PCM at
/// <see cref="SampleRate"/>.</summary>
internal interface ITtsRunner : IDisposable
{
    /// <summary>Output sample rate in Hz.</summary>
    int SampleRate { get; }

    /// <summary>Synthesizes the job on the shared compute <paramref name="backend"/>.</summary>
    float[] Synthesize(IBackend backend, TtsJob job);
}
