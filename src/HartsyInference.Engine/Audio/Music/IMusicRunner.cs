using HartsyInference.Core.Backends;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Audio;

/// <summary>A loaded music model reduced to: prompt → PCM at <see cref="SampleRate"/>. The cancellation token is observed inside the synth loop so a stop request interrupts long autoregressive decodes mid-flight.</summary>
internal interface IMusicRunner : IDisposable
{
    /// <summary>Output sample rate in Hz.</summary>
    int SampleRate { get; }

    /// <summary>Generates audio for <paramref name="request"/>.</summary>
    MusicAudio Synthesize(IBackend backend, MusicRequest request, CancellationToken cancel);
}
