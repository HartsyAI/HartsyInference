using HartsyInference.Core.Backends;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Audio;

/// <summary>Wraps a synth delegate plus the disposables a loaded model owns, so each model is a descriptor.</summary>
internal sealed class MusicRunner(
    int sampleRate,
    Func<IBackend, MusicRequest, CancellationToken, MusicAudio> synth,
    params IDisposable?[] disposables) : IMusicRunner
{
    /// <inheritdoc/>
    public int SampleRate => sampleRate;

    /// <inheritdoc/>
    public MusicAudio Synthesize(IBackend backend, MusicRequest request, CancellationToken cancel) => synth(backend, request, cancel);

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (IDisposable? disposable in disposables)
        {
            disposable?.Dispose();
        }
    }
}
