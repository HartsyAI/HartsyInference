using HartsyInference.Core.Backends;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Audio;

/// <summary>Wraps a convert delegate plus the disposables a loaded model owns, so each model is a descriptor.</summary>
internal sealed class VcRunner(
    int sampleRate,
    Func<IBackend, float[], float[]?, VoiceConversionRequest, float[]> convert,
    params IDisposable?[] disposables) : IVcRunner
{
    /// <inheritdoc/>
    public int SampleRate => sampleRate;

    /// <inheritdoc/>
    public float[] Convert(IBackend backend, float[] sourceMono, float[]? targetMono, VoiceConversionRequest request) =>
        convert(backend, sourceMono, targetMono, request);

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (IDisposable? disposable in disposables)
        {
            disposable?.Dispose();
        }
    }
}
