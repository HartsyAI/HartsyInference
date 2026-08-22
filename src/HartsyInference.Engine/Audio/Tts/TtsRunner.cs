using HartsyInference.Core.Backends;

namespace HartsyInference.Engine.Audio;

/// <summary>Wraps a synth delegate plus the disposables a loaded model owns (pipeline, and the weight loaders whose buffers its tensors reference), so each model is a descriptor rather than a runner class.</summary>
internal sealed class TtsRunner(int sampleRate, Func<IBackend, TtsJob, float[]> synth, params IDisposable?[] disposables) : ITtsRunner
{
    /// <inheritdoc/>
    public int SampleRate => sampleRate;

    /// <inheritdoc/>
    public float[] Synthesize(IBackend backend, TtsJob job) => synth(backend, job);

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (IDisposable? disposable in disposables)
        {
            disposable?.Dispose();
        }
    }
}
