using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;

namespace HartsyInference.Engine.Audio;

/// <summary>Wraps both a synchronous synth delegate (the non-streaming <see cref="ITtsRunner.Synthesize"/> path —
/// kept byte-identical to what a plain <see cref="TtsRunner"/> would produce) and a streaming delegate, plus the
/// disposables the loaded model owns. Mirrors <see cref="TtsRunner"/>'s shape for the models that genuinely
/// support incremental output.</summary>
internal sealed class StreamingTtsRunner(
    int sampleRate,
    Func<IBackend, TtsJob, float[]> synth,
    Func<IBackend, TtsJob, CancellationToken, IAsyncEnumerable<AudioChunk>> stream,
    params IDisposable?[] disposables) : IStreamingTtsRunner
{
    /// <inheritdoc/>
    public int SampleRate => sampleRate;

    /// <inheritdoc/>
    public float[] Synthesize(IBackend backend, TtsJob job) => synth(backend, job);

    /// <inheritdoc/>
    public IAsyncEnumerable<AudioChunk> SynthesizeStream(IBackend backend, TtsJob job, CancellationToken cancel) => stream(backend, job, cancel);

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (IDisposable? disposable in disposables)
        {
            disposable?.Dispose();
        }
    }
}
