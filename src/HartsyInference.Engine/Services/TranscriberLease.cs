using HartsyInference.Audio.Io;
using HartsyInference.Core.Backends;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary><see cref="ITranscriberLease"/> over a pinned <see cref="ISttRunner"/>: takes PCM straight to the runner,
/// through the same resampler <see cref="AudioClipCodec"/> uses when the rate differs, and runs it on the engine's
/// backend under the lease's call lock.</summary>
internal sealed class TranscriberLease : AudioRunnerLease, ITranscriberLease
{
    private readonly ISttRunner _runner;
    private readonly IBackend _backend;
    private readonly int _inputSampleRate;

    /// <summary>Pins <paramref name="key"/> in the engine's STT cache; <paramref name="inputSampleRate"/> is the rate the
    /// runner expects.</summary>
    internal TranscriberLease(AudioRuntime runtime, string key, ISttRunner runner, IBackend backend, int inputSampleRate)
        : base(runtime, runtime.Stt, key)
    {
        _runner = runner;
        _backend = backend;
        _inputSampleRate = inputSampleRate;
    }

    /// <inheritdoc/>
    public string Transcribe(ReadOnlySpan<float> pcm, int sampleRate, AudioRequest options)
    {
        ThrowIfClosed();
        ArgumentNullException.ThrowIfNull(options);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);
        if (pcm.IsEmpty)
        {
            throw new ArgumentException("No audio supplied to transcribe.", nameof(pcm));
        }
        // The service takes a timed decode's text when these are set, so a plain decode here would not mean the same.
        if (options.WordTimestamps || options.Diarization)
        {
            throw new ArgumentException("A transcriber lease returns plain text; request word timestamps or diarization "
                + "through ITranscribeService.RunAsync.", nameof(options));
        }
        float[] audio = sampleRate == _inputSampleRate ? pcm.ToArray()
            : Resampler.Create(sampleRate, _inputSampleRate).Resample(pcm);
        string text;
        lock (CallLock)
        {
            ThrowIfClosed();
            text = _runner.Transcribe(_backend, audio, options);
        }
        return text?.Trim() ?? string.Empty;
    }
}
