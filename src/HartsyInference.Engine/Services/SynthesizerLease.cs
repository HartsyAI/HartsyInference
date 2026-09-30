using HartsyInference.Core.Backends;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary><see cref="ISynthesizerLease"/> over a pinned <see cref="ITtsRunner"/>: builds each job exactly as
/// <see cref="SpeechService"/> does and runs it on the engine's backend under the lease's call lock.</summary>
internal sealed class SynthesizerLease : AudioRunnerLease, ISynthesizerLease
{
    private readonly ITtsRunner _runner;
    private readonly IBackend _backend;
    private readonly string? _weightsVoice;

    /// <summary>Pins <paramref name="key"/> in the engine's TTS cache. <paramref name="weightsVoice"/> is the voice the
    /// loaded weights ARE for a descriptor whose voice selects them (Piper), else null.</summary>
    internal SynthesizerLease(AudioRuntime runtime, string key, ITtsRunner runner, IBackend backend, string? weightsVoice)
        : base(runtime, runtime.Tts, key)
    {
        _runner = runner;
        _backend = backend;
        _weightsVoice = weightsVoice;
        SampleRate = runner.SampleRate;
    }

    /// <inheritdoc/>
    public int SampleRate { get; }

    /// <inheritdoc/>
    public float[] Synthesize(string text, SpeechRequest options)
    {
        ThrowIfClosed();
        if (string.IsNullOrWhiteSpace(text))
        {
            throw new ArgumentException("No text supplied to synthesize.", nameof(text));
        }
        ArgumentNullException.ThrowIfNull(options);
        // The service would load another runner for this voice; this lease holds one set of weights.
        if (_weightsVoice is not null && SpeechService.IsNamedVoice(options.Voice)
            && !string.Equals(options.Voice, _weightsVoice, StringComparison.Ordinal))
        {
            throw new ArgumentException($"This lease holds the '{_weightsVoice}' voice; open a lease for "
                + $"'{options.Voice}' to speak with it.", nameof(options));
        }
        (float[]? referenceMono, string? referenceWavPath) = SpeechService.MaterializeReference(options.Reference);
        try
        {
            TtsJob job = SpeechService.BuildJob(text, options, referenceMono, referenceWavPath);
            float[] samples;
            lock (CallLock)
            {
                ThrowIfClosed();
                samples = _runner.Synthesize(_backend, job);
            }
            if (samples is null || samples.Length == 0)
            {
                throw new InvalidOperationException("The text-to-speech model produced no audio.");
            }
            return samples;
        }
        finally
        {
            SpeechService.DeleteTempReference(referenceWavPath);
        }
    }
}
