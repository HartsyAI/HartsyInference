using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Pseudo-streaming wrapper around <see cref="WhisperPipeline"/> for live / RealtimeSTT-style use. Audio
/// is pushed in incrementally; every re-decode of the active window is passed through a
/// <see cref="HypothesisBuffer"/> (LocalAgreement-2) so the wrapper emits stable confirmed text plus a live
/// partial, instead of rewriting earlier output.
///
/// <para><b>Not true low-latency streaming.</b> Whisper encodes a full window per step, so first-confirmation
/// latency is on the order of the re-decode cadence plus the encode time (roughly 1 to 1.5 s on small models).
/// The active window is capped (default 28 s, under Whisper's 30 s limit); when it fills, the wrapper cuts the
/// window: it confirms the remaining tail and starts a fresh epoch. A word straddling a cut may be split. This is
/// the standard whisper_streaming tradeoff and is honest about its limits; VAD-driven cuts and timestamp-based
/// trimming are a later refinement.</para></summary>
public sealed class WhisperStreamingPipeline : IDisposable
{
    private const int SampleRate = 16_000;

    private readonly WhisperPipeline _whisper;
    private readonly IBackend _backend;
    private readonly WhisperOptions _options;
    private readonly HypothesisBuffer _hyp = new();
    private readonly List<float> _window = new();
    private readonly StringBuilder _confirmedAll = new();
    private readonly int _maxWindowSamples;
    private readonly int _minStepSamples;
    private int _newSinceProcess;
    private int _disposed;

    /// <param name="whisper">A loaded pipeline. The wrapper does not own it; the caller disposes it.</param>
    /// <param name="backend">Compute device to run each window decode on.</param>
    /// <param name="options">Decode options (language/task); timestamps are ignored by the word-level policy.</param>
    /// <param name="maxWindowSeconds">Active window cap; must stay under Whisper's 30 s. Default 28 s.</param>
    /// <param name="minStepSeconds">Minimum new audio before a re-decode runs (the cadence). Default 1 s.</param>
    public WhisperStreamingPipeline(WhisperPipeline whisper, IBackend backend, WhisperOptions? options = null,
        double maxWindowSeconds = 28.0, double minStepSeconds = 1.0)
    {
        _whisper = whisper ?? throw new ArgumentNullException(nameof(whisper));
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _options = options ?? new WhisperOptions();
        _maxWindowSamples = (int)(Math.Clamp(maxWindowSeconds, 5.0, 29.0) * SampleRate);
        _minStepSamples = (int)(Math.Max(0.1, minStepSeconds) * SampleRate);
    }

    /// <summary>All text confirmed so far across the whole stream.</summary>
    public string ConfirmedText => _confirmedAll.ToString().Trim();

    /// <summary>Appends incoming mono audio, resampling to 16 kHz if needed. Does not run inference.</summary>
    public void PushAudio(ReadOnlySpan<float> mono, int sampleRate)
    {
        ThrowIfDisposed();
        if (mono.Length == 0)
        {
            return;
        }
        if (sampleRate == SampleRate)
        {
            for (int i = 0; i < mono.Length; i++)
            {
                _window.Add(mono[i]);
            }
            _newSinceProcess += mono.Length;
            return;
        }
        Resampler resampler = Resampler.Create(sampleRate, SampleRate);
        float[] resampled = resampler.Resample(mono.ToArray());
        _window.AddRange(resampled);
        _newSinceProcess += resampled.Length;
    }

    /// <summary>Re-decodes the active window if enough new audio has arrived (or <paramref name="force"/>),
    /// advances the LocalAgreement buffer, and returns the words newly confirmed this step plus the current
    /// live partial. Cuts the window to a fresh epoch when it reaches the cap.</summary>
    public StreamingTranscription ProcessAvailable(bool force = false)
    {
        ThrowIfDisposed();
        if (_window.Count == 0 || (!force && _newSinceProcess < _minStepSamples))
        {
            return new StreamingTranscription(string.Empty, JoinWords(_hyp.PendingTail));
        }
        _newSinceProcess = 0;

        string text = _whisper.TranscribeAudio(_backend, _window.ToArray(), SampleRate, _options);
        IReadOnlyList<string> words = SplitWords(text);
        IReadOnlyList<string> confirmed = _hyp.Insert(words);
        AppendConfirmed(confirmed);

        // Window full: confirm the remaining tail and start a fresh epoch so the next decode stays under 30 s.
        if (_window.Count >= _maxWindowSamples)
        {
            AppendConfirmed(_hyp.Flush());
            _window.Clear();
            _hyp.Reset();
        }

        return new StreamingTranscription(JoinWords(confirmed), JoinWords(_hyp.PendingTail));
    }

    /// <summary>Flushes any remaining audio and pending words, returning the complete confirmed transcript.</summary>
    public string Finish()
    {
        ThrowIfDisposed();
        if (_window.Count > 0)
        {
            string text = _whisper.TranscribeAudio(_backend, _window.ToArray(), SampleRate, _options);
            AppendConfirmed(_hyp.Insert(SplitWords(text)));
        }
        AppendConfirmed(_hyp.Flush());
        _window.Clear();
        _hyp.Reset();
        return ConfirmedText;
    }

    private void AppendConfirmed(IReadOnlyList<string> words)
    {
        foreach (string w in words)
        {
            if (_confirmedAll.Length > 0)
            {
                _confirmedAll.Append(' ');
            }
            _confirmedAll.Append(w);
        }
    }

    private static IReadOnlyList<string> SplitWords(string text)
        => string.IsNullOrWhiteSpace(text) ? Array.Empty<string>()
            : text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).ToList();

    private static string JoinWords(IReadOnlyList<string> words) => string.Join(' ', words);

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0)
        {
            throw new ObjectDisposedException(nameof(WhisperStreamingPipeline));
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _window.Clear();
        }
    }
}

/// <summary>One streaming step's output: the words confirmed this step and the current live partial.</summary>
public readonly record struct StreamingTranscription(string ConfirmedDelta, string Partial);
