using HartsyInference.Audio.Models.Wake;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.Audio.Pipelines;

/// <summary>One wake detection: which word fired, its smoothed score, and where in the stream.</summary>
public readonly record struct WakeDetection(string Word, float Score, long SampleIndex);

/// <summary>Per-word detection tuning. Thresholds are per-head because head families disagree — openWakeWord's
/// are trained to fire at 0.5, while microWakeWord-style models want ~0.97 over a smoothed window.</summary>
public sealed record WakeWordSettings
{
    /// <summary>Smoothed score at or above which the word fires.</summary>
    public float Threshold { get; init; } = 0.5f;
    /// <summary>Score steps (80 ms each) averaged before thresholding. 1 disables smoothing.</summary>
    public int SmoothingWindow { get; init; } = 3;
    /// <summary>Silence enforced after a detection before the same word can fire again.</summary>
    public double RefractorySeconds { get; init; } = 2.0;
}

/// <summary>Streaming wake-word detection: PCM in, <see cref="WakeDetection"/> out, scoring every 80 ms.
///
/// <para>This reproduces openWakeWord's streaming contract rather than inventing one, because the mel model's
/// dB floor is a reduction over whatever buffer it is given: each step re-computes the mel of the newest 1280
/// samples <b>plus 480 samples of left context</b> (which is what yields exactly 8 frames), appends those to a
/// mel ring, embeds the newest 76 frames, and scores the newest 16 embeddings. Treating the front-end as a
/// continuous incremental STFT instead would be a different computation from the one the weights were trained
/// and shipped for.</para>
///
/// <para>Unlike upstream, scoring is withheld until 76 real mel frames and 16 real embedding frames have
/// accumulated (~1.3 s) instead of seeding the buffers with random audio. That removes a nondeterministic
/// warm-up from a process that runs for months and cannot produce a detection from seed data.</para>
///
/// <para>Audio is int16-scaled (±32768), 16 kHz mono. Not thread-safe: one instance per device session, driven
/// by the single wake worker thread.</para></summary>
public sealed class WakeDetectionPipeline : IDisposable
{
    /// <summary>Samples per scoring step (80 ms at 16 kHz).</summary>
    public const int ChunkSamples = 1280;
    /// <summary>Samples of previous audio prepended to each mel call.</summary>
    public const int LeftContextSamples = 3 * WakeMelFrontend.HopLength;
    /// <summary>Mel frames produced per step.</summary>
    public const int FramesPerChunk = 8;

    private const int MelRingFrames = 128;      // > 76 + 8, power of two
    private const int FeatureRingFrames = 32;   // > 16 + 1, power of two

    private readonly IBackend _backend;
    private readonly WakeMelFrontend _mel;
    private readonly SpeechEmbeddingModel _embedding;
    private readonly List<HeadState> _heads = [];

    private readonly float[] _pending = new float[ChunkSamples];
    private readonly float[] _history = new float[ChunkSamples + LeftContextSamples];
    private readonly float[] _melScratch = new float[FramesPerChunk * WakeMelFrontend.MelBins];
    private readonly float[] _melRing = new float[MelRingFrames * WakeMelFrontend.MelBins];
    private readonly float[] _melWindow = new float[SpeechEmbeddingModel.WindowFrames * WakeMelFrontend.MelBins];
    private readonly float[] _featureRing = new float[FeatureRingFrames * SpeechEmbeddingModel.EmbeddingDim];
    private readonly float[] _featureWindow = new float[WakeHead.InputDim];
    private readonly float[] _embedScratch = new float[SpeechEmbeddingModel.EmbeddingDim];

    private int _pendingCount;
    private int _historyCount;
    private long _melFramesWritten;
    private long _featureFramesWritten;
    private long _samplesConsumed;
    private int _disposed;

    /// <summary>Total samples pushed since construction or the last <see cref="Reset"/>; detection timestamps
    /// are expressed in this frame of reference.</summary>
    public long SamplesConsumed => _samplesConsumed;

    /// <summary>Wake words currently loaded.</summary>
    public IReadOnlyCollection<string> Words => _heads.Select(static h => h.Head.Name).ToArray();

    /// <summary>Takes ownership of <paramref name="mel"/> and <paramref name="embedding"/>; both are disposed with the pipeline.</summary>
    public WakeDetectionPipeline(IBackend backend, WakeMelFrontend mel, SpeechEmbeddingModel embedding)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _mel = mel ?? throw new ArgumentNullException(nameof(mel));
        _embedding = embedding ?? throw new ArgumentNullException(nameof(embedding));
    }

    /// <summary>Adds a wake word. Takes ownership of <paramref name="head"/>. Safe to call between pushes, so
    /// a newly trained word reaches live sessions without restarting them.</summary>
    public void AddWord(WakeHead head, WakeWordSettings? settings = null)
    {
        ArgumentNullException.ThrowIfNull(head);
        if (_heads.Any(h => h.Head.Name == head.Name))
            throw new HartsyInferenceException($"Wake word '{head.Name}' is already loaded.");
        _heads.Add(new HeadState(head, settings ?? new WakeWordSettings()));
    }

    /// <summary>Removes a wake word and disposes its head. Returns false if it was not loaded.</summary>
    public bool RemoveWord(string name)
    {
        int i = _heads.FindIndex(h => h.Head.Name == name);
        if (i < 0) return false;
        _heads[i].Head.Dispose();
        _heads.RemoveAt(i);
        return true;
    }

    /// <summary>Feeds audio and reports any words that fired. <paramref name="detections"/> is cleared first.
    /// Any number of samples may be pushed; work happens per whole 80 ms chunk.</summary>
    public void Push(ReadOnlySpan<float> samples, List<WakeDetection> detections)
    {
        ArgumentNullException.ThrowIfNull(detections);
        detections.Clear();
        if (_heads.Count == 0) return;

        while (!samples.IsEmpty)
        {
            int take = Math.Min(ChunkSamples - _pendingCount, samples.Length);
            samples[..take].CopyTo(_pending.AsSpan(_pendingCount));
            _pendingCount += take;
            samples = samples[take..];

            if (_pendingCount < ChunkSamples) break;
            ProcessChunk(detections);
            _pendingCount = 0;
        }
    }

    /// <summary>Clears all audio and model state. Call after a stream discontinuity — a reconnect or a gap in
    /// sequence numbers — since splicing across a gap feeds the model a transient that never occurred.</summary>
    public void Reset()
    {
        _pendingCount = 0;
        _historyCount = 0;
        _melFramesWritten = 0;
        _featureFramesWritten = 0;
        _samplesConsumed = 0;
        Array.Clear(_history);
        Array.Clear(_melRing);
        Array.Clear(_featureRing);
        foreach (HeadState head in _heads) head.Reset();
    }

    private void ProcessChunk(List<WakeDetection> detections)
    {
        // Slide the 480-sample left context down and append the new chunk.
        if (_historyCount > LeftContextSamples)
        {
            Array.Copy(_history, _historyCount - LeftContextSamples, _history, 0, LeftContextSamples);
            _historyCount = LeftContextSamples;
        }
        Array.Copy(_pending, 0, _history, _historyCount, ChunkSamples);
        _historyCount += ChunkSamples;
        _samplesConsumed += ChunkSamples;

        int frames = _mel.Compute(_backend, _history.AsSpan(0, _historyCount), _melScratch);
        for (int f = 0; f < frames; f++)
        {
            int slot = (int)((_melFramesWritten + f) % MelRingFrames);
            Array.Copy(_melScratch, f * WakeMelFrontend.MelBins,
                _melRing, slot * WakeMelFrontend.MelBins, WakeMelFrontend.MelBins);
        }
        _melFramesWritten += frames;

        if (_melFramesWritten < SpeechEmbeddingModel.WindowFrames) return;

        CopyRingTail(_melRing, MelRingFrames, WakeMelFrontend.MelBins, _melFramesWritten,
            SpeechEmbeddingModel.WindowFrames, _melWindow);
        _embedding.Forward(_backend, _melWindow, _embedScratch);

        int featureSlot = (int)(_featureFramesWritten % FeatureRingFrames);
        Array.Copy(_embedScratch, 0, _featureRing,
            featureSlot * SpeechEmbeddingModel.EmbeddingDim, SpeechEmbeddingModel.EmbeddingDim);
        _featureFramesWritten++;

        if (_featureFramesWritten < WakeHead.ContextFrames) return;

        CopyRingTail(_featureRing, FeatureRingFrames, SpeechEmbeddingModel.EmbeddingDim, _featureFramesWritten,
            WakeHead.ContextFrames, _featureWindow);

        foreach (HeadState head in _heads)
        {
            float smoothed = head.Observe(_backend, _featureWindow);
            if (smoothed < head.Settings.Threshold) continue;
            if (_samplesConsumed < head.RefractoryUntilSample) continue;

            head.RefractoryUntilSample = _samplesConsumed + (long)(head.Settings.RefractorySeconds * 16_000);
            head.ClearHistory();
            detections.Add(new WakeDetection(head.Head.Name, smoothed, _samplesConsumed));
        }
    }

    private static void CopyRingTail(float[] ring, int capacity, int width, long written, int count, float[] destination)
    {
        long first = written - count;
        for (int i = 0; i < count; i++)
        {
            int slot = (int)((first + i) % capacity);
            Array.Copy(ring, slot * width, destination, i * width, width);
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (HeadState head in _heads) head.Head.Dispose();
        _heads.Clear();
        _mel.Dispose();
        _embedding.Dispose();
    }

    private sealed class HeadState(WakeHead head, WakeWordSettings settings)
    {
        private readonly float[] _recent = new float[Math.Max(1, settings.SmoothingWindow)];
        private int _count;

        public WakeHead Head { get; } = head;
        public WakeWordSettings Settings { get; } = settings;
        public long RefractoryUntilSample { get; set; }

        /// <summary>Scores the window and returns the smoothed score. Thresholding a moving average rather than
        /// the raw per-step probability is what keeps a single noisy step from firing.</summary>
        public float Observe(IBackend backend, ReadOnlySpan<float> features)
        {
            float score = Head.Score(backend, features);
            _recent[_count % _recent.Length] = score;
            _count++;
            int filled = Math.Min(_count, _recent.Length);
            float sum = 0f;
            for (int i = 0; i < filled; i++) sum += _recent[i];
            return sum / filled;
        }

        /// <summary>Drops the smoothing history so a fired word does not immediately re-fire on its own tail.</summary>
        public void ClearHistory()
        {
            Array.Clear(_recent);
            _count = 0;
        }

        public void Reset()
        {
            ClearHistory();
            RefractoryUntilSample = 0;
        }
    }
}
