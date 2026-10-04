using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Wav2Vec2Bert;

/// <summary>HuggingFace <c>Wav2Vec2BertModel</c> front end: the real <c>SeamlessM4TFeatureExtractor</c>'s 80-bin
/// Kaldi fbank → per-channel zero-mean/unit-variance normalize → stride-2 frame-pair stacking (giving the model's
/// real 160-dim input) → <c>feature_projection</c> → <see cref="Wav2Vec2BertConformerLayer"/> stack, stopping at
/// <see cref="Wav2Vec2BertConfig.NumLayersToRun"/> layers (an HF <c>output_hidden_states</c> consumer never needs
/// more than its target index, so later layers are never loaded).
/// <para>Generic to the <c>facebook/w2v-bert-2.0</c> checkpoint shape — which caller-chosen layer index and
/// downstream mean/std normalization to apply is specific to each consumer (IndexTTS-2's semantic conditioning
/// reads <c>hidden_states[17]</c> then normalizes by its own <c>wav2vec2bert_stats.pt</c>) and lives outside this
/// class.</para></summary>
public sealed class Wav2Vec2BertExtractor : IDisposable
{
    private const int SampleRate = 16_000;
    private const int NumMelBins = 80;
    private const int Stride = 2;

    private readonly Wav2Vec2BertConfig _cfg;
    private readonly KaldiFbankExtractor _fbank = new(SampleRate, NumMelBins);
    private readonly Wav2Vec2BertConformerLayer[] _layers;
    private Tensor? _projLnW, _projLnB, _projW, _projB;
    private int _disposed;

    public Wav2Vec2BertExtractor(Wav2Vec2BertConfig cfg)
    {
        _cfg = cfg;
        _layers = new Wav2Vec2BertConformerLayer[cfg.NumLayersToRun];
        for (int i = 0; i < _layers.Length; i++) _layers[i] = new Wav2Vec2BertConformerLayer(cfg);
    }

    /// <summary><paramref name="prefix"/> is the checkpoint's own top-level module name (empty string for a
    /// standalone <c>Wav2Vec2BertModel</c> — <c>feature_projection.*</c>/<c>encoder.layers.N.*</c> — or e.g.
    /// <c>"wav2vec2_bert"</c> if embedded under a larger task-head checkpoint).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _projLnW = WhisperOps.EnsureF32(w[$"{p}feature_projection.layer_norm.weight"]);
        _projLnB = WhisperOps.EnsureF32(w[$"{p}feature_projection.layer_norm.bias"]);
        _projW = WhisperOps.EnsureF32(w[$"{p}feature_projection.projection.weight"]);
        _projB = WhisperOps.EnsureF32(w[$"{p}feature_projection.projection.bias"]);
        for (int i = 0; i < _layers.Length; i++)
            _layers[i].LoadWeights(w, $"{p}encoder.layers.{i}");
    }

    /// <summary>Runs 16 kHz mono audio through the front end and the first <see cref="Wav2Vec2BertConfig.NumLayersToRun"/>
    /// Conformer layers, returning the real <c>hidden_states[NumLayersToRun]</c> tensor <c>[1, T, hiddenSize]</c>
    /// (un-normalized — any <c>(x - mean) / std</c> semantic normalization is the caller's own checkpoint data).</summary>
    public Tensor Forward(IBackend backend, ReadOnlySpan<float> audio16k)
    {
        ThrowIfDisposed();
        if (_projW is null) throw new InvalidOperationException("Wav2Vec2BertExtractor weights not loaded.");

        Tensor stacked = ExtractStackedFeatures(audio16k);
        int t = (int)stacked.Shape[1];

        Tensor projNormed = new(stacked.Shape, DType.F32);
        backend.LayerNorm(projNormed, stacked, _projLnW!, _projLnB!, _cfg.LayerNormEps);
        stacked.Dispose();
        Tensor hidden = WhisperOps.ProjectLinear(backend, projNormed, _projW!, _projB, 1, t, _cfg.FeatureProjectionInputDim, _cfg.HiddenSize);
        projNormed.Dispose();

        foreach (Wav2Vec2BertConformerLayer layer in _layers)
        {
            Tensor next = layer.Forward(backend, hidden, t);
            hidden.Dispose();
            hidden = next;
        }
        return hidden;
    }

    /// <summary>Real <c>SeamlessM4TFeatureExtractor.__call__</c>: Kaldi fbank (the extractor's own ×2^15 Kaldi
    /// scaling already matches torchaudio's convention), per-mel-bin zero-mean/unit-variance normalize over time
    /// (sample variance, ddof=1, +1e-7 epsilon), drop a trailing odd frame, then reshape consecutive frame PAIRS
    /// into one <c>numMelBins*stride</c>-wide vector (so the 80-bin fbank becomes this model's real 160-dim
    /// input and the frame rate halves to ~20 ms).</summary>
    private Tensor ExtractStackedFeatures(ReadOnlySpan<float> audio16k)
    {
        float[,] fbank = _fbank.Compute(audio16k);
        int frames = fbank.GetLength(0), bins = fbank.GetLength(1);

        float[,] normed = new float[frames, bins];
        for (int m = 0; m < bins; m++)
        {
            double mean = 0d;
            for (int x = 0; x < frames; x++) mean += fbank[x, m];
            mean /= Math.Max(frames, 1);
            double sumSq = 0d;
            for (int x = 0; x < frames; x++) { double d = fbank[x, m] - mean; sumSq += d * d; }
            double variance = frames > 1 ? sumSq / (frames - 1) : 0d;   // ddof=1, matching torch's default
            float invStd = (float)(1.0 / Math.Sqrt(variance + 1e-7));
            for (int x = 0; x < frames; x++) normed[x, m] = (float)(fbank[x, m] - mean) * invStd;
        }

        int stackedFrames = frames / Stride;
        Tensor stacked = new(new TensorShape(1, Math.Max(stackedFrames, 0), bins * Stride), DType.F32);
        Span<float> d2 = stacked.AsSpan<float>();
        for (int x = 0; x < stackedFrames; x++)
            for (int s = 0; s < Stride; s++)
                for (int m = 0; m < bins; m++)
                    d2[x * bins * Stride + s * bins + m] = normed[x * Stride + s, m];
        return stacked;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] top = [_projLnW, _projLnB, _projW, _projB];
        foreach (Tensor? t in top) if (t is not null) yield return t;
        foreach (Wav2Vec2BertConformerLayer layer in _layers)
            foreach (Tensor t in layer.EnumerateWeights()) yield return t;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(Wav2Vec2BertExtractor));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
