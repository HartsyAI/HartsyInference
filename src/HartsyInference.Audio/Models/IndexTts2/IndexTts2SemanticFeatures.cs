using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Models.Wav2Vec2Bert;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.IndexTts2;

/// <summary>IndexTTS-2's semantic-conditioning front end: real <c>get_emb</c>
/// (<c>wav2vec2_model(...).hidden_states[17]</c>, normalized by the real <c>wav2vec2bert_stats.pt</c>'s
/// per-channel mean/std). Used for BOTH speaker conditioning (<c>spk_cond_emb</c>) and emotion conditioning
/// (<c>emo_cond_emb</c>) — the same extractor run on two different reference-audio clips, confirmed from the
/// real <c>infer_v2_5.py</c>: <c>get_emb</c> is called identically for both. Owns the choice of hidden layer
/// (17 of 24) and the normalization — <see cref="Wav2Vec2BertExtractor"/> itself stays generic to the
/// checkpoint architecture and knows neither.</summary>
public sealed unsafe class IndexTts2SemanticFeatures : IDisposable
{
    private const int HiddenLayer = 17;

    private readonly Wav2Vec2BertExtractor _extractor = new(Wav2Vec2BertConfig.V2(HiddenLayer));
    private Tensor? _mean, _invStd;
    private int _disposed;

    /// <param name="extractorPrefix">The real <c>Wav2Vec2BertModel</c>'s own top-level prefix (empty string for
    /// a standalone <c>facebook/w2v-bert-2.0</c> checkpoint).</param>
    /// <param name="mean">Real <c>wav2vec2bert_stats.pt</c>'s <c>mean</c> tensor, <c>[1024]</c>.</param>
    /// <param name="variance">Real <c>wav2vec2bert_stats.pt</c>'s <c>var</c> tensor, <c>[1024]</c> —
    /// <c>std = sqrt(var)</c> is computed once at load, matching the real
    /// <c>self.semantic_std = torch.sqrt(stat_mean_var["var"])</c>.</param>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string extractorPrefix, Tensor mean, Tensor variance)
    {
        _extractor.LoadWeights(w, extractorPrefix);
        _mean = WhisperOps.EnsureF32(mean);
        Tensor varF32 = WhisperOps.EnsureF32(variance);
        Tensor invStd = new(varF32.Shape, DType.F32);
        float* vp = (float*)varF32.DataPointer;
        float* ip = (float*)invStd.DataPointer;
        long n = varF32.ElementCount;
        for (long i = 0; i < n; i++) ip[i] = 1f / MathF.Sqrt(vp[i]);
        if (!ReferenceEquals(varF32, variance)) varF32.Dispose();
        _invStd = invStd;
    }

    /// <summary>Runs 16 kHz mono audio through the w2v-bert front end and returns the normalized
    /// <c>[1, T, 1024]</c> feature — <c>(hidden_states[17] - mean) / std</c>.</summary>
    public Tensor Forward(IBackend backend, ReadOnlySpan<float> audio16k)
    {
        ThrowIfDisposed();
        Tensor raw = _extractor.Forward(backend, audio16k);
        int t = (int)raw.Shape[1], hidden = (int)raw.Shape[2];

        Tensor normed = new(raw.Shape, DType.F32);
        float* rp = (float*)raw.DataPointer;
        float* np_ = (float*)normed.DataPointer;
        float* mp = (float*)_mean!.DataPointer;
        float* sp = (float*)_invStd!.DataPointer;
        for (int ti = 0; ti < t; ti++)
        {
            float* row = rp + (long)ti * hidden;
            float* outRow = np_ + (long)ti * hidden;
            for (int c = 0; c < hidden; c++) outRow[c] = (row[c] - mp[c]) * sp[c];
        }
        raw.Dispose();
        return normed;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _extractor.EnumerateWeights()) yield return t;
        if (_mean is not null) yield return _mean;
        if (_invStd is not null) yield return _invStd;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(IndexTts2SemanticFeatures));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _extractor.Dispose();
        _mean?.Dispose();
        _invStd?.Dispose();
        GC.SuppressFinalize(this);
    }
}
