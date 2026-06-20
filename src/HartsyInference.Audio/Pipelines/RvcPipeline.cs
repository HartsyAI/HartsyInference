using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Rvc;
using HartsyInference.Audio.Models.Vits;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>RVC v2 voice-conversion pipeline: HuBERT/ContentVec content features + per-frame F0 → content+pitch
/// encoder → prior sample → reused g-conditioned <see cref="VitsFlow"/> → NSF-HiFi-GAN decoder (the reused
/// <see cref="VitsHiFiGan"/> with an <see cref="NsfVocoderDsp">NSF harmonic source</see> injected per stage).
/// Content features + F0 are caller-supplied (HuBERT is built; RMVPE pitch extraction is staged).</summary>
public sealed unsafe class RvcPipeline : IDisposable
{
    private readonly RvcConfig _cfg;
    private readonly RvcTextEncoder _enc;
    private readonly VitsFlow _flow;
    private readonly VitsHiFiGan _dec;
    private Tensor? _embG, _mSourceW, _mSourceB;
    private int _disposed;

    public RvcPipeline(RvcConfig cfg)
    {
        _cfg = cfg;
        _enc = new RvcTextEncoder(cfg);
        _flow = new VitsFlow(cfg.Core);
        _dec = new VitsHiFiGan(cfg.Core);
    }

    public int SampleRate => _cfg.Core.SampleRate;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _enc.LoadWeights(w);
        _flow.LoadWeights(w);
        _dec.LoadWeights(w);
        _embG = WhisperOps.EnsureF32(w["emb_g.weight"]);
        _mSourceW = WhisperOps.EnsureF32(w["dec.m_source.l_linear.weight"]);
        _mSourceB = WhisperOps.EnsureF32(w["dec.m_source.l_linear.bias"]);
    }

    /// <summary>Converts content features <c>[1, 768, T]</c> + per-frame F0 (Hz, length T) into audio in the
    /// target speaker <paramref name="sid"/>'s voice.</summary>
    public float[] Convert(IBackend backend, Tensor content, ReadOnlySpan<float> f0, int sid, int seed = 0)
    {
        ThrowIfDisposed();
        int t = f0.Length, gin = _cfg.Core.GinChannels, inter = _cfg.Core.InterChannels;
        int upp = _cfg.Core.HopLength;     // ∏ upsample_rates == hop

        int[] pitchBins = RvcPitch.CoarseQuantize(f0, _cfg.F0Min, _cfg.F0Max);
        (Tensor hidden, Tensor mP, Tensor logsP) = _enc.Forward(backend, content, pitchBins, t);
        hidden.Dispose();

        // z_p = m_p + exp(logs_p) * randn * 0.66666.
        uint rng = DeterministicRng.Seed(seed);
        Tensor zP = new(new TensorShape(1, inter, t), DType.F32);
        float* zp = (float*)zP.DataPointer; float* mp = (float*)mP.DataPointer; float* lp = (float*)logsP.DataPointer;
        for (long n = 0; n < (long)inter * t; n++) zp[n] = mp[n] + MathF.Exp(lp[n]) * DeterministicRng.NextGaussian(ref rng) * 0.66666f;
        mP.Dispose(); logsP.Dispose();

        // Speaker embedding g [1, gin, 1].
        Tensor g = new(new TensorShape(1, gin, 1), DType.F32);
        Buffer.MemoryCopy((float*)_embG!.DataPointer + (long)sid * gin, (void*)g.DataPointer, gin * 4, gin * 4);

        Tensor z = _flow.Reverse(backend, zP, t, g); zP.Dispose();

        // NSF harmonic source from F0 (reuses NsfVocoderDsp) → [1, 1, T*upp].
        Tensor f0T = new(new TensorShape(1, 1, t), DType.F32);
        f0.CopyTo(new Span<float>((void*)f0T.DataPointer, t));
        float[] src = NsfVocoderDsp.GenerateHarmonicSource(f0T, upp, _cfg.Core.SampleRate, 1, _mSourceW!, _mSourceB!, voicedThreshold: 0f);
        f0T.Dispose();
        Tensor harSource = new(new TensorShape(1, 1, src.Length), DType.F32);
        src.AsSpan().CopyTo(new Span<float>((void*)harSource.DataPointer, src.Length));

        float[] audio = _dec.Forward(backend, z, t, g, harSource);
        z.Dispose(); g.Dispose(); harSource.Dispose();
        return audio;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _enc.EnumerateWeights()) yield return t;
        foreach (Tensor t in _flow.EnumerateWeights()) yield return t;
        foreach (Tensor t in _dec.EnumerateWeights()) yield return t;
        if (_embG is not null) yield return _embG;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(RvcPipeline));
    }
}
