using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Models.ResembleEnhance;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Resemble-enhance LCFM enhancer: a conditioning mel → latent CFM (the <see cref="ResembleWnEstimator"/>
/// velocity net solved by the reused OT-CFM <see cref="ConditionalCfm"/>, CFG off) → <see cref="ResembleIrmaeDecoder"/>
/// → enhanced mel. The 2D-STFT denoiser (mel pre-conditioner) and the UnivNet vocoder (mel → 44.1 kHz waveform)
/// are staged; this pipeline produces the enhanced mel that those wrap.</summary>
public sealed unsafe class ResembleEnhancePipeline : IDisposable
{
    private readonly ResembleEnhanceConfig _cfg;
    private readonly ResembleWnEstimator _wn;
    private readonly ConditionalCfm _cfm;
    private readonly ResembleIrmaeDecoder _ae;
    private int _disposed;

    public ResembleEnhancePipeline(ResembleEnhanceConfig cfg)
    {
        _cfg = cfg;
        _wn = new ResembleWnEstimator(cfg, cfg.NMels);
        _cfm = new ConditionalCfm(_wn, cfg.LatentDim);
        _ae = new ResembleIrmaeDecoder(cfg);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _wn.LoadWeights(w);
        _ae.LoadWeights(w);
    }

    /// <summary>Enhances a conditioning mel <c>[1, mels, T]</c> → enhanced mel <c>[1, mels, T]</c>: solves the
    /// latent CFM conditioned on the mel, unscales, and IRMAE-decodes.</summary>
    public Tensor EnhanceMel(IBackend backend, Tensor condMel, int t, int seed = 0)
    {
        ThrowIfDisposed();
        int steps = _cfg.Solver == "midpoint" ? _cfg.Nfe / 2 : _cfg.Solver == "rk4" ? _cfg.Nfe / 4 : _cfg.Nfe;
        // Solver conditioning: mel via `cond` (the WN uses it); mu/spk are dummies (unused by the estimator).
        Tensor mu = new(new TensorShape(1, _cfg.LatentDim, t), DType.F32);
        Tensor spk = new(new TensorShape(1, _cfg.LatentDim, 1), DType.F32);
        Tensor z = _cfm.Solve(backend, mu, spk, condMel, steps, cfgRate: 0f, seed);
        mu.Dispose(); spk.Dispose();

        // Unscale latent then decode to mel.
        float* zp = (float*)z.DataPointer;
        for (long n = 0; n < z.ElementCount; n++) zp[n] *= _cfg.LatentScale;
        Tensor mel = _ae.Decode(backend, z, t);
        z.Dispose();
        return mel;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _wn.EnumerateWeights()) yield return t;
        foreach (Tensor t in _ae.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(ResembleEnhancePipeline));
    }
}
