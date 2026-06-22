using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Models.ResembleEnhance;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Resemble-enhance LCFM enhancer: a conditioning mel → latent CFM (the <see cref="ResembleWnEstimator"/>
/// velocity net solved by the reused OT-CFM <see cref="ConditionalCfm"/>, CFG off) → <see cref="ResembleIrmaeDecoder"/>
/// → enhanced mel. The 2D-STFT denoiser (mel pre-conditioner) and the UnivNet vocoder (mel → 44.1 kHz waveform)
/// are staged; this pipeline produces the enhanced mel that those wrap.</summary>
public sealed unsafe class ResembleEnhancePipeline : IDisposable
{
    private const float MelDbFloor = -80f;
    private const float MelDbHeadroom = 15f;

    private readonly ResembleEnhanceConfig _cfg;
    private readonly ResembleWnEstimator _wn;
    private readonly ConditionalCfm _cfm;
    private readonly ResembleIrmaeDecoder _ae;
    private readonly ResembleDenoiser? _denoiser;
    private readonly ResembleUnivNet? _vocoder;
    private readonly MelSpectrogramExtractor _melExtractor;
    private int _disposed;

    public ResembleEnhancePipeline(ResembleEnhanceConfig cfg, bool withDenoiserAndVocoder = false, int vocoderSeed = 0)
    {
        _cfg = cfg;
        _wn = new ResembleWnEstimator(cfg, cfg.NMels);
        _cfm = new ConditionalCfm(_wn, cfg.LatentDim);
        _ae = new ResembleIrmaeDecoder(cfg);
        if (withDenoiserAndVocoder)
        {
            _denoiser = new ResembleDenoiser(cfg.NormEps);
            _vocoder = new ResembleUnivNet(cfg.NMels, seed: vocoderSeed);
        }
        // Enhancer mel front-end: n_fft 2048, win 2048, hop 420, 128 mels, fmax 22050, magnitude, slaney norm.
        _melExtractor = new MelSpectrogramExtractor(new MelSpectrogramExtractor.Config(
            SampleRate: cfg.SampleRate, NFft: cfg.NFft, WinLength: cfg.NFft, HopLength: cfg.HopLength,
            NMels: cfg.NMels, Fmin: 0.0, Fmax: cfg.SampleRate / 2.0,
            Norm: MelSpectrogramExtractor.Normalization.None, DropLastStftFrame: false,
            LogBase: MelSpectrogramExtractor.LogBase.Natural, LogFloor: 1e-5f,
            DynamicRangeDb: 0f, NormOffset: 0f, NormScale: 1f, PowerSpectrum: false));
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _wn.LoadWeights(w);
        _ae.LoadWeights(w);
        _denoiser?.LoadWeights(w);
        _vocoder?.LoadWeights(w);
    }

    /// <summary>Full enhancement: denoise the noisy PCM, mel it, blend <c>λ·denoised + (1−λ)·noisy</c> mels,
    /// run the latent CFM enhancer, then vocode to a 44.1 kHz waveform. Requires the pipeline to have been
    /// constructed with <c>withDenoiserAndVocoder: true</c>.</summary>
    public float[] Enhance(IBackend backend, float[] noisyPcm44k, float lambd, float tau, int seed = 0)
    {
        ThrowIfDisposed();
        if (_denoiser is null || _vocoder is null)
            throw new InvalidOperationException("Construct the pipeline with withDenoiserAndVocoder: true to call Enhance.");

        // Stage 1: denoiser → cleaned PCM, then mel both clean and noisy paths.
        float[] denoised = _denoiser.Denoise(backend, noisyPcm44k);
        Tensor noisyMel = ComputeMel(noisyPcm44k);
        Tensor cleanMel = ComputeMel(denoised);
        int t = (int)Math.Min(noisyMel.Shape[2], cleanMel.Shape[2]);

        // Stage 2: λ-blend the conditioning mel.
        Tensor condMel = new(new TensorShape(1, _cfg.NMels, t), DType.F32);
        float* cp = (float*)condMel.DataPointer;
        float* np = (float*)noisyMel.DataPointer;
        float* dp = (float*)cleanMel.DataPointer;
        int nt = (int)noisyMel.Shape[2], dt = (int)cleanMel.Shape[2];
        for (int c = 0; c < _cfg.NMels; c++)
            for (int j = 0; j < t; j++)
                cp[(long)c * t + j] = lambd * dp[(long)c * dt + j] + (1f - lambd) * np[(long)c * nt + j];
        noisyMel.Dispose();
        cleanMel.Dispose();

        // Stage 3: LCFM enhancer → enhanced mel (midpoint solver, τ-scaled prior).
        Tensor enhancedMel = EnhanceMelMidpoint(backend, condMel, t, tau, seed);
        condMel.Dispose();

        // Stage 4: vocode.
        float[] wav = _vocoder.Forward(backend, enhancedMel);
        enhancedMel.Dispose();
        return wav;
    }

    /// <summary>LCFM enhance using a self-contained midpoint/rk4 OT-CFM solver with exponential time mapping
    /// (divisor 4) and a τ-scaled Gaussian prior — the resemble-enhance defaults, kept here so the shared
    /// Euler-only <see cref="ConditionalCfm"/> stays untouched.</summary>
    public Tensor EnhanceMelMidpoint(IBackend backend, Tensor condMel, int t, float tau, int seed = 0)
    {
        ThrowIfDisposed();
        int steps = _cfg.Solver == "midpoint" ? _cfg.Nfe / 2 : _cfg.Solver == "rk4" ? _cfg.Nfe / 4 : _cfg.Nfe;
        if (steps < 1) steps = 1;

        Tensor mu = new(new TensorShape(1, _cfg.LatentDim, t), DType.F32);
        Tensor spk = new(new TensorShape(1, _cfg.LatentDim, 1), DType.F32);
        Tensor z = SolveCfm(backend, condMel, mu, spk, steps, tau, seed);
        mu.Dispose(); spk.Dispose();

        float* zp = (float*)z.DataPointer;
        for (long n = 0; n < z.ElementCount; n++) zp[n] *= _cfg.LatentScale;
        Tensor mel = _ae.Decode(backend, z, t);
        z.Dispose();
        return mel;
    }

    /// <summary>OT-CFM ODE integration with euler/midpoint/rk4 step and exponential time mapping. The flow
    /// time-grid is mapped through <c>(exp(divisor·u) − 1) / (exp(divisor) − 1)</c> so steps cluster near t=0
    /// (resemble-enhance's <c>get_time_steps</c>).</summary>
    private Tensor SolveCfm(IBackend backend, Tensor condMel, Tensor mu, Tensor spk, int steps, float tau, int seed)
    {
        int t = (int)mu.Shape[2];
        // τ-scaled Gaussian prior ψ0 = τ·randn.
        Tensor x = new(new TensorShape(1, _cfg.LatentDim, t), DType.F32);
        float* xp = (float*)x.DataPointer;
        uint rng = DeterministicRng.Seed(seed);
        for (long i = 0; i < x.ElementCount; i++) xp[i] = tau * DeterministicRng.NextGaussian(ref rng);

        float divisor = _cfg.TimeMappingDivisor;
        float denom = MathF.Exp(divisor) - 1f;
        long n = x.ElementCount;
        for (int step = 0; step < steps; step++)
        {
            float u0 = (float)step / steps;
            float u1 = (float)(step + 1) / steps;
            float t0 = (MathF.Exp(divisor * u0) - 1f) / denom;
            float t1 = (MathF.Exp(divisor * u1) - 1f) / denom;
            float dt = t1 - t0;

            if (_cfg.Solver == "rk4")
            {
                Tensor k1 = _wn.Estimate(backend, x, mu, t0, spk, condMel);
                Tensor x2 = AxpyClone(x, k1, dt * 0.5f);
                Tensor k2 = _wn.Estimate(backend, x2, mu, t0 + dt * 0.5f, spk, condMel); x2.Dispose();
                Tensor x3 = AxpyClone(x, k2, dt * 0.5f);
                Tensor k3 = _wn.Estimate(backend, x3, mu, t0 + dt * 0.5f, spk, condMel); x3.Dispose();
                Tensor x4 = AxpyClone(x, k3, dt);
                Tensor k4 = _wn.Estimate(backend, x4, mu, t1, spk, condMel); x4.Dispose();
                float* p1 = (float*)k1.DataPointer; float* p2 = (float*)k2.DataPointer;
                float* p3 = (float*)k3.DataPointer; float* p4 = (float*)k4.DataPointer;
                float c = dt / 6f;
                for (long i = 0; i < n; i++) xp[i] += c * (p1[i] + 2f * p2[i] + 2f * p3[i] + p4[i]);
                k1.Dispose(); k2.Dispose(); k3.Dispose(); k4.Dispose();
            }
            else if (_cfg.Solver == "midpoint")
            {
                Tensor k1 = _wn.Estimate(backend, x, mu, t0, spk, condMel);
                Tensor xm = AxpyClone(x, k1, dt * 0.5f); k1.Dispose();
                Tensor k2 = _wn.Estimate(backend, xm, mu, t0 + dt * 0.5f, spk, condMel); xm.Dispose();
                float* p2 = (float*)k2.DataPointer;
                for (long i = 0; i < n; i++) xp[i] += dt * p2[i];
                k2.Dispose();
            }
            else
            {
                Tensor v = _wn.Estimate(backend, x, mu, t0, spk, condMel);
                float* vp = (float*)v.DataPointer;
                for (long i = 0; i < n; i++) xp[i] += dt * vp[i];
                v.Dispose();
            }
        }
        return x;
    }

    private static unsafe Tensor AxpyClone(Tensor x, Tensor v, float a)
    {
        Tensor o = new(x.Shape, DType.F32);
        float* op = (float*)o.DataPointer; float* xp = (float*)x.DataPointer; float* vp = (float*)v.DataPointer;
        long n = x.ElementCount;
        for (long i = 0; i < n; i++) op[i] = xp[i] + a * vp[i];
        return o;
    }

    /// <summary>Mels a PCM clip with the enhancer front-end, then applies the resemble-enhance dB transform
    /// (natural-log power → dB, clamped to a −80 dB floor with 15 dB headroom, normalized into ~[-1, 0]).</summary>
    private Tensor ComputeMel(float[] pcm)
    {
        float[,] mel = _melExtractor.Compute(pcm);
        int mels = mel.GetLength(0), frames = mel.GetLength(1);
        Tensor t = new(new TensorShape(1, mels, frames), DType.F32);
        float* tp = (float*)t.DataPointer;
        // mel holds ln(magnitude); convert to dB and apply the floor + headroom normalization.
        float scale = 20f / MathF.Log(10f);
        for (int m = 0; m < mels; m++)
            for (int f = 0; f < frames; f++)
            {
                float db = mel[m, f] * scale;
                if (db < MelDbFloor) db = MelDbFloor;
                tp[(long)m * frames + f] = (db - MelDbHeadroom) / (-MelDbFloor);
            }
        return t;
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
        if (_denoiser is not null) foreach (Tensor t in _denoiser.EnumerateWeights()) yield return t;
        if (_vocoder is not null) foreach (Tensor t in _vocoder.EnumerateWeights()) yield return t;
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
