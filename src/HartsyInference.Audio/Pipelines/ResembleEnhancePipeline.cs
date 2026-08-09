using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.ResembleEnhance;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Resemble-enhance inference pipeline, mirroring upstream <c>Enhancer.forward</c> +
/// <c>inference_chunk</c>: peak-normalize and right-pad the 44.1 kHz input; mel it (preemphasis 0.97, n_fft/win
/// 2048, hop 420, slaney filterbank from the checkpoint, dB floor −80 / headroom 15) and z-normalize with the
/// checkpoint's running stats; λ-blend with the <see cref="ResembleDenoiser"/>'d mel; build the CFM prior
/// <c>ψ0 = τ·ε + (1−τ)·z_scale·IRMAE-encode(original mel)</c>; solve the OT-CFM ODE (midpoint, exponential-decay
/// time mapping) with the <see cref="ResembleWnEstimator"/> velocity net conditioned on the blended mel; unscale,
/// <see cref="ResembleIrmae"/>-decode to 160-channel features; and <see cref="ResembleUnivNet"/>-vocode back to
/// PCM, trimmed to the input length and un-normalized.</summary>
public sealed unsafe class ResembleEnhancePipeline : IDisposable
{
    private const int MelNFft = 2_048;
    private const int MelHop = 420;
    private const int MelBins = MelNFft / 2 + 1;
    private const int InferencePadSamples = 441;
    private const float MelHeadroomDb = 15f;

    private readonly ResembleEnhanceConfig _cfg;
    private readonly ResembleWnEstimator _wn;
    private readonly ResembleIrmae _irmae;
    private readonly ResembleDenoiser? _denoiser;
    private readonly ResembleUnivNet? _vocoder;
    private Tensor? _melFb;
    private float _normMean;
    private float _normStd = 1f;
    private float _magnitudeMin;
    private int _disposed;

    public ResembleEnhancePipeline(ResembleEnhanceConfig cfg, bool withDenoiserAndVocoder = false, int vocoderSeed = 0)
    {
        _cfg = cfg;
        _magnitudeMin = cfg.StftMagnitudeMin;
        _wn = new ResembleWnEstimator(cfg, cfg.NMels);
        _irmae = new ResembleIrmae(cfg);
        if (withDenoiserAndVocoder)
        {
            _denoiser = new ResembleDenoiser();
            _vocoder = new ResembleUnivNet(cfg.NMels + cfg.VocoderExtraDim, seed: vocoderSeed);
        }
    }

    /// <summary>Loads the enhancer state dict with a strict key check: a key any module needs but the checkpoint
    /// lacks throws on read, and any checkpoint key no module (or documented inference-unused buffer) consumed
    /// throws after the load.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        ResembleWeightReader r = new ResembleWeightReader(w);
        _wn.LoadWeights(r);
        _irmae.LoadWeights(r);
        if (_denoiser is not null && _vocoder is not null)
        {
            _denoiser.LoadWeights(r);
            _vocoder.LoadWeights(r);

            // Enhancer-level buffers: the slaney mel filterbank and the mel-normalizer running stats are used
            // directly; the F16 hann window is regenerated in F32; the denoiser's own mel front-end and the
            // persistent `dummy` device marker have no inference role.
            _melFb = r.F32("mel_fn.melspec.mel_scale.fb");
            _magnitudeMin = r.ScalarF32("mel_fn.stft_magnitude_min");
            float mean = r.ScalarF32("normalizer.running_mean_unsafe");
            float variance = r.ScalarF32("normalizer.running_var_unsafe");
            bool started = !float.IsNaN(mean);
            _normMean = started ? mean : 0f;
            _normStd = started ? MathF.Sqrt(variance + 1e-9f) : 1f;
            r.MarkUnused(
                "dummy",
                "mel_fn.melspec.spectrogram.window",
                "denoiser.mel_fn.melspec.mel_scale.fb",
                "denoiser.mel_fn.melspec.spectrogram.window",
                "denoiser.mel_fn.stft_magnitude_min");
        }

        IReadOnlyList<string> extra = r.UnconsumedKeys();
        if (extra.Count > 0)
        {
            throw new InvalidDataException(
                $"Resemble-enhance checkpoint has {extra.Count} unexpected key(s) no module consumed: "
                + string.Join(", ", extra.Take(20)) + (extra.Count > 20 ? ", ..." : string.Empty));
        }
    }

    /// <summary>Full enhancement of a mono 44.1 kHz clip; <paramref name="lambd"/> blends the denoised mel into
    /// the conditioning, <paramref name="tau"/> is the CFM prior temperature. Requires
    /// <c>withDenoiserAndVocoder: true</c>.</summary>
    public float[] Enhance(IBackend backend, float[] noisyPcm44k, float lambd, float tau, int seed = 0)
        => Enhance(backend, noisyPcm44k, lambd, tau, seed, nfe: null, solver: null);

    /// <summary>As above, overriding the CFM solver settings. <paramref name="nfe"/> is the function-evaluation
    /// budget (upstream's Gradio app exposes 1–128, default 64) and <paramref name="solver"/> is one of
    /// <c>euler</c>/<c>midpoint</c>/<c>rk4</c>. Null keeps the configured value.</summary>
    public float[] Enhance(IBackend backend, float[] noisyPcm44k, float lambd, float tau, int seed, int? nfe, string? solver)
    {
        ThrowIfDisposed();
        if (_denoiser is null || _vocoder is null)
        {
            throw new InvalidOperationException("Construct the pipeline with withDenoiserAndVocoder: true to call Enhance.");
        }
        if (noisyPcm44k is null || noisyPcm44k.Length == 0)
        {
            throw new ArgumentException("noisyPcm44k must be non-empty.", nameof(noisyPcm44k));
        }
        if (_melFb is null)
        {
            throw new InvalidOperationException("LoadWeights must run before Enhance.");
        }

        // inference_chunk: peak-normalize, right-pad 441 samples (Enhancer.forward's own re-normalization of the
        // already-unit-peak signal is a 1/(1+1e-7) factor, folded in here).
        int length = noisyPcm44k.Length;
        float absMax = 0f;
        for (int i = 0; i < length; i++)
        {
            float a = MathF.Abs(noisyPcm44k[i]);
            if (a > absMax) absMax = a;
        }
        if (absMax < 1e-7f) absMax = 1e-7f;
        float[] x = new float[length + InferencePadSamples];
        float invPeak = 1f / (absMax + 1e-7f);
        for (int i = 0; i < length; i++)
        {
            x[i] = noisyPcm44k[i] * invPeak;
        }

        Tensor melOriginal = ComputeNormalizedMel(x);

        // λ-blend the denoised mel into the conditioning (λ=0 short-circuits like upstream).
        Tensor cond;
        if (lambd == 0f)
        {
            cond = CloneTensor(melOriginal);
        }
        else
        {
            float[] denoised = _denoiser.Denoise(backend, x);
            Tensor melDenoised = ComputeNormalizedMel(denoised);
            cond = new Tensor(melOriginal.Shape, DType.F32);
            float* cp = (float*)cond.DataPointer;
            float* op = (float*)melOriginal.DataPointer;
            float* dp = (float*)melDenoised.DataPointer;
            long n = melOriginal.ElementCount;
            for (long i = 0; i < n; i++)
            {
                cp[i] = lambd * dp[i] + (1f - lambd) * op[i];
            }
            melDenoised.Dispose();
        }

        // ψ0 = τ·ε + (1−τ)·z_scale·encode(original mel).
        Tensor z0 = _irmae.Encode(backend, melOriginal);
        melOriginal.Dispose();
        float* zp = (float*)z0.DataPointer;
        uint rng = DeterministicRng.Seed(seed);
        long zN = z0.ElementCount;
        for (long i = 0; i < zN; i++)
        {
            zp[i] = tau * DeterministicRng.NextGaussian(ref rng) + (1f - tau) * (_cfg.LatentScale * zp[i]);
        }

        Tensor psi1 = Solve(backend, cond, z0, nfe, solver);
        cond.Dispose();

        // Unscale the latent, decode to 160-channel features, vocode.
        float* pp = (float*)psi1.DataPointer;
        float invScale = 1f / _cfg.LatentScale;
        for (long i = 0; i < psi1.ElementCount; i++)
        {
            pp[i] *= invScale;
        }
        Tensor decoded = _irmae.Decode(backend, psi1);
        psi1.Dispose();
        float[] wav = _vocoder.Forward(backend, decoded);
        decoded.Dispose();

        // Trim to the input length and undo the peak normalization.
        float[] outPcm = new float[length];
        int copy = Math.Min(wav.Length, length);
        for (int i = 0; i < copy; i++)
        {
            outPcm[i] = wav[i] * absMax;
        }
        return outPcm;
    }

    /// <summary>LCFM enhancement of an already-normalized conditioning mel <c>[1, mels, T]</c> with a pure
    /// Gaussian prior (upstream's <c>force_gaussian_prior</c> branch) — the synthetic-weights test surface.
    /// Returns the decoded <c>[1, mels+extra, T]</c> features.</summary>
    public Tensor EnhanceMel(IBackend backend, Tensor condMel, int seed = 0)
    {
        ThrowIfDisposed();
        int t = (int)condMel.Shape[2];
        Tensor psi0 = new(new TensorShape(1, _cfg.LatentDim, t), DType.F32);
        float* p0 = (float*)psi0.DataPointer;
        uint rng = DeterministicRng.Seed(seed);
        for (long i = 0; i < psi0.ElementCount; i++)
        {
            p0[i] = DeterministicRng.NextGaussian(ref rng);
        }
        Tensor psi1 = Solve(backend, condMel, psi0, null, null);
        float* pp = (float*)psi1.DataPointer;
        float invScale = 1f / _cfg.LatentScale;
        for (long i = 0; i < psi1.ElementCount; i++)
        {
            pp[i] *= invScale;
        }
        Tensor decoded = _irmae.Decode(backend, psi1);
        psi1.Dispose();
        return decoded;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _wn.EnumerateWeights()) yield return t;
        foreach (Tensor t in _irmae.EnumerateWeights()) yield return t;
        if (_denoiser is not null) foreach (Tensor t in _denoiser.EnumerateWeights()) yield return t;
        if (_vocoder is not null) foreach (Tensor t in _vocoder.EnumerateWeights()) yield return t;
        if (_melFb is not null) yield return _melFb;
    }

    /// <summary>Integrates the OT-CFM ODE from <paramref name="psi0"/> (consumed and returned as ψ1) with
    /// upstream's solver: the time grid maps <c>u ∈ linspace(0,1)</c> through <c>(a^u − 1)/(a − 1)</c> where
    /// <c>a</c> solves <c>h(1/divisor) = 0.5</c> (steps cluster near t=0), stepped by euler/midpoint/rk4.</summary>
    private Tensor Solve(IBackend backend, Tensor cond, Tensor psi0, int? nfeOverride, string? solverOverride)
    {
        int nfe = nfeOverride is > 0 ? nfeOverride.Value : _cfg.Nfe;
        string solver = string.IsNullOrWhiteSpace(solverOverride) ? _cfg.Solver : solverOverride.Trim().ToLowerInvariant();
        int steps = solver == "midpoint" ? nfe / 2 : solver == "rk4" ? nfe / 4 : nfe;
        if (steps < 1) steps = 1;
        float[] ts = TimeGrid(steps, _cfg.TimeMappingDivisor);

        Tensor x = psi0;
        float* xp = (float*)x.DataPointer;
        long n = x.ElementCount;
        for (int step = 0; step < steps; step++)
        {
            float t0 = ts[step];
            float dt = ts[step + 1] - t0;

            if (solver == "rk4")
            {
                Tensor k1 = _wn.Estimate(backend, x, cond, t0);
                Tensor x2 = AxpyClone(x, k1, dt * 0.5f);
                Tensor k2 = _wn.Estimate(backend, x2, cond, t0 + dt * 0.5f);
                x2.Dispose();
                Tensor x3 = AxpyClone(x, k2, dt * 0.5f);
                Tensor k3 = _wn.Estimate(backend, x3, cond, t0 + dt * 0.5f);
                x3.Dispose();
                Tensor x4 = AxpyClone(x, k3, dt);
                Tensor k4 = _wn.Estimate(backend, x4, cond, t0 + dt);
                x4.Dispose();
                float* p1 = (float*)k1.DataPointer;
                float* p2 = (float*)k2.DataPointer;
                float* p3 = (float*)k3.DataPointer;
                float* p4 = (float*)k4.DataPointer;
                float c = dt / 6f;
                for (long i = 0; i < n; i++)
                {
                    xp[i] += c * (p1[i] + 2f * p2[i] + 2f * p3[i] + p4[i]);
                }
                k1.Dispose(); k2.Dispose(); k3.Dispose(); k4.Dispose();
            }
            else if (solver == "midpoint")
            {
                Tensor k1 = _wn.Estimate(backend, x, cond, t0);
                Tensor xm = AxpyClone(x, k1, dt * 0.5f);
                k1.Dispose();
                Tensor k2 = _wn.Estimate(backend, xm, cond, t0 + dt * 0.5f);
                xm.Dispose();
                float* p2 = (float*)k2.DataPointer;
                for (long i = 0; i < n; i++)
                {
                    xp[i] += dt * p2[i];
                }
                k2.Dispose();
            }
            else
            {
                Tensor v = _wn.Estimate(backend, x, cond, t0);
                float* vp = (float*)v.DataPointer;
                for (long i = 0; i < n; i++)
                {
                    xp[i] += dt * vp[i];
                }
                v.Dispose();
            }
        }
        return x;
    }

    /// <summary>Upstream <c>Solver.exponential_decay_mapping</c>: solves <c>(a^(1/divisor) − 1)/(a − 1) = 0.5</c>
    /// for <c>a</c> by bisection (the ratio is monotonically decreasing on (0, 1)), then maps the uniform grid
    /// through <c>(a^u − 1)/(a − 1)</c>.</summary>
    private static float[] TimeGrid(int steps, int divisor)
    {
        double exponent = 1.0 / divisor;
        double lo = 1e-12, hi = 1.0 - 1e-12;
        for (int i = 0; i < 200; i++)
        {
            double mid = 0.5 * (lo + hi);
            double value = (Math.Pow(mid, exponent) - 1.0) / (mid - 1.0);
            if (value > 0.5) lo = mid;
            else hi = mid;
        }
        double a = 0.5 * (lo + hi);

        float[] ts = new float[steps + 1];
        for (int i = 0; i <= steps; i++)
        {
            double u = (double)i / steps;
            ts[i] = (float)((Math.Pow(a, u) - 1.0) / (a - 1.0));
        }
        return ts;
    }

    private static Tensor AxpyClone(Tensor x, Tensor v, float a)
    {
        Tensor o = new(x.Shape, DType.F32);
        float* op = (float*)o.DataPointer;
        float* xp = (float*)x.DataPointer;
        float* vp = (float*)v.DataPointer;
        long n = x.ElementCount;
        for (long i = 0; i < n; i++)
        {
            op[i] = xp[i] + a * vp[i];
        }
        return o;
    }

    private static Tensor CloneTensor(Tensor x)
    {
        Tensor o = new(x.Shape, DType.F32);
        long bytes = x.ElementCount * sizeof(float);
        Buffer.MemoryCopy((void*)x.DataPointer, (void*)o.DataPointer, bytes, bytes);
        return o;
    }

    /// <summary>Upstream <c>MelSpectrogram.forward</c> + the enhancer's <c>Normalizer</c>: preemphasis 0.97 →
    /// magnitude STFT (n_fft/win 2048, hop 420, periodic hann, center with CONSTANT zero padding) → checkpoint
    /// slaney filterbank → dB with the <c>stft_magnitude_min</c> floor → <c>(dB − min_db)/(−min_db + 15)</c> →
    /// z-normalize with the checkpoint running stats. The last STFT frame is dropped (<c>to_mel</c>).</summary>
    private Tensor ComputeNormalizedMel(float[] wav)
    {
        int t = wav.Length;
        int frames = t / MelHop;    // 1 + t/hop STFT frames, minus the dropped last frame.
        float[] pre = new float[t];
        pre[0] = wav[0];
        for (int i = 1; i < t; i++)
        {
            pre[i] = wav[i] - _cfg.Preemphasis * wav[i - 1];
        }

        float[] window = HannWindow.Get(MelNFft);
        float[] frame = new float[MelNFft];
        float[] re = new float[MelBins];
        float[] im = new float[MelBins];
        float[] mag = new float[MelBins];
        int half = MelNFft / 2;

        Tensor mel = new(new TensorShape(1, _cfg.NMels, frames), DType.F32);
        float* mp = (float*)mel.DataPointer;
        float* fb = (float*)_melFb!.DataPointer;    // [MelBins, NMels]
        float minLevelDb = 20f * MathF.Log10(_magnitudeMin);
        float invRange = 1f / (-minLevelDb + MelHeadroomDb);
        float invStd = 1f / _normStd;

        for (int f = 0; f < frames; f++)
        {
            int center = f * MelHop;
            for (int i = 0; i < MelNFft; i++)
            {
                int idx = center - half + i;
                float s = idx >= 0 && idx < t ? pre[idx] : 0f;    // pad_mode="constant"
                frame[i] = s * window[i];
            }
            Fft.RealTransform(frame, re, im, MelNFft);
            for (int k = 0; k < MelBins; k++)
            {
                mag[k] = MathF.Sqrt(re[k] * re[k] + im[k] * im[k]);
            }
            for (int m = 0; m < _cfg.NMels; m++)
            {
                float acc = 0f;
                for (int k = 0; k < MelBins; k++)
                {
                    acc += fb[(long)k * _cfg.NMels + m] * mag[k];
                }
                if (acc < _magnitudeMin) acc = _magnitudeMin;
                float db = 20f * MathF.Log10(acc);
                float normed = (db - minLevelDb) * invRange;
                mp[(long)m * frames + f] = (normed - _normMean) * invStd;
            }
        }
        return mel;
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
