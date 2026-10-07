using HartsyInference.Audio.Dsp;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>Optimal-Transport Conditional Flow Matching solver (speech-token-mel → target mel): wraps an <see cref="ICfmEstimator"/> velocity network in a first-order Euler ODE with classifier-free guidance.</summary>
/// <remarks>Mirrors <c>cosyvoice/flow/flow_matching.py:CausalConditionalCFM.solve_euler</c>, integrating from <c>x0 ~ N(0, I)</c> at <c>t=0</c> to the predicted mel at <c>t=1</c>:
///
/// <code>
///   x = x0
///   for each Euler step (t over linspace(0, 1, N+1)):
///     v  = estimator(x, mu, t, spk, cond)
///     if cfg &gt; 0:
///       v0 = estimator(x, 0,  t, 0,   0)        // unconditional
///       v  = (1 + cfg)·v − cfg·v0
///     x  = x + dt·v
///   return x
/// </code>
///
/// <para>Vanilla Euler — no sway-sampling / omega-shift (the mel operating dimension is small). NFE 10,
/// CFG 0.7 by default. Deterministic for a fixed seed.</para></remarks>
public sealed unsafe class ConditionalCfm(ICfmEstimator estimator, int melBins)
{
    private readonly ICfmEstimator _estimator = estimator;
    private readonly int _melBins = melBins;

    /// <summary>Solves the flow ODE to produce a mel <c>[1, melBins, T]</c>.</summary>
    /// <remarks><paramref name="mu"/> is the
    /// token-conditioning mel (<c>[1, melBins, T]</c>); <paramref name="spk"/> is the mel-projected
    /// speaker vector broadcast over time; <paramref name="cond"/> is the reference-mel prefix (zeros
    /// outside the prompt region). Pass <c>cfgRate ≤ 0</c> to disable CFG.
    /// <paramref name="x0Override"/>, when set, replaces the fresh <c>seed</c>-derived draw — the caller owns
    /// it and it is copied, never mutated (this Solve call's Euler loop always mutates its OWN copy in
    /// place). Exists for chunked streaming (<see cref="CosyVoiceFlow.InferenceGrowingWindowed"/>): re-seeding fresh
    /// noise from the same integer seed on every chunk call would give each chunk's target frames a
    /// DIFFERENT random draw than the one the corresponding frames get in a monolithic call (the RNG stream
    /// position depends on how many frames precede it in THIS call, which varies chunk to chunk) — the fix is
    /// to draw noise ONCE for the whole utterance and slice the caller's own absolute-position sub-range.</remarks>
    /// <param name="promptLen">When greater than 0, the leading <paramref name="promptLen"/> frames of <c>x</c>
    /// are clamped to zero before the loop starts and again after every Euler step — IndexTTS-2's S2Mel CFM
    /// keeps the in-context reference-mel prefix's noise channel at exactly zero throughout the trajectory
    /// (confirmed from the real <c>BASECFM.solve_euler</c>: <c>x[..., :prompt_len] = 0</c> both before the loop
    /// and at the end of every iteration). 0 (the default) disables this and matches every existing call site
    /// (CosyVoice has no such clamp).</param>
    public Tensor Solve(IBackend backend, Tensor mu, Tensor spk, Tensor cond,
        int numSteps, float cfgRate, int seed, Tensor? attnMask = null, Tensor? x0Override = null, int promptLen = 0)
    {
        // Derived from `cond` rather than `mu`: both are frame-count-aligned with the solved `x` in every
        // known estimator, but `cond`'s shape convention (`[1, melBins, T]`, channel-first — same as `x`) is the
        // one guaranteed across estimators, unlike `mu`'s (CosyVoice's own token-conditioning mel happens to
        // share that layout, but IndexTTS-2's content conditioning is channel-LAST with a different width at
        // this axis — reading T from `mu` there would silently pick up the content width instead).
        int t = (int)cond.Shape[2];
        Tensor x = x0Override is not null ? CopyTensor(x0Override) : RandNormal(_melBins, t, seed);
        if (promptLen > 0) ZeroPrefix(x, promptLen);

        // t_span = linspace(0, 1, numSteps + 1); uniform dt.
        float dt = 1f / numSteps;
        Tensor? zMu = null, zSpk = null, zCond = null;
        if (cfgRate > 0f)
        {
            zMu = Zeros(mu.Shape);
            zSpk = Zeros(spk.Shape);
            zCond = Zeros(cond.Shape);
        }

        for (int step = 0; step < numSteps; step++)
        {
            float tNow = step * dt;
            Tensor v = _estimator.Estimate(backend, x, mu, tNow, spk, cond, attnMask);
            if (cfgRate > 0f)
            {
                Tensor v0 = _estimator.Estimate(backend, x, zMu!, tNow, zSpk!, zCond!, attnMask);
                CombineCfg(v, v0, cfgRate);     // v ← (1+cfg)·v − cfg·v0 (in place)
                v0.Dispose();
            }
            // x ← x + dt·v.
            float* xp = (float*)x.DataPointer;
            float* vp = (float*)v.DataPointer;
            long n = x.ElementCount;
            for (long i = 0; i < n; i++) xp[i] += dt * vp[i];
            v.Dispose();
            if (promptLen > 0) ZeroPrefix(x, promptLen);
        }
        zMu?.Dispose();
        zSpk?.Dispose();
        zCond?.Dispose();
        return x;
    }

    /// <summary>Zeroes <c>x[:, :, :promptLen]</c> in place (<c>x</c> is <c>[1, channels, T]</c>).</summary>
    private static void ZeroPrefix(Tensor x, int promptLen)
    {
        int channels = (int)x.Shape[1];
        int t = (int)x.Shape[2];
        float* xp = (float*)x.DataPointer;
        for (int c = 0; c < channels; c++)
            for (int j = 0; j < promptLen && j < t; j++)
                xp[(long)c * t + j] = 0f;
    }

    private static void CombineCfg(Tensor cond, Tensor uncond, float cfg)
    {
        float* cp = (float*)cond.DataPointer;
        float* up = (float*)uncond.DataPointer;
        long n = cond.ElementCount;
        float a = 1f + cfg;
        for (long i = 0; i < n; i++) cp[i] = a * cp[i] - cfg * up[i];
    }

    private static Tensor Zeros(TensorShape shape)
    {
        return new Tensor(shape, DType.F32);     // NativeMemory alloc is zero-initialized.
    }

    private static Tensor RandNormal(int channels, int t, int seed)
    {
        Tensor x = new(new TensorShape(1, channels, t), DType.F32);
        float* p = (float*)x.DataPointer;
        uint rng = DeterministicRng.Seed(seed);
        long n = (long)channels * t;
        for (long i = 0; i < n; i++) p[i] = DeterministicRng.NextGaussian(ref rng);
        return x;
    }

    /// <summary>Draws the FULL <c>[1, channels, totalT]</c> Gaussian noise buffer once, for chunked callers to slice contiguous absolute-position sub-ranges from (see <see cref="Solve"/>'s <c>x0Override</c> doc); identical bit-for-bit to what a monolithic <see cref="Solve"/> call over <c>totalT</c> frames would draw internally.</summary>
    public static Tensor DrawFullNoise(int channels, int totalT, int seed) => RandNormal(channels, totalT, seed);

    /// <summary>Returns a fresh copy of the <c>[1, channels, len]</c> slice <c>full[:, :, start..start+len)</c>.</summary>
    public static Tensor SliceNoise(Tensor full, int channels, int fullT, int start, int len)
    {
        if (start < 0 || len < 0 || start + len > fullT)
            throw new ArgumentOutOfRangeException(nameof(start), $"slice [{start},{start + len}) out of range [0,{fullT}).");
        Tensor slice = new(new TensorShape(1, channels, len), DType.F32);
        float* sp = (float*)full.DataPointer;
        float* dp = (float*)slice.DataPointer;
        for (int c = 0; c < channels; c++)
            Buffer.MemoryCopy(sp + (long)c * fullT + start, dp + (long)c * len, (long)len * 4, (long)len * 4);
        return slice;
    }

    private static Tensor CopyTensor(Tensor src)
    {
        Tensor dst = new(src.Shape, DType.F32);
        Buffer.MemoryCopy((void*)src.DataPointer, (void*)dst.DataPointer, src.ElementCount * 4, src.ElementCount * 4);
        return dst;
    }
}
