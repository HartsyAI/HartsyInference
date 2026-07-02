using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.Diamond;

/// <summary>DIAMOND EDM denoiser: wraps <see cref="DiamondInnerModel"/> with Karras preconditioning
/// (<c>c_in/c_out/c_skip/c_noise</c>, with <c>sigma_offset_noise</c> folded into the effective sigma) and the
/// <c>denoise</c> path that also quantizes back to the 256 pixel levels. Mirrors <c>denoiser.py</c>.</summary>
public sealed unsafe class DiamondDenoiser
{
    private readonly DiamondConfig _cfg;
    private readonly DiamondInnerModel _inner;

    public DiamondConfig Config => _cfg;
    public DiamondInnerModel Inner => _inner;

    public DiamondDenoiser(DiamondConfig cfg) { _cfg = cfg; _inner = new DiamondInnerModel(cfg); }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "inner_model") => _inner.LoadWeights(w, prefix);
    public IEnumerable<Tensor> EnumerateWeights() => _inner.EnumerateWeights();

    /// <summary>EDM preconditioning coefficients for a noise level (offset-noise folded in).</summary>
    public (float CIn, float COut, float CSkip, float CNoise) ComputeConditioners(float sigma)
    {
        float sd = _cfg.SigmaData, off = _cfg.SigmaOffsetNoise;
        float s2 = sigma * sigma + off * off;               // effective sigma^2
        float denom = s2 + sd * sd;
        float cSkip = sd * sd / denom;
        float cIn = 1f / MathF.Sqrt(denom);
        float cOut = MathF.Sqrt(s2) * MathF.Sqrt(cSkip);
        float cNoise = MathF.Log(MathF.Sqrt(s2)) / 4f;
        return (cIn, cOut, cSkip, cNoise);
    }

    /// <summary>Denoises <paramref name="noisy"/> <c>[1,3,H,W]</c> at <paramref name="sigma"/> given the context
    /// <paramref name="obs"/> <c>[1, 4·3, H, W]</c> + <paramref name="act"/>. Returns the quantized clean frame in
    /// [-1,1] (unless <paramref name="quantize"/> is false, which returns the raw EDM output).</summary>
    public Tensor Denoise(IBackend backend, Tensor noisy, float sigma, Tensor obs, ReadOnlySpan<int> act, bool quantize = true)
    {
        (float cIn, float cOut, float cSkip, float cNoise) = ComputeConditioners(sigma);
        long n = noisy.ElementCount;
        Tensor rn = new(noisy.Shape, DType.F32);
        float* npn = (float*)noisy.DataPointer, rp = (float*)rn.DataPointer;
        for (long i = 0; i < n; i++) rp[i] = npn[i] * cIn;

        Tensor ro = new(obs.Shape, DType.F32);
        float* op = (float*)obs.DataPointer, rop = (float*)ro.DataPointer;
        float invSd = 1f / _cfg.SigmaData;
        for (long i = 0; i < obs.ElementCount; i++) rop[i] = op[i] * invSd;

        Tensor innerOut = _inner.Forward(backend, rn, cNoise, ro, act);
        rn.Dispose(); ro.Dispose();

        Tensor d = new(noisy.Shape, DType.F32);
        float* ip = (float*)innerOut.DataPointer, dp = (float*)d.DataPointer;
        for (long i = 0; i < n; i++)
        {
            float v = cSkip * npn[i] + cOut * ip[i];
            if (quantize)
            {
                if (v > 1f) v = 1f; else if (v < -1f) v = -1f;
                int b = (int)((v + 1f) * 0.5f * 255f);      // torch .byte() truncates toward zero
                v = b / 255f * 2f - 1f;
            }
            dp[i] = v;
        }
        innerOut.Dispose();
        return d;
    }
}
