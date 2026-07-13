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
        // Device-resident EDM step (drain-free — no host DataPointer reads → graph-capturable):
        // rn = noisy·cIn, ro = obs/sigma_data → U-Net → d = cSkip·noisy + cOut·inner, then (optional) pixel-quantize.
        Tensor rn = new(noisy.Shape, DType.F32); backend.Scale(rn, noisy, cIn);
        Tensor ro = new(obs.Shape, DType.F32); backend.Scale(ro, obs, 1f / _cfg.SigmaData);

        Tensor innerOut = _inner.Forward(backend, rn, cNoise, ro, act);
        rn.Dispose(); ro.Dispose();

        Tensor skip = new(noisy.Shape, DType.F32); backend.Scale(skip, noisy, cSkip);
        Tensor outp = new(noisy.Shape, DType.F32); backend.Scale(outp, innerOut, cOut);
        innerOut.Dispose();
        Tensor d = new(noisy.Shape, DType.F32); backend.Add(d, skip, outp);
        skip.Dispose(); outp.Dispose();
        if (quantize)
        {
            Tensor q = new(noisy.Shape, DType.F32); backend.PixelQuantize(q, d);  // clamp[-1,1] + 256-level quantize
            d.Dispose();
            return q;
        }
        return d;
    }
}
