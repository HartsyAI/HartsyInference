using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.Diamond;

/// <summary>DIAMOND diffusion sampler: a Karras (EDM) rho-schedule with an Euler (order-1) integrator that
/// rolls the <see cref="DiamondDenoiser"/> forward to sample the next frame from the last
/// <c>NumStepsConditioning</c> frames + actions. Mirrors <c>diffusion_sampler.py</c> (s_churn=0 → deterministic
/// Euler). Since the denoiser is bit-exact vs the reference, the sample loop is verified by composition.</summary>
public sealed unsafe class DiamondSampler
{
    private readonly DiamondDenoiser _denoiser;
    private readonly DiamondConfig _cfg;
    private readonly float[] _sigmas;   // length NumStepsDenoising + 1 (trailing 0)

    public DiamondSampler(DiamondDenoiser denoiser)
    {
        _denoiser = denoiser;
        _cfg = denoiser.Config;
        _sigmas = BuildSigmas(_cfg.NumStepsDenoising, _cfg.SigmaMin, _cfg.SigmaMax, _cfg.Rho);
    }

    /// <summary>Karras rho-schedule: <c>(σ_max^(1/ρ) + l·(σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ</c> for l∈linspace(0,1,n),
    /// with a trailing 0.</summary>
    public static float[] BuildSigmas(int numSteps, float sigmaMin, float sigmaMax, int rho)
    {
        float minInv = MathF.Pow(sigmaMin, 1f / rho), maxInv = MathF.Pow(sigmaMax, 1f / rho);
        float[] s = new float[numSteps + 1];
        for (int i = 0; i < numSteps; i++)
        {
            float l = numSteps == 1 ? 0f : i / (float)(numSteps - 1);
            s[i] = MathF.Pow(maxInv + l * (minInv - maxInv), rho);
        }
        s[numSteps] = 0f;
        return s;
    }

    public IReadOnlyList<float> Sigmas => _sigmas;

    /// <summary>Samples the next frame given the initial noise <paramref name="xInit"/> <c>[1,3,H,W]</c> (the
    /// caller supplies it — seeded or dumped for parity), the context <paramref name="obs"/> <c>[1, 4·3, H, W]</c>,
    /// and <paramref name="act"/>. Returns the clean next frame in [-1,1].</summary>
    public Tensor Sample(IBackend backend, Tensor xInit, Tensor obs, ReadOnlySpan<int> act)
    {
        Tensor x = new(xInit.Shape, DType.F32);
        backend.CopyInto(x, xInit);
        for (int step = 0; step < _cfg.NumStepsDenoising; step++)
        {
            float sigma = _sigmas[step], nextSigma = _sigmas[step + 1];
            Tensor denoised = _denoiser.Denoise(backend, x, sigma, obs, act, quantize: true);
            // Euler on device: x += (x − denoised)·(dt/sigma)  (drain-free — no host readback → graph-capturable).
            float a = (nextSigma - sigma) / sigma;
            Tensor negDen = new(x.Shape, DType.F32); backend.Scale(negDen, denoised, -1f);
            Tensor diff = new(x.Shape, DType.F32); backend.Add(diff, x, negDen);   // x − denoised
            negDen.Dispose(); denoised.Dispose();
            Tensor scaled = new(x.Shape, DType.F32); backend.Scale(scaled, diff, a);
            diff.Dispose();
            Tensor next = new(x.Shape, DType.F32); backend.Add(next, x, scaled);
            scaled.Dispose(); x.Dispose();
            x = next;
        }
        return x;
    }
}
