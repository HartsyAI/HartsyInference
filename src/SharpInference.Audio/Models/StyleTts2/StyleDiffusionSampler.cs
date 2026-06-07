using SharpInference.Audio.Dsp;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.StyleTts2;

/// <summary>The denoiser <c>D(x, σ)</c> the style sampler integrates — it returns the denoised
/// 256-d style estimate (x0) for a noisy latent <paramref name="x"/> at noise level
/// <paramref name="sigma"/>. Implementations bundle the EDM (Karras) preconditioning + classifier-free
/// guidance + the conditioning (PLBERT text features, optional reference style).</summary>
public interface IStyleDenoiser
{
    Tensor Denoise(IBackend backend, Tensor x, float sigma);
}

/// <summary>StyleTTS 2's diffusion style sampler — generates the 256-d style vector by integrating an
/// <see cref="IStyleDenoiser"/> over a Karras sigma schedule with the ADPM2 (ancestral DPM-Solver-2)
/// second-order sampler. Mirrors <c>Demo/Inference_LibriTTS.ipynb</c>:
/// <c>DiffusionSampler(model.diffusion, sampler=ADPM2Sampler(), sigma_schedule=KarrasSchedule(1e-4, 3.0, rho=9), clamp=False)</c>.
///
/// <para>The schedule + sampler are exactly specified and unit-tested; the <see cref="IStyleDenoiser"/>
/// network (StyleTransformer1d) is the checkpoint-gated piece.</para></summary>
public sealed unsafe class StyleDiffusionSampler
{
    private readonly StyleTts2Config _cfg;

    public StyleDiffusionSampler(StyleTts2Config cfg) => _cfg = cfg;

    /// <summary>Builds the Karras sigma schedule: <c>σ_i = (σ_max^{1/ρ} + i/(n−1)·(σ_min^{1/ρ} − σ_max^{1/ρ}))^ρ</c>
    /// for <c>i ∈ [0, n)</c>, with a trailing <c>0</c> appended (n+1 entries).</summary>
    public static float[] KarrasSchedule(int numSteps, float sigmaMin, float sigmaMax, float rho)
    {
        float[] sigmas = new float[numSteps + 1];
        float minInv = MathF.Pow(sigmaMin, 1f / rho);
        float maxInv = MathF.Pow(sigmaMax, 1f / rho);
        for (int i = 0; i < numSteps; i++)
        {
            float ramp = numSteps == 1 ? 0f : (float)i / (numSteps - 1);
            sigmas[i] = MathF.Pow(maxInv + ramp * (minInv - maxInv), rho);
        }
        sigmas[numSteps] = 0f;
        return sigmas;
    }

    /// <summary>Samples a 256-d style vector. Mirrors <c>ADPM2Sampler</c>: starts from
    /// <c>σ_max·noise</c> and runs an ancestral second-order (midpoint) update per step. Deterministic
    /// for a fixed <paramref name="seed"/>.</summary>
    /// <returns>A <c>[1, 1, 256]</c> style latent (caller squeezes to <c>[1, 256]</c>).</returns>
    public Tensor Sample(IBackend backend, IStyleDenoiser denoiser, int seed)
    {
        float[] sigmas = KarrasSchedule(_cfg.DiffusionSteps, _cfg.SigmaMin, _cfg.SigmaMax, _cfg.SigmaRho);
        int dim = _cfg.StyleDim;
        uint rng = DeterministicRng.Seed(seed);

        // x = σ_0 · noise.
        Tensor x = new(new TensorShape(1, 1, dim), DType.F32);
        float* xp = (float*)x.DataPointer;
        for (int i = 0; i < dim; i++) xp[i] = sigmas[0] * DeterministicRng.NextGaussian(ref rng);

        for (int step = 0; step < _cfg.DiffusionSteps; step++)
        {
            float sigma = sigmas[step];
            float sigmaNext = sigmas[step + 1];

            // Ancestral split: σ_up = sqrt(σ_next²·(σ²−σ_next²)/σ²); σ_down = sqrt(σ_next²−σ_up²).
            float sigmaUp = sigma > 0f
                ? MathF.Sqrt(MathF.Max(0f, sigmaNext * sigmaNext * (sigma * sigma - sigmaNext * sigmaNext) / (sigma * sigma)))
                : 0f;
            float sigmaDown = MathF.Sqrt(MathF.Max(0f, sigmaNext * sigmaNext - sigmaUp * sigmaUp));

            // Midpoint σ (rho-1 interpolation, the ADPM2 default).
            float sigmaMid = MathF.Pow((MathF.Pow(sigma, 1f) + MathF.Pow(sigmaDown, 1f)) / 2f, 1f);

            // Euler to the midpoint using the denoised direction.
            Tensor d0 = denoiser.Denoise(backend, x, sigma);
            Tensor xMid = StepAlong(x, d0, sigma, sigmaMid);
            d0.Dispose();

            // Second-order correction from the midpoint to σ_down.
            Tensor dMid = denoiser.Denoise(backend, xMid, sigmaMid);
            Tensor xNext = StepAlong(xMid, dMid, sigmaMid, sigmaDown);
            dMid.Dispose();
            xMid.Dispose();

            // Ancestral noise injection.
            if (sigmaUp > 0f)
            {
                float* np = (float*)xNext.DataPointer;
                for (int i = 0; i < dim; i++) np[i] += sigmaUp * DeterministicRng.NextGaussian(ref rng);
            }
            x.Dispose();
            x = xNext;
        }
        return x;
    }

    /// <summary>One step <c>x' = x + (x − D(x,σ))/σ · (σ_target − σ)</c> using the denoised estimate
    /// <paramref name="denoised"/> to form the score direction <c>d = (x − D)/σ</c>.</summary>
    private static Tensor StepAlong(Tensor x, Tensor denoised, float sigma, float sigmaTarget)
    {
        Tensor outT = new(x.Shape, DType.F32);
        float* xp = (float*)x.DataPointer;
        float* dp = (float*)denoised.DataPointer;
        float* op = (float*)outT.DataPointer;
        long n = x.ElementCount;
        float invSigma = sigma > 0f ? 1f / sigma : 0f;
        float dt = sigmaTarget - sigma;
        for (long i = 0; i < n; i++)
        {
            float d = (xp[i] - dp[i]) * invSigma;       // score direction
            op[i] = xp[i] + d * dt;
        }
        return outT;
    }
}
