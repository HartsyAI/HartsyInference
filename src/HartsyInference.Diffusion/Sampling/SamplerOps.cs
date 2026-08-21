using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>Latent-space primitives the sampler implementations share, so each integrator reads as its own formula
/// rather than as tensor bookkeeping.
///
/// <para><b>Why the scratch tensors.</b> <see cref="IBackend.AffineMix"/> forbids its output aliasing an input, but
/// samplers overwhelmingly want in-place updates of the latent. These helpers allocate a latent-sized scratch, compute
/// into it, then copy back — preserving the caller's tensor identity, which matters because the device weight cache and
/// the pipeline both hold that reference. One extra latent-sized allocation per call is negligible against a UNet or
/// DiT forward; it is not worth a fused kernel until profiling says otherwise.</para></summary>
public static class SamplerOps
{
    /// <summary>In-place <c>target ← targetScale·target + otherScale·other</c>.</summary>
    public static void MixInto(IBackend backend, Tensor target, Tensor other, float targetScale, float otherScale)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(target);
        ArgumentNullException.ThrowIfNull(other);
        Tensor scratch = new Tensor(target.Shape, DType.F32);
        try
        {
            backend.AffineMix(scratch, target, other, targetScale, otherScale);
            backend.Scale(target, scratch, 1.0f);
        }
        finally
        {
            scratch.Dispose();
        }
    }

    /// <summary>In-place <c>target ← aScale·a + bScale·b</c>, where neither source is the target.</summary>
    public static void SetMix(IBackend backend, Tensor target, Tensor a, Tensor b, float aScale, float bScale)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(target);
        backend.AffineMix(target, a, b, aScale, bScale);
    }

    /// <summary>Adds <c>scale·N(0,1)</c> to <paramref name="target"/> in place, drawing the noise from a seed derived
    /// from <paramref name="baseSeed"/> and <paramref name="stepIndex"/>.
    ///
    /// <para>The draw is host-side (<see cref="SeedGenerator.CreateNoise"/>) and uploaded once per noised step. At a
    /// 1024² SDXL latent that is ~4 MB against a multi-second forward, so it does not register — but it IS a per-step
    /// H2D transfer that deterministic samplers never pay, which is the honest cost of the ancestral/SDE family.</para>
    ///
    /// <para><b>Reproducibility bound.</b> This is a seeded Gaussian, not ComfyUI's torchsde BrownianTree. Same seed
    /// reproduces exactly on this engine; it does NOT reproduce a ComfyUI image at the same seed. Swapping the noise
    /// source is a self-contained follow-up that leaves every integrator below untouched.</para></summary>
    public static void AddNoise(IBackend backend, Tensor target, TensorShape shape, int baseSeed, int stepIndex, float scale)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(target);
        if (scale <= 0f)
        {
            return;
        }
        Tensor noise = SeedGenerator.CreateNoise(shape, StepSeed(baseSeed, stepIndex));
        try
        {
            MixInto(backend, target, noise, 1.0f, scale);
        }
        finally
        {
            noise.Dispose();
        }
    }

    /// <summary>Per-step noise seed. Mixed through the two SplitMix64 odd constants rather than incremented, so
    /// neighbouring steps do not draw correlated streams and a repeated run reproduces step for step.</summary>
    public static int StepSeed(int baseSeed, int stepIndex) =>
        unchecked((baseSeed * 6364136223846793005L) + (stepIndex * 1442695040888963407L)).GetHashCode();

    /// <summary>Log-sigma, the domain the DPM-Solver family integrates in. Clamped away from zero so the terminal
    /// sigma cannot produce −∞ and poison an otherwise finite step.</summary>
    public static float LogSigma(float sigma) => MathF.Log(MathF.Max(sigma, 1e-10f));

    /// <summary>DPM-Solver++ half-log-SNR <c>lambda = −log(sigma)</c>, increasing as the schedule denoises.</summary>
    public static float Lambda(float sigma) => -LogSigma(sigma);
}
