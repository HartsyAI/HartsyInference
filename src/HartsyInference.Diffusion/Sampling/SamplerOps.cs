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

    /// <summary>Adds <c>scale·N(0,1)</c> to <paramref name="target"/> in place, drawn from a seed derived from
    /// <paramref name="baseSeed"/> and <paramref name="stepIndex"/>.</summary>
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

    /// <summary>In-place <c>target ← target + scale·noise</c> for noise drawn from <paramref name="source"/>.</summary>
    public static void AddNoise(IBackend backend, Tensor target, INoiseSource source, int stepIndex, int subDraw,
        float sigma, float sigmaNext, float scale)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(target);
        ArgumentNullException.ThrowIfNull(source);
        if (scale == 0f)
        {
            return;
        }
        using Tensor noise = source.Sample(stepIndex, subDraw, sigma, sigmaNext);
        MixInto(backend, target, noise, 1.0f, scale);
    }

    /// <summary>In-place <c>target ← scale·target</c>.</summary>
    public static void ScaleInPlace(IBackend backend, Tensor target, float scale)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(target);
        if (scale == 1.0f)
        {
            return;
        }
        using Tensor scratch = new Tensor(target.Shape, DType.F32);
        backend.Scale(scratch, target, scale);
        backend.Scale(target, scratch, 1.0f);
    }

    /// <summary>In-place <c>target ← a·target + b·x + c·y</c>.</summary>
    public static void MixInto(IBackend backend, Tensor target, Tensor x, Tensor y, float targetScale, float xScale, float yScale)
    {
        MixInto(backend, target, x, targetScale, xScale);
        MixInto(backend, target, y, 1.0f, yScale);
    }

    /// <summary><c>output ← a·x + b·y + c·z</c>; <paramref name="output"/> must not alias an input.</summary>
    public static void SetMix(IBackend backend, Tensor output, Tensor x, Tensor y, Tensor z, float xScale, float yScale, float zScale)
    {
        ArgumentNullException.ThrowIfNull(backend);
        backend.AffineMix(output, x, y, xScale, yScale);
        MixInto(backend, output, z, 1.0f, zScale);
    }

    /// <summary>Per-step noise seed, mixed through the SplitMix64 odd constants so neighbouring steps draw
    /// uncorrelated streams.</summary>
    public static int StepSeed(int baseSeed, int stepIndex) =>
        unchecked((baseSeed * 6364136223846793005L) + (stepIndex * 1442695040888963407L)).GetHashCode();

    /// <summary>Seed for draw <paramref name="subDraw"/> inside a step; draw 0 equals <see cref="StepSeed(int,int)"/>.</summary>
    public static int StepSeed(int baseSeed, int stepIndex, int subDraw) => subDraw == 0
        ? StepSeed(baseSeed, stepIndex)
        : unchecked((baseSeed * 6364136223846793005L) + (stepIndex * 1442695040888963407L)
            + (subDraw * -7046029254386353131L)).GetHashCode();

    /// <summary>Log-sigma, the domain the DPM-Solver family integrates in. Clamped away from zero so the terminal
    /// sigma cannot produce −∞ and poison an otherwise finite step.</summary>
    public static float LogSigma(float sigma) => MathF.Log(MathF.Max(sigma, 1e-10f));

    /// <summary>DPM-Solver++ half-log-SNR <c>lambda = −log(sigma)</c>, increasing as the schedule denoises.</summary>
    public static float Lambda(float sigma) => -LogSigma(sigma);
}
