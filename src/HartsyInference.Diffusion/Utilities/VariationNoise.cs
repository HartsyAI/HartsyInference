using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>Variation-seed noise blending (ComfyUI's SwarmKSampler second-seed semantics): rewrites already-seeded
/// base noise as <c>slerp(base, noise(variationSeed), strength)</c>. Lives in the Diffusion layer so
/// <c>DiffusionPipelineBase.TakeOrCreateNoise</c> can blend at the exact shape each pipeline seeds in; the single
/// implementation in the codebase — the Engine forwards the request fields and pipelines blend here.</summary>
public static class VariationNoise
{
    /// <summary>Blends noise for <paramref name="variationSeed"/> into <paramref name="baseNoise"/> in place.
    /// Strength is clamped to [0,1]; 1 replaces the base entirely, and a negative seed draws a random one.</summary>
    public static void BlendInPlace(Tensor baseNoise, TensorShape shape, long variationSeed, double strength)
    {
        int varSeed = variationSeed < 0 ? SeedGenerator.RandomSeed() : (int)(variationSeed & 0x7FFFFFFF);
        double t = Math.Clamp(strength, 0.0, 1.0);
        Tensor varNoise = SeedGenerator.CreateNoise(shape, varSeed);
        SlerpInPlace(baseNoise, varNoise, (float)t);
        varNoise.Dispose();
        Logs.Verbose($"[Variation] Seed {varSeed} blended at strength {t} (slerp, shape={shape}).");
    }

    /// <summary>Spherical interpolation of <paramref name="a"/> toward <paramref name="b"/> by <paramref name="t"/>,
    /// written into <paramref name="a"/>. Slerp (not lerp) keeps the norm consistent with unit-variance Gaussian
    /// noise — a straight lerp of two independent Gaussians shrinks variance and washes the image out.</summary>
    public static unsafe void SlerpInPlace(Tensor a, Tensor b, float t)
    {
        long count = a.Shape.ElementCount;
        float* pa = (float*)a.DataPointer;
        float* pb = (float*)b.DataPointer;

        double dot = 0;
        double na = 0;
        double nb = 0;
        for (long i = 0; i < count; i++)
        {
            dot += (double)pa[i] * pb[i];
            na += (double)pa[i] * pa[i];
            nb += (double)pb[i] * pb[i];
        }
        na = Math.Sqrt(na);
        nb = Math.Sqrt(nb);
        double cosTheta = Math.Clamp(dot / Math.Max(na * nb, 1e-12), -1.0, 1.0);
        double theta = Math.Acos(cosTheta);
        double sinTheta = Math.Sin(theta);

        float wa;
        float wb;
        if (sinTheta < 1e-6)
        {
            // Nearly colinear — fall back to lerp (slerp is numerically unstable here).
            wa = 1.0f - t;
            wb = t;
        }
        else
        {
            wa = (float)(Math.Sin((1.0 - t) * theta) / sinTheta);
            wb = (float)(Math.Sin(t * theta) / sinTheta);
        }
        for (long i = 0; i < count; i++)
        {
            pa[i] = wa * pa[i] + wb * pb[i];
        }
    }
}
