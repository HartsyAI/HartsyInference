using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;
using VariationSeedRequest = HartsyInference.Engine.Requests.VariationSeed;

namespace HartsyInference.Engine.Features;

/// <summary>Variation-seed support (ComfyUI's SwarmKSampler second-seed blended noise): builds the initial noise as
/// <c>slerp(noise(seed), noise(variationSeed), strength)</c> for the pipeline's injected-noise slot. Strength 0
/// reproduces the base seed exactly, since <see cref="SeedGenerator.CreateNoise"/> is the same generator the pipeline
/// would have used.</summary>
public static class VariationSeedResolver
{
    /// <summary>Latent channel count for SD 1.5 / SDXL spatial latents.</summary>
    public const int SdLatentChannels = 4;

    /// <summary>Latent channel count for Flux and other 16-channel-VAE architectures. Flux injects the unpacked
    /// <c>[1, 16, H/8, W/8]</c> noise BEFORE its 2×2 patchify, so the variation noise is built in that unpacked space.</summary>
    public const int FluxLatentChannels = 16;

    /// <summary>Returns the blended noise tensor, or null when no variation seed is requested (null spec or strength 0) —
    /// null means "let the pipeline seed normally". Pass the SAME base seed placed on the request so strength→0 continuity
    /// holds. <paramref name="latentChannels"/> must match the architecture's unpacked latent channel count; the pipeline
    /// validates the injected-noise shape and throws on mismatch.</summary>
    public static Tensor? Resolve(
        VariationSeedRequest? variation, int width, int height, long? requestSeed, int latentChannels = SdLatentChannels)
    {
        if (variation is null || variation.Strength <= 0)
        {
            return null;
        }
        int baseSeed = requestSeed is null or < 0 ? SeedGenerator.RandomSeed() : (int)(requestSeed.Value & 0x7FFFFFFF);
        int varSeed = variation.Seed < 0 ? SeedGenerator.RandomSeed() : (int)(variation.Seed & 0x7FFFFFFF);
        double strength = Math.Clamp(variation.Strength, 0.0, 1.0);

        TensorShape shape = new TensorShape(1, latentChannels, height / 8, width / 8);
        Tensor baseNoise = SeedGenerator.CreateNoise(shape, baseSeed);
        if (strength >= 1.0)
        {
            // Full replacement: just use the variation seed's noise.
            baseNoise.Dispose();
            Logs.Verbose($"[Features][Variation] Seed {varSeed} at strength 1 — replacing base noise entirely.");
            return SeedGenerator.CreateNoise(shape, varSeed);
        }
        Tensor varNoise = SeedGenerator.CreateNoise(shape, varSeed);
        SlerpInPlace(baseNoise, varNoise, (float)strength);
        varNoise.Dispose();
        Logs.Verbose($"[Features][Variation] Seed {varSeed} blended at strength {strength} (slerp).");
        return baseNoise;
    }

    /// <summary>Spherical interpolation of <paramref name="a"/> toward <paramref name="b"/> by <paramref name="t"/>,
    /// written into <paramref name="a"/>. Slerp (not lerp) keeps the norm consistent with unit-variance Gaussian noise —
    /// a straight lerp of two independent Gaussians shrinks variance and washes the image out.</summary>
    private static unsafe void SlerpInPlace(Tensor a, Tensor b, float t)
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
