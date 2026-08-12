using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Covers the distilled 2.5 sampling contract. The schedule was baked in at distillation time, so it has
/// to survive verbatim — a re-derived or re-shifted schedule still produces an image, just a worse one.</summary>
public sealed class LtxVideo2DistilledScheduleTests
{
    [Fact]
    public void DistilledPresetCarriesTheReferenceSchedule()
    {
        // From the reference pipeline's DISTILLED_SIGMA_VALUES.
        float[] expected = [1.0f, 0.99375f, 0.9875f, 0.98125f, 0.975f, 0.909375f, 0.725f, 0.421875f, 0.0f];
        float[] actual = LtxVideo2Config.V25Distilled.FixedSigmas!;

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 6);
    }

    [Fact]
    public void ScheduleIsMonotonicAndTerminatesAtZero()
    {
        float[] sigmas = LtxVideo2Config.V25Distilled.FixedSigmas!;

        Assert.Equal(1.0f, sigmas[0], 6);
        Assert.Equal(0.0f, sigmas[^1], 6);
        for (int i = 1; i < sigmas.Length; i++)
            Assert.True(sigmas[i] < sigmas[i - 1], $"sigma {i} ({sigmas[i]}) is not below sigma {i - 1} ({sigmas[i - 1]})");
    }

    [Fact]
    public void StepCountIsOneLessThanTheSigmaCount()
    {
        // The loop consumes tsteps[k] and tsteps[k+1], so N steps need N+1 sigmas.
        LtxVideo2Config config = LtxVideo2Config.V25Distilled;
        Assert.Equal(config.FixedSigmas!.Length - 1, config.NumInferenceSteps);
        Assert.Equal(8, config.NumInferenceSteps);
    }

    [Fact]
    public void DistilledRunsUnguided()
    {
        // CFG 1 is what lets the pipeline drop the unconditional branch, and it is also why the distilled model
        // sidesteps the CFG-dispersion pathology behind the 2.3 quiet-audio issue.
        Assert.Equal(1.0f, LtxVideo2Config.V25Distilled.GuidanceScale, 6);
        Assert.Null(LtxVideo2Config.V25Distilled.AudioGuidanceScale);
    }

    [Fact]
    public void DevAndBaseVariantsKeepTheDynamicSchedule()
    {
        Assert.Null(LtxVideo2Config.V25.FixedSigmas);
        Assert.Null(LtxVideo2Config.V23.FixedSigmas);
    }

    [Fact]
    public void DistilledDiffersFromDevOnlyInSampling()
    {
        // The two checkpoints are architecturally indistinguishable, so the presets must not diverge on anything
        // the weights would have to agree with.
        LtxVideo2Config dev = LtxVideo2Config.V25;
        LtxVideo2Config distilled = LtxVideo2Config.V25Distilled;

        Assert.Equal(dev.FfBias, distilled.FfBias);
        Assert.Equal(dev.UseKeyframesAbsPosEmbedding, distilled.UseKeyframesAbsPosEmbedding);
        Assert.Equal(dev.NumLayers, distilled.NumLayers);
        Assert.Equal(dev.NumHeads, distilled.NumHeads);
        Assert.Equal(dev.HeadDim, distilled.HeadDim);
        Assert.Equal(dev.RopeType, distilled.RopeType);
        Assert.Equal(dev.CrossAttentionDim, distilled.CrossAttentionDim);
        Assert.Equal(dev.AudioCrossAttentionDim, distilled.AudioCrossAttentionDim);
    }
}
