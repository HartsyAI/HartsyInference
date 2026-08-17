using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Covers the distilled 2.5 sampling contract. Every failure here is silent: a drifted schedule or a
/// preset that diverged from dev still produces video, just worse video.</summary>
public sealed class LtxVideo2DistilledScheduleTests
{
    [Fact]
    public void DistilledPresetCarriesTheReferenceSchedule()
    {
        // Transcribed from the reference pipeline's DISTILLED_SIGMA_VALUES; distillation baked these in, so a
        // typo is not recoverable by tuning anything else.
        float[] expected = [1.0f, 0.99375f, 0.9875f, 0.98125f, 0.975f, 0.909375f, 0.725f, 0.421875f, 0.0f];
        float[] actual = LtxVideo2Config.V25Distilled.FixedSigmas!;

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 6);

        // The shipped 2.5 templates are all two-stage; the preset must carry the flow, not just the sigmas.
        Assert.True(LtxVideo2Config.V25Distilled.TwoStage);
    }

    [Fact]
    public void DistilledDiffersFromDevOnlyInSampling()
    {
        // The two checkpoints are architecturally indistinguishable, so the presets must not diverge on anything
        // the weights would have to agree with — only on the sampling contract (schedule, guidance, two-stage).
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
        Assert.Null(dev.FixedSigmas);
        Assert.False(dev.TwoStage);
    }

    [Fact]
    public void DistilledSigmasAreNotSharedBetweenConfigs()
    {
        // A shared array instance would let one in-place edit anywhere corrupt every config built afterwards.
        float[] first = LtxVideo2Config.V25Distilled.FixedSigmas!;
        first[0] = -1f;

        Assert.Equal(1.0f, LtxVideo2Config.V25Distilled.FixedSigmas![0]);
    }
}
