using HartsyInference.Vision.Dinov2;
using HartsyInference.Vision.Siglip;
using Xunit;

namespace HartsyInference.Vision.Tests;

/// <summary>Structural sanity checks for the B1 encoder presets (SigLIP 2, DINOv3). Pure data — guards
/// against typo'd dimensions that would silently break checkpoint loading. Real-weight cos-sim parity is
/// env-gated and lives in the checkpoint tests.</summary>
public sealed class EncoderPresetSurfaceTests
{
    [Fact]
    public void Siglip2_FixedResPresets_HaveConsistentDims()
    {
        foreach (SiglipPreset p in new[]
        {
            SiglipPreset.V2Base16_224, SiglipPreset.V2Large16_384,
            SiglipPreset.V2So400m14_384, SiglipPreset.V2Giant16_384,
        })
        {
            Assert.Equal(p.HiddenSize, p.EmbeddingDim);
            Assert.Equal(0, p.HiddenSize % p.NumHeads);
            Assert.Equal((p.ImageSize / p.PatchSize) * (p.ImageSize / p.PatchSize), p.NumPatches);
            Assert.StartsWith("google/siglip2-", p.Name);
        }
    }

    [Fact]
    public void Siglip2Giant_Has40Layers_1536Hidden()
    {
        SiglipPreset p = SiglipPreset.V2Giant16_384;
        Assert.Equal(1536, p.HiddenSize);
        Assert.Equal(40, p.NumLayers);
        Assert.Equal(6144, p.IntermediateSize);
    }

    [Fact]
    public void Dinov3_Presets_FlagRope_And4Registers()
    {
        foreach (Dinov2Preset p in new[]
        {
            Dinov2Preset.V3Small16, Dinov2Preset.V3Base16, Dinov2Preset.V3Large16,
        })
        {
            // DINOv3 uses RoPE (encoder support pending) + 4 register tokens + 16px patch.
            Assert.True(p.UsesRotaryPositionEmbedding);
            Assert.Equal(4, p.NumRegisterTokens);
            Assert.Equal(16, p.PatchSize);
            Assert.Equal(1 + 4 + p.NumPatches, p.SequenceLength);
        }
    }

    [Fact]
    public void Dinov2_Presets_DoNotFlagRope()
    {
        Assert.False(Dinov2Preset.Large.UsesRotaryPositionEmbedding);
        Assert.False(Dinov2Preset.Giant.UsesRotaryPositionEmbedding);
        Assert.Equal(14, Dinov2Preset.Large.PatchSize);
    }
}
