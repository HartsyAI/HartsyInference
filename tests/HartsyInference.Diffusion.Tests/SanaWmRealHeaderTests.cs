using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real SANA-WM DiT header check. It reads only the header bytes, never the tensor data. Integration lane: needs
/// <c>HARTSYINFERENCE_MODELS_DIR</c> to hold <c>World/SanaWm/dit/sana_wm_1600m_720p.safetensors</c>.</summary>
[Trait("Category", "Integration")]
public sealed class SanaWmRealHeaderTests
{
    [SanaWmRealWeightsFact]
    public void RealDitHeader_DerivesDitDimensionsAndMatchesTheLayout()
    {
        SafeTensorHeader header = SafeTensorHeaderReader.Read(SanaWmRealWeightsFactAttribute.ResolveDitPath());

        Assert.Equal(872, header.Tensors.Count);
        SanaWmConfig config = SanaWmConfig.FromWeights(header.Tensors);
        SanaWmConfigTests.AssertRealDitDimensions(config);
        Assert.Empty(SanaWmCheckpointConverter.ValidateKeys(config, header.Tensors));
    }

    /// <summary>The streaming release's DiT, converted from its pickle to safetensors, keeps the stage-1 layout: the same 871
    /// tensors (stage 1's 872 without <c>pos_embed</c>), the same derived dimensions, and every expected shape.</summary>
    [SanaWmStreamingWeightsFact]
    public void StreamingDitHeader_KeepsTheStageOneLayout()
    {
        SafeTensorHeader header = SafeTensorHeaderReader.Read(SanaWmStreamingWeightsFactAttribute.DitPath());

        // The conversion drops pos_embed, as upstream's loader does; every other dimension is stage 1's.
        SanaWmConfig config = SanaWmConfig.FromWeights(header.Tensors);
        Assert.Equal(871, header.Tensors.Count);
        Assert.Equal(0, config.PosEmbedTokens);
        SanaWmConfigTests.AssertRealDitDimensions(config with { PosEmbedTokens = 484 });
        Assert.Empty(SanaWmCheckpointConverter.ValidateKeys(config, header.Tensors));
    }
}
