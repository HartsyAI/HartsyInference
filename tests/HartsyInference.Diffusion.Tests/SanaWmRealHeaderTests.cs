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
}
