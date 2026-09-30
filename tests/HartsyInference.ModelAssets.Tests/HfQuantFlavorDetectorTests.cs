using System.Text.Json;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Quant;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Flavor detection over the quantization blocks the published DeepSeek-V4.1 derivatives declare (trimmed from real configs).</summary>
public sealed class HfQuantFlavorDetectorTests
{
    private static QuantFlavor? Detect(string json)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        return HfQuantFlavorDetector.FromConfig(document.RootElement);
    }

    [Fact]
    public void Official_IsFp8WithoutAProducer()
    {
        Assert.Equal(QuantFlavor.Official, Detect(
            "{\"quantization_config\":{\"quant_method\":\"fp8\",\"scale_fmt\":\"ue8m0\",\"expert_dtype\":\"fp4\"}}"));
    }

    [Fact]
    public void Nvidia_SaysFp8ButItsModelOptProducerDecides()
    {
        Assert.Equal(QuantFlavor.NvidiaNvfp4, Detect(
            "{\"quantization_config\":{\"quant_method\":\"fp8\",\"moe_quant_algo\":\"NVFP4\",\"producer\":{\"name\":\"modelopt\"}}}"));
        Assert.Equal(QuantFlavor.NvidiaNvfp4, Detect(
            "{\"quantization_config\":{\"quant_method\":\"fp8\",\"producer\":{\"name\":\"modelopt\"}}}"));
        Assert.Equal(QuantFlavor.NvidiaNvfp4, Detect("{\"quantization_config\":{\"quant_method\":\"modelopt\"}}"));
    }

    [Fact]
    public void QuantizationBlockNestedUnderTextConfig_IsFound()
    {
        Assert.Equal(QuantFlavor.AmdQuark, Detect("{\"text_config\":{\"quantization_config\":{\"quant_method\":\"quark\"}}}"));
    }

    [Fact]
    public void AmdQuark_IsQuarkMethod()
    {
        Assert.Equal(QuantFlavor.AmdQuark, Detect("{\"quantization_config\":{\"quant_method\":\"quark\",\"algo_config\":null}}"));
    }

    [Fact]
    public void Exl3_IsExl3MethodEvenWithANestedFp8Block()
    {
        Assert.Equal(QuantFlavor.Exl3, Detect(
            "{\"quantization_config\":{\"quant_method\":\"exl3\",\"non_routed_quantization\":{\"quant_method\":\"x\"}}}"));
    }

    [Fact]
    public void Mlx_IsAnAffineQuantizationBlockWithoutAMethod()
    {
        Assert.Equal(QuantFlavor.Mlx, Detect("{\"quantization\":{\"group_size\":64,\"bits\":4,\"mode\":\"affine\"}}"));
        Assert.Equal(QuantFlavor.Mlx, Detect("{\"quantization_config\":{\"group_size\":64,\"bits\":4,\"mode\":\"Affine\"}}"));
    }

    [Theory]
    [InlineData("{}")]
    [InlineData("[]")]
    [InlineData("{\"quantization_config\":\"fp8\"}")]
    [InlineData("{\"quantization_config\":{\"quant_method\":\"gptq\"}}")]
    [InlineData("{\"quantization\":{\"mode\":\"mxfp4\"}}")]
    public void UnknownOrAbsentQuantization_IsNull(string json) => Assert.Null(Detect(json));
}
