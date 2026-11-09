using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

public sealed class ShapeVaeKeyMapTests
{
    [Theory]
    [InlineData("blocks.1.0.conv.conv.weight", "Blocks[1][0].Conv.Conv.Weight")]
    [InlineData("blocks.2.4.to_subdiv.bias", "Blocks[2][4].ToSubdiv.Bias")]
    [InlineData("input_layer.weight", "InputLayer.Weight")]
    [InlineData("blocks.0.1.mlp.2.weight", "Blocks[0][1].Mlp[2].Weight")]
    public void ToParameterName_ConvertsDottedKeysToPathNames(string key, string expected)
    {
        Assert.Equal(expected, ShapeVaeKeyMap.ToParameterName(key));
    }

    [Fact]
    public void Encoder_CountMatchesSyntheticFormula()
    {
        // 4 (input + to_latent) + 8 per ConvNeXt block (4 blocks) + 6 per S2C transition (2).
        IReadOnlyList<ShapeVaeKeyMap.TensorSpec> specs = ShapeVaeKeyMap.Encoder(ShapeVaeTestData.Encoder());
        Assert.Equal(4 + 8 * 4 + 6 * 2, specs.Count);
    }

    [Fact]
    public void Decoder_CountMatchesSyntheticFormula()
    {
        // 4 (from_latent + output_layer) + 8 per ConvNeXt block (3 blocks) + 8 per C2S transition (2).
        IReadOnlyList<ShapeVaeKeyMap.TensorSpec> specs = ShapeVaeKeyMap.Decoder(ShapeVaeTestData.Decoder());
        Assert.Equal(4 + 8 * 3 + 8 * 2, specs.Count);
    }

    [Fact]
    public void ShippedConfig_CountsMatchUpstreamModules()
    {
        // Encoder: 4 + 8·32 ConvNeXt + 6·4 S2C. Decoder: 4 + 8·32 ConvNeXt + 8·4 C2S.
        ShapeVaeConfig shipped = ShapeVaeConfig.NextDcF16C32;
        Assert.Equal(4 + 8 * 32 + 6 * 4, ShapeVaeKeyMap.Encoder(shipped.Encoder).Count);
        Assert.Equal(4 + 8 * 32 + 8 * 4, ShapeVaeKeyMap.Decoder(shipped.Decoder).Count);
    }

    [Fact]
    public void Specs_HaveUniqueKeysAndParameterNames()
    {
        IReadOnlyList<ShapeVaeKeyMap.TensorSpec> specs = ShapeVaeKeyMap.Encoder(ShapeVaeTestData.Encoder());

        Assert.Equal(specs.Count, specs.Select(s => s.CheckpointKey).Distinct().Count());
        Assert.Equal(specs.Count, specs.Select(s => s.ParameterName).Distinct().Count());
        foreach (ShapeVaeKeyMap.TensorSpec spec in specs)
            Assert.Equal(ShapeVaeKeyMap.ToParameterName(spec.CheckpointKey), spec.ParameterName);
    }

    [Fact]
    public void Encoder_LoadsWhenEveryExpectedKeyIsPresent()
    {
        ShapeVaeEncoderConfig config = ShapeVaeTestData.Encoder();
        Dictionary<string, Tensor> weights = ShapeVaeTestData.RandomWeights(ShapeVaeKeyMap.Encoder(config), seed: 3);

        using ShapeVaeEncoder encoder = new(config, weights);
        Assert.NotNull(encoder);
    }

    [Fact]
    public void Decoder_LoadsWhenEveryExpectedKeyIsPresent()
    {
        ShapeVaeDecoderConfig config = ShapeVaeTestData.Decoder();
        Dictionary<string, Tensor> weights = ShapeVaeTestData.RandomWeights(ShapeVaeKeyMap.Decoder(config), seed: 4);

        using ShapeVaeDecoder decoder = new(config, weights);
        Assert.NotNull(decoder);
    }

    [Fact]
    public void Encoder_NamesEachMissingKey()
    {
        ShapeVaeEncoderConfig config = ShapeVaeTestData.Encoder();
        IReadOnlyList<ShapeVaeKeyMap.TensorSpec> specs = ShapeVaeKeyMap.Encoder(config);
        Dictionary<string, Tensor> full = ShapeVaeTestData.RandomWeights(specs, seed: 5);

        // Dropping any single tensor must fail at construction, naming that key.
        foreach (ShapeVaeKeyMap.TensorSpec spec in specs)
        {
            Dictionary<string, Tensor> partial = new(full);
            partial.Remove(spec.CheckpointKey);
            HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => new ShapeVaeEncoder(config, partial));
            Assert.Contains(spec.CheckpointKey, ex.Message);
        }
    }

    [Fact]
    public void Decoder_RejectsWrongShape()
    {
        ShapeVaeDecoderConfig config = ShapeVaeTestData.Decoder();
        Dictionary<string, Tensor> weights = ShapeVaeTestData.RandomWeights(ShapeVaeKeyMap.Decoder(config), seed: 6);
        weights["output_layer.bias"] = new Tensor(new TensorShape(6), DType.F32);

        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => new ShapeVaeDecoder(config, weights));
        Assert.Contains("output_layer.bias", ex.Message);
    }
}
