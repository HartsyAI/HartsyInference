using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Utonia;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Utonia;

public sealed class UtoniaCheckpointKeysTests
{
    [Theory]
    [InlineData(true, true, 72)]
    [InlineData(true, false, 69)]
    [InlineData(false, true, 128)]
    public void Parameters_CountMatchesTheArchitectureWalk(bool encoderOnly, bool qkvBias, int expected)
    {
        IReadOnlyList<UtoniaParameterSpec> specs = UtoniaCheckpointKeys.Parameters(UtoniaTestData.TinyConfig(encoderOnly, qkvBias));

        Assert.Equal(expected, specs.Count);
    }

    [Fact]
    public void Parameters_EveryImpliedTorchKeyMapsToItsCSharpNameAndNamesAreUnique()
    {
        IReadOnlyList<UtoniaParameterSpec> specs = UtoniaCheckpointKeys.Parameters(UtoniaTestData.TinyConfig(encoderOnly: false));

        int mapped = specs.Count(spec => UtoniaCheckpointKeys.ToCSharpName(spec.TorchKey) == spec.CSharpName);

        Assert.Equal(specs.Count, mapped);
        Assert.Equal(specs.Count, specs.Select(spec => spec.CSharpName).Distinct(StringComparer.Ordinal).Count());
    }

    [Fact]
    public void Buffers_AreComputedRopeFrequenciesAndNeverMapped()
    {
        IReadOnlyList<string> buffers = UtoniaCheckpointKeys.Buffers(UtoniaTestData.TinyConfig(encoderOnly: false));

        Assert.Equal(5, buffers.Count);
        Assert.All(buffers, key => Assert.True(UtoniaCheckpointKeys.IsComputedBuffer(key)));
        Assert.All(buffers, key => Assert.Null(UtoniaCheckpointKeys.ToCSharpName(key)));
    }

    [Theory]
    [InlineData("enc.enc0.block0.cpe.0.weight", "Encoder.Stage0.Block0.Cpe.Conv.Weight")]
    [InlineData("enc.enc3.block11.cpe.0.bias", "Encoder.Stage3.Block11.Cpe.Conv.Bias")]
    [InlineData("enc.enc1.down.proj.weight", "Encoder.Stage1.Down.Proj.Weight")]
    [InlineData("enc.enc1.down.norm.0.bias", "Encoder.Stage1.Down.Norm.Bias")]
    [InlineData("enc.enc0.block0.norm1.0.weight", "Encoder.Stage0.Block0.Norm1.Weight")]
    [InlineData("enc.enc0.block0.ls2.0.gamma", "Encoder.Stage0.Block0.LayerScale2.Gamma")]
    [InlineData("enc.enc0.block0.attn.qkv.weight", "Encoder.Stage0.Block0.Attn.Qkv.Weight")]
    [InlineData("enc.enc0.block0.attn.proj.bias", "Encoder.Stage0.Block0.Attn.Proj.Bias")]
    [InlineData("enc.enc0.block0.mlp.0.fc2.weight", "Encoder.Stage0.Block0.Mlp.Fc2.Weight")]
    [InlineData("dec.dec1.up.proj.0.weight", "Decoder.Stage1.Up.Proj.Linear.Weight")]
    [InlineData("dec.dec1.up.proj.1.bias", "Decoder.Stage1.Up.Proj.Norm.Bias")]
    [InlineData("dec.dec0.up.proj_skip.0.weight", "Decoder.Stage0.Up.Skip.Linear.Weight")]
    [InlineData("dec.dec0.up.proj_skip.1.weight", "Decoder.Stage0.Up.Skip.Norm.Weight")]
    [InlineData("dec.dec0.block2.cpe.2.bias", "Decoder.Stage0.Block2.Cpe.Norm.Bias")]
    [InlineData("embedding.stem.linear.weight", "Embedding.Linear.Weight")]
    [InlineData("embedding.stem.norm.bias", "Embedding.Norm.Bias")]
    public void ToCSharpName_MapsStateDictKeysToStableNames(string torchKey, string csharpName)
    {
        Assert.Equal(csharpName, UtoniaCheckpointKeys.ToCSharpName(torchKey));
    }

    [Theory]
    [InlineData("foo.bar")]
    [InlineData("embedding.mask_token")]
    [InlineData("enc.enc0.block0.attn.qkv")]
    [InlineData("enc.enc0.block0.cpe.3.weight")]
    [InlineData("enc.encX.block0.norm1.0.weight")]
    [InlineData("enc.enc0.block0.attn.rpe.rpe_table")]
    public void ToCSharpName_ReturnsNullForKeysTheEncoderDoesNotUse(string torchKey)
    {
        Assert.Null(UtoniaCheckpointKeys.ToCSharpName(torchKey));
    }

    [Fact]
    public void Shapes_FollowSpconvAndLinearLayouts()
    {
        Dictionary<string, UtoniaParameterSpec> byKey = UtoniaCheckpointKeys.Parameters(UtoniaTestData.TinyConfig())
            .ToDictionary(spec => spec.TorchKey, StringComparer.Ordinal);

        Assert.Equal(new long[] { 12, 3, 3, 3, 12 }, byKey["enc.enc0.block0.cpe.0.weight"].Shape);
        Assert.Equal(new long[] { 72, 24 }, byKey["enc.enc1.block0.attn.qkv.weight"].Shape);
        Assert.Equal(new long[] { 48, 24 }, byKey["enc.enc1.block0.mlp.0.fc1.weight"].Shape);
        Assert.Equal(new long[] { 36, 24 }, byKey["enc.enc2.down.proj.weight"].Shape);
    }

    [Theory]
    [InlineData(true, true)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    public void Encoder_ConsumesExactlyTheParametersTheConfigImplies(bool encoderOnly, bool qkvBias)
    {
        UtoniaConfig config = UtoniaTestData.TinyConfig(encoderOnly, qkvBias);
        UtoniaWeights weights = UtoniaTestData.RandomWeights(config, 3);

        using UtoniaEncoder encoder = new(new CpuBackend(), config, weights);

        HashSet<string> implied = UtoniaCheckpointKeys.Parameters(config).Select(spec => spec.CSharpName).ToHashSet(StringComparer.Ordinal);
        Assert.True(implied.SetEquals(weights.Consumed));
        Assert.True(implied.SetEquals(weights.Provided));
    }

    [Fact]
    public void Encoder_NamesAMissingParameterInTheError()
    {
        UtoniaConfig config = UtoniaTestData.TinyConfig();
        Dictionary<string, Tensor> tensors = UtoniaTestData.RandomTensors(config, 5);
        tensors.Remove("Encoder.Stage0.Block0.Attn.Qkv.Weight");

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() =>
            new UtoniaEncoder(new CpuBackend(), config, new UtoniaWeights(tensors)));
        Assert.Contains("Encoder.Stage0.Block0.Attn.Qkv.Weight", error.Message, StringComparison.Ordinal);
    }
}
