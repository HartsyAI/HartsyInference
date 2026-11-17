using System.Text.Json;
using HartsyInference.Audio.Models.Haru;
using HartsyInference.Audio.Models.VibeVoice;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests.VibeVoice;

public sealed class HaruCodecWeightsTests
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Haru");

    // The codec block of higurashi-ai/Haru-4B config.json @ 8d6e83bc.
    internal const string ReleasedCodec = """
        {"latent_dim":64,"base_channels":32,"ratios":[8,5,5,4,2,2],"encoder_depths":[3,3,3,3,3,3,8],
         "decoder_depths":[8,3,3,3,3,3,3],"norm_eps":1e-05,"latent_std":0.5}
        """;

    [Theory]
    [InlineData("codec.encoder.stem.weight", "codec.encoder.downsample_layers.0.0.conv.conv.weight")]
    [InlineData("codec.encoder.down.0.bias", "codec.encoder.downsample_layers.1.0.conv.conv.bias")]
    [InlineData("codec.encoder.down.5.weight", "codec.encoder.downsample_layers.6.0.conv.conv.weight")]
    [InlineData("codec.encoder.head.weight", "codec.encoder.head.conv.conv.weight")]
    [InlineData("codec.encoder.stages.6.7.conv.weight", "codec.encoder.stages.6.7.mixer.conv.conv.conv.weight")]
    [InlineData("codec.encoder.stages.2.1.fc1.bias", "codec.encoder.stages.2.1.ffn.linear1.bias")]
    [InlineData("codec.encoder.stages.2.1.fc2.weight", "codec.encoder.stages.2.1.ffn.linear2.weight")]
    [InlineData("codec.encoder.stages.0.0.ffn_gamma", "codec.encoder.stages.0.0.ffn_gamma")]
    [InlineData("codec.encoder.stages.0.0.norm.weight", "codec.encoder.stages.0.0.norm.weight")]
    [InlineData("codec.decoder.stem.bias", "codec.decoder.upsample_layers.0.0.conv.conv.bias")]
    [InlineData("codec.decoder.up.0.weight", "codec.decoder.upsample_layers.1.0.convtr.convtr.weight")]
    [InlineData("codec.decoder.up.5.bias", "codec.decoder.upsample_layers.6.0.convtr.convtr.bias")]
    [InlineData("codec.decoder.head.weight", "codec.decoder.head.conv.conv.weight")]
    [InlineData("codec.decoder.stages.0.7.gamma", "codec.decoder.stages.0.7.gamma")]
    public void RemapKey_MapsToTheEngineLayout(string haru, string engine) => Assert.Equal(engine, HaruCodecWeights.RemapKey(haru));

    [Theory]
    [InlineData("lm.layers.0.mlp.up_proj.weight")]
    [InlineData("codec.encoder.up.0.weight")]
    [InlineData("codec.decoder.down.0.weight")]
    [InlineData("codec.encoder.stem.scale")]
    [InlineData("codec.encoder.stages.x.0.fc1.weight")]
    [InlineData("codec.encoder.stages.1.0.fc3.weight")]
    [InlineData("codec.encoder.norm.weight")]
    [InlineData("codec.encoder.stages.01.0.fc1.weight")]
    [InlineData("codec.encoder.stages.+1.0.fc1.weight")]
    [InlineData("codec.encoder.stages. 1.0.fc1.weight")]
    [InlineData("codec.encoder.down.-1.weight")]
    public void RemapKey_RefusesAKeyOutsideTheLayout(string key) => Assert.Throws<InvalidDataException>(() => HaruCodecWeights.RemapKey(key));

    [Fact]
    public void Remap_MapsEveryCodecTensorOfTheCheckpointToADistinctKey()
    {
        using SafeTensorsLoader weights = new();
        weights.Load(Path.Combine(Dir, "haru_tiny.safetensors"));
        int codecTensors = weights.Descriptors.Keys.Count(static k => k.StartsWith("codec.", StringComparison.Ordinal));
        Assert.True(codecTensors > 100);
        Assert.Equal(codecTensors, HaruCodecWeights.Remap(weights.GetAllTensors()).Count);
    }

    [Theory]
    [InlineData("codec.encoder.down.9.weight")] // a seventh down layer: the config has six ratios
    [InlineData("codec.decoder.up.9.bias")]
    [InlineData("codec.encoder.stages.9.0.fc1.weight")] // a stage the config does not have
    [InlineData("codec.encoder.stages.0.5.fc1.weight")] // a block beyond the stage's depth
    public void Create_RefusesATensorOutsideTheConfiguredLayout(string extraKey)
    {
        using SafeTensorsLoader weights = new();
        weights.Load(Path.Combine(Dir, "haru_tiny.safetensors"));
        Dictionary<string, Tensor> all = new(weights.GetAllTensors(), StringComparer.Ordinal);
        all[extraKey] = all["codec.encoder.stem.weight"];
        using JsonDocument config = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "config.json")));
        InvalidDataException error = Assert.Throws<InvalidDataException>(() => HaruCodec.Create(config.RootElement.GetProperty("codec"), all));
        Assert.Contains("codec tensors", error.Message);
    }

    [Fact]
    public void ConfigFrom_RefusesAFieldItDoesNotRead()
    {
        using JsonDocument doc = JsonDocument.Parse(ReleasedCodec.Replace("\"latent_std\":0.5", "\"latent_std\":0.5,\"causal\":false"));
        InvalidDataException error = Assert.Throws<InvalidDataException>(() => HaruCodec.ConfigFrom(doc.RootElement));
        Assert.Contains("causal", error.Message);
    }

    [Fact]
    public void ConfigFrom_ReadsTheReleasedCodecBlock()
    {
        using JsonDocument doc = JsonDocument.Parse(ReleasedCodec);
        VibeVoiceTokenizerConfig config = HaruCodec.ConfigFrom(doc.RootElement);
        Assert.Equal(64, config.VaeDim);
        Assert.Equal(32, config.EncoderNFilters);
        Assert.Equal(32, config.DecoderNFilters);
        Assert.Equal([8, 5, 5, 4, 2, 2], config.EncoderRatios);
        Assert.Equal([8, 5, 5, 4, 2, 2], config.DecoderRatios);
        Assert.Equal("3-3-3-3-3-3-8", config.EncoderDepths);
        Assert.Equal("8-3-3-3-3-3-3", config.DecoderDepths);
        Assert.Equal(1e-5f, config.LayerNormEps);
        Assert.Equal(0.5f, config.FixStd);
        Assert.Equal("gaussian", config.StdDistType);
        Assert.True(config.DisableLastNorm);
    }

    [Theory]
    [InlineData("""{"latent_dim":64,"base_channels":32,"ratios":[8,5,5,4,2],"encoder_depths":[3,3,3,3,3,8],"decoder_depths":[8,3,3,3,3,3],"norm_eps":1e-05,"latent_std":0.5}""")]
    [InlineData("""{"latent_dim":64,"base_channels":32,"ratios":[8,5,5,4,2,2],"encoder_depths":[3,3,3,3,3,8],"decoder_depths":[8,3,3,3,3,3],"norm_eps":1e-05,"latent_std":0.5}""")]
    public void ConfigFrom_RefusesAFrameThatIsNot3200SamplesOrADepthListOfTheWrongLength(string json)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        Assert.Throws<InvalidDataException>(() => HaruCodec.ConfigFrom(doc.RootElement));
    }
}
