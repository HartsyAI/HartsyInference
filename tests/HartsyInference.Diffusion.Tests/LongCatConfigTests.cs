using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

public sealed class LongCatConfigTests
{
    /// <summary>The base repository's <c>dit/config.json</c> at 03b55529 (the avatar's differs only by its audio fields).</summary>
    private const string ShippedConfig = """
        {"_class_name":"LongCatVideoTransformer3DModel","adaln_tembed_dim":512,"caption_channels":4096,"cp_split_hw":null,"depth":48,
         "enable_bsa":false,"frequency_embedding_size":256,"hidden_size":4096,"in_channels":16,"text_tokens_zero_pad":true,"mlp_ratio":4,
         "num_heads":32,"out_channels":16,"patch_size":[1,2,2]}
        """;

    [Fact]
    public void ShippedConfigParsesToTheDefaults()
    {
        LongCatConfig config = LongCatConfig.FromJson(ShippedConfig);

        Assert.Equal(new LongCatConfig(), config);
        Assert.Equal(128, config.HeadDim);
        Assert.Equal(11008, config.FfnHiddenDim);
        Assert.Equal(64, config.PatchVector);
    }

    [Fact]
    public void SwiGluWidthRoundsUpToTheNext256()
    {
        Assert.Equal(512, LongCatTestData.Tiny.FfnHiddenDim);
        Assert.Equal(11008, new LongCatConfig().FfnHiddenDim);
    }

    [Fact]
    public void ZeroPadFlagFollowsTheJson()
    {
        Assert.False(LongCatConfig.FromJson("""{"text_tokens_zero_pad":false}""").TextTokensZeroPad);
        Assert.True(LongCatConfig.FromJson("{}").TextTokensZeroPad);
    }

    [Fact]
    public void ShapesGiveTheSameGeometryAsTheConfig()
    {
        LongCatConfig tiny = LongCatTestData.Tiny;
        Dictionary<string, TensorShape> shapes = LongCatCheckpoint.ExpectedShapes(tiny);

        LongCatConfig derived = LongCatConfig.FromShapes(name => shapes.TryGetValue(name, out TensorShape s) ? s : null, tiny.Depth, tiny.PatchSize, true);

        Assert.Equal(tiny, derived);
    }

    [Fact]
    public void HeadWidthMustBeAMultipleOfEight()
    {
        Assert.Throws<ArgumentException>(() => LongCatConfig.FromJson("""{"hidden_size":120,"num_heads":8}"""));
    }

    [Fact]
    public void ValidateNamesWhatIsMissingOrMisshapen()
    {
        LongCatConfig tiny = LongCatTestData.Tiny;
        Dictionary<string, TensorShape> shapes = LongCatCheckpoint.ExpectedShapes(tiny);
        LongCatCheckpoint.Validate(tiny, shapes);

        shapes.Remove("blocks.1.ffn.w2.weight");
        shapes["blocks.0.attn.qkv.weight"] = new TensorShape(tiny.HiddenSize, tiny.HiddenSize);

        InvalidDataException error = Assert.Throws<InvalidDataException>(() => LongCatCheckpoint.Validate(tiny, shapes));
        Assert.Contains("blocks.0.attn.qkv.weight", error.Message);
        Assert.Contains("blocks.1.ffn.w2.weight", error.Message);
    }
}
