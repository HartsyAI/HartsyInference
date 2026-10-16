using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Registry;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>FLUX.3 key handling: the fused <c>qkv</c> split must carry quantization companions with its rows, the norm
/// <c>.scale</c> spelling must land on <c>.weight</c>, and a stray key must be refused rather than dropped.</summary>
public sealed class Flux3CheckpointConverterTests
{
    private const int Hidden = 4;

    [Fact]
    public void FusedQkv_SplitsIntoThirds_InRowOrder()
    {
        using Tensor fused = Iota(3 * Hidden, Hidden);
        Dictionary<string, Tensor> converted = Flux3CheckpointConverter.ConvertTransformer(new Dictionary<string, Tensor>
        {
            ["model.diffusion_model.single_blocks.0.qkv.weight"] = fused,
        });
        try
        {
            Assert.False(converted.ContainsKey("single_blocks.0.qkv.weight"));
            string[] names = ["q_proj", "k_proj", "v_proj"];
            for (int part = 0; part < 3; part++)
            {
                Tensor piece = converted[$"single_blocks.0.{names[part]}.weight"];
                Assert.Equal(new TensorShape(Hidden, Hidden), piece.Shape);
                Assert.Equal((float)(part * Hidden * Hidden), piece.AsSpan<float>()[0]);
            }
        }
        finally
        {
            DisposeAll(converted);
        }
    }

    [Fact]
    public void FusedQkv_CarriesFp8ScaleAndNarrowsInt8RowScale()
    {
        using Tensor rowScale = new Tensor(new TensorShape(3 * Hidden, 1), DType.F32);
        Span<float> scales = rowScale.AsSpan<float>();
        for (int i = 0; i < scales.Length; i++) scales[i] = i + 1;
        using Tensor fused = new Tensor(new TensorShape(3 * Hidden, Hidden), DType.I8)
        {
            Fp8ScaleFactor = 0.5f,
            QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = rowScale },
        };
        Dictionary<string, Tensor> converted = Flux3CheckpointConverter.ConvertTransformer(new Dictionary<string, Tensor>
        {
            ["content_mode_blocks.image.0.qkv.weight"] = fused,
        });
        try
        {
            Tensor k = converted["content_mode_blocks.image.0.k_proj.weight"];
            Assert.Equal(0.5f, k.Fp8ScaleFactor);
            Assert.NotNull(k.QuantInfo?.RowScale);
            Assert.Equal(Hidden + 1, k.QuantInfo!.RowScale!.AsSpan<float>()[0]);
        }
        finally
        {
            DisposeAll(converted);
        }
    }

    [Theory]
    [InlineData("txt_mode_blocks.0.norm.query_norm.scale", "txt_mode_blocks.0.norm.query_norm.weight")]
    [InlineData("single_blocks.3.norm.key_norm.scale", "single_blocks.3.norm.key_norm.weight")]
    [InlineData("txt_mode_blocks.0.norm.query_norm.weight", "txt_mode_blocks.0.norm.query_norm.weight")]
    [InlineData("diffusion_model.final_layer.image.linear.weight", "final_layer.image.linear.weight")]
    public void NormScale_AndPrefixes_Normalize(string key, string expected)
    {
        using Tensor tensor = new Tensor(new TensorShape(Hidden), DType.F32);
        Dictionary<string, Tensor> converted = Flux3CheckpointConverter.ConvertTransformer(new Dictionary<string, Tensor> { [key] = tensor });
        Assert.Equal([expected], converted.Keys);
    }

    [Fact]
    public void BundledComponents_AreLeftToTheirLoaders()
    {
        using Tensor tensor = new Tensor(new TensorShape(Hidden), DType.F32);
        Dictionary<string, Tensor> converted = Flux3CheckpointConverter.ConvertTransformer(new Dictionary<string, Tensor>
        {
            ["time_in.in_layer.weight"] = tensor,
            ["vae.decoder.conv_in.weight"] = tensor,
            ["text_encoders.qwen3vl_4b.transformer.model.norm.weight"] = tensor,
        });
        Assert.Equal(["time_in.in_layer.weight"], converted.Keys);
    }

    [Fact]
    public void UnknownKey_IsRefused()
    {
        using Tensor tensor = new Tensor(new TensorShape(Hidden), DType.F32);
        Assert.Throws<NotSupportedException>(() => Flux3CheckpointConverter.ConvertTransformer(
            new Dictionary<string, Tensor> { ["double_blocks.0.img_attn.qkv.weight"] = tensor }));
    }

    [Fact]
    public void UnfoldedCompanion_IsRefused()
    {
        using Tensor tensor = new Tensor(new TensorShape(Hidden), DType.F32);
        Assert.Throws<NotSupportedException>(() => Flux3CheckpointConverter.ConvertTransformer(
            new Dictionary<string, Tensor> { ["single_blocks.0.mlp_in.weight_scale"] = tensor }));
    }

    [Fact]
    public void Detector_RoutesFlux3_AndNotFlux1OrFlux2()
    {
        string[] keys =
        [
            "model.diffusion_model.content_mode_blocks.image.0.attn_out.weight",
            "model.diffusion_model.single_stream_modulations.txt.lin.weight",
            "model.diffusion_model.single_blocks.0.mlp_in.weight",
        ];
        Assert.Equal(ModelArchitecture.Flux3, ModelArchitectureDetector.Detect(keys));
        Assert.True(Flux3CheckpointConverter.MatchesByKeys(keys));
        Assert.False(Flux3CheckpointConverter.MatchesByKeys(["double_blocks.0.img_attn.qkv.weight", "single_blocks.0.linear1.weight"]));
    }

    private static Tensor Iota(int rows, int cols)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        Span<float> span = t.AsSpan<float>();
        for (int i = 0; i < span.Length; i++) span[i] = i;
        return t;
    }

    private static void DisposeAll(Dictionary<string, Tensor> weights)
    {
        foreach (Tensor t in weights.Values) t.Dispose();
    }
}
