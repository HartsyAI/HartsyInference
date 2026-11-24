using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

public sealed class LongCatInt8ConverterTests
{
    private static Tensor Make(DType dtype, params long[] dims) => new(new TensorShape(dims), dtype);

    private static Dictionary<string, Tensor> Source() => new()
    {
        ["blocks.0.attn.qkv.weight_int8"] = Make(DType.I8, 6, 4),
        ["blocks.0.attn.qkv.weight_scale"] = Make(DType.F32, 6),
        ["blocks.0.attn.qkv.bias"] = Make(DType.BF16, 6),
        ["final_layer.linear.weight"] = Make(DType.BF16, 4, 4),
        ["x_embedder.proj.weight"] = Make(DType.BF16, 4, 2, 1, 2, 2),
    };

    [Fact]
    public void QuantizedWeightsTakeTheDenseKeyAndCarryTheirRowScale()
    {
        Dictionary<string, Tensor> source = Source();
        Tensor scale = source["blocks.0.attn.qkv.weight_scale"];

        Dictionary<string, Tensor> converted = LongCatInt8Converter.Convert(source);

        Assert.Equal(["blocks.0.attn.qkv.bias", "blocks.0.attn.qkv.weight", "final_layer.linear.weight", "x_embedder.proj.weight"],
            converted.Keys.Order(StringComparer.Ordinal));
        Tensor weight = converted["blocks.0.attn.qkv.weight"];
        Assert.Equal(DType.I8, weight.DType);
        Assert.Equal("int8_tensorwise", weight.QuantInfo!.Format);
        Assert.Same(scale, weight.QuantInfo.RowScale);
        Assert.Equal(0, weight.QuantInfo.ConvRotGroupSize);
        Assert.Null(converted["final_layer.linear.weight"].QuantInfo);
        Assert.Null(converted["x_embedder.proj.weight"].QuantInfo);
    }

    [Fact]
    public void ADenseCheckpointPassesThroughUntouched()
    {
        Dictionary<string, Tensor> dense = new() { ["a.weight"] = Make(DType.F32, 2, 2), ["a.bias"] = Make(DType.F32, 2) };

        Assert.False(LongCatInt8Converter.IsInt8Layout(dense.Keys));
        Assert.Equal(dense.Keys.Order(), LongCatInt8Converter.Convert(dense).Keys.Order());
        Assert.True(LongCatInt8Converter.IsInt8Layout(Source().Keys));
    }

    [Fact]
    public void ASingleScaleIsAcceptedAsPerTensor()
    {
        Dictionary<string, Tensor> source = Source();
        source["blocks.0.attn.qkv.weight_scale"] = Make(DType.F32, 1);

        Assert.NotNull(LongCatInt8Converter.Convert(source)["blocks.0.attn.qkv.weight"].QuantInfo);
    }

    [Fact]
    public void MissingScaleIsRefused()
    {
        Dictionary<string, Tensor> source = Source();
        source.Remove("blocks.0.attn.qkv.weight_scale");

        Assert.Contains("dequant scale", Assert.Throws<InvalidDataException>(() => LongCatInt8Converter.Convert(source)).Message);
    }

    [Fact]
    public void ScaleOfTheWrongTypeOrCountIsRefused()
    {
        Dictionary<string, Tensor> wrongType = Source();
        wrongType["blocks.0.attn.qkv.weight_scale"] = Make(DType.BF16, 6);
        Dictionary<string, Tensor> wrongCount = Source();
        wrongCount["blocks.0.attn.qkv.weight_scale"] = Make(DType.F32, 5);

        Assert.Throws<InvalidDataException>(() => LongCatInt8Converter.Convert(wrongType));
        Assert.Throws<InvalidDataException>(() => LongCatInt8Converter.Convert(wrongCount));
    }

    [Fact]
    public void NonI8OrNonMatrixQuantizedNameIsRefused()
    {
        Dictionary<string, Tensor> dense = Source();
        dense["blocks.0.attn.qkv.weight_int8"] = Make(DType.BF16, 6, 4);
        Dictionary<string, Tensor> vector = Source();
        vector["blocks.0.attn.qkv.weight_int8"] = Make(DType.I8, 24);

        Assert.Throws<InvalidDataException>(() => LongCatInt8Converter.Convert(dense));
        Assert.Throws<InvalidDataException>(() => LongCatInt8Converter.Convert(vector));
    }

    [Fact]
    public void BothSpellingsOfOneWeightAreRefused()
    {
        Dictionary<string, Tensor> source = Source();
        source["blocks.0.attn.qkv.weight"] = Make(DType.BF16, 6, 4);

        Assert.Throws<InvalidDataException>(() => LongCatInt8Converter.Convert(source));
    }
}
