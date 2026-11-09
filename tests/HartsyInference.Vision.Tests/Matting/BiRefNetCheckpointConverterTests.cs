using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;

namespace HartsyInference.Vision.Tests.Matting;

public sealed class BiRefNetCheckpointConverterTests
{
    [Fact]
    public void Convert_SplitsFusedQkvIntoQueryKeyValueRowBlocks()
    {
        const int channels = 4;
        Tensor fusedW = Filled(new TensorShape(3 * channels, channels), i => i);
        Tensor fusedB = Filled(new TensorShape(3 * channels), i => 100 + i);
        Dictionary<string, Tensor> raw = new()
        {
            ["bb.layers.0.blocks.0.attn.qkv.weight"] = fusedW,
            ["bb.layers.0.blocks.0.attn.qkv.bias"] = fusedB,
        };

        Dictionary<string, Tensor> converted = BiRefNetCheckpointConverter.Convert(raw);

        Assert.DoesNotContain("bb.layers.0.blocks.0.attn.qkv.weight", converted.Keys);
        Tensor q = converted["bb.layers.0.blocks.0.attn.q.weight"];
        Tensor k = converted["bb.layers.0.blocks.0.attn.k.weight"];
        Tensor v = converted["bb.layers.0.blocks.0.attn.v.weight"];
        Assert.Equal(new long[] { channels, channels }, Dims(q));
        Assert.Equal(0f, At(q, 0, 0));
        Assert.Equal((float)(channels * channels), At(k, 0, 0));
        Assert.Equal((float)(2 * channels * channels), At(v, 0, 0));
        Assert.Equal((float)(3 * channels * channels - 1), At(v, channels - 1, channels - 1));
        Assert.Equal(100f + channels, At(converted["bb.layers.0.blocks.0.attn.k.bias"], 0));
        Assert.Equal(100f + 2 * channels, At(converted["bb.layers.0.blocks.0.attn.v.bias"], 0));
    }

    [Fact]
    public void Convert_DropsOnlyDocumentedInferenceUnusedBuffers()
    {
        Dictionary<string, Tensor> raw = new()
        {
            ["bb.layers.0.blocks.0.attn.relative_position_index"] = Filled(new TensorShape(4), i => i),
            ["bb.layers.0.blocks.0.norm1.weight"] = Filled(new TensorShape(4), i => i),
            ["bb.patch_embed.norm.num_batches_tracked"] = Filled(new TensorShape(1), i => 0),
            ["decoder.conv_ms_spvn_4.weight"] = Filled(new TensorShape(1), i => 0),
            ["decoder.gdt_convs_pred_4.0.weight"] = Filled(new TensorShape(1), i => 0),
            ["decoder.gdt_convs_attn_4.0.weight"] = Filled(new TensorShape(1), i => 0),
            ["squeeze_module.0.conv_in.weight"] = Filled(new TensorShape(1), i => 0),
        };

        Dictionary<string, Tensor> converted = BiRefNetCheckpointConverter.Convert(raw);

        Assert.Equal(3, converted.Count);
        Assert.Contains("bb.layers.0.blocks.0.norm1.weight", converted.Keys);
        Assert.Contains("decoder.gdt_convs_attn_4.0.weight", converted.Keys);
        Assert.Contains("squeeze_module.0.conv_in.weight", converted.Keys);
    }

    [Fact]
    public void Convert_RejectsKeyThatIsNeitherRoutedNorDocumentedUnused()
    {
        Dictionary<string, Tensor> raw = new()
        {
            ["decoder.refiner.conv.weight"] = Filled(new TensorShape(1), i => 0),
        };

        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() => BiRefNetCheckpointConverter.Convert(raw));
        Assert.Contains("decoder.refiner.conv.weight", error.Message);
    }

    private static Tensor Filled(TensorShape shape, Func<int, float> value)
    {
        Tensor t = new(shape, DType.F32);
        Span<float> span = t.AsSpan<float>();
        for (int i = 0; i < span.Length; i++) span[i] = value(i);
        return t;
    }

    private static float At(Tensor t, int row, int col = 0)
    {
        long cols = t.Shape.Rank > 1 ? t.Shape[1] : 1;
        return t.AsReadOnlySpan<float>()[(int)(row * cols + col)];
    }

    private static long[] Dims(Tensor t)
    {
        long[] dims = new long[t.Shape.Rank];
        for (int i = 0; i < dims.Length; i++) dims[i] = t.Shape[i];
        return dims;
    }
}
