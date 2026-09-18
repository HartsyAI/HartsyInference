using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Covers the quantization companions a fused-projection split has to carry. <c>int8_tensorwise</c>'s
/// <c>.weight_scale</c> is one float per output row, so splitting a <c>[3N, K]</c> QKV weight without splitting the
/// scale gives K and V rows Q's scales — every value off by the ratio between two layers' dynamic ranges, which is
/// noise, not a crash. Dropping the companion entirely (what happened before) is worse still: the backend then
/// consumes raw int8 at scale 1.</summary>
public sealed unsafe class SplitQkvRowScaleTests
{
    private static Tensor Int8Weight(int rows, int columns)
    {
        Tensor weight = new Tensor(new TensorShape(rows, columns), DType.I8);
        Span<sbyte> values = weight.AsSpan<sbyte>();
        for (int i = 0; i < values.Length; i++) values[i] = (sbyte)(i % 127);
        return weight;
    }

    private static Tensor RowScale(int rows)
    {
        Tensor scale = new Tensor(new TensorShape(rows, 1), DType.F32);
        Span<float> values = scale.AsSpan<float>();
        for (int i = 0; i < rows; i++) values[i] = 0.001f * (i + 1);
        return scale;
    }

    private static void DisposeAll(Dictionary<string, Tensor> weights)
    {
        foreach (Tensor tensor in weights.Values) tensor.Dispose();
    }

    [Fact]
    public void SplitQkvWeight_GivesEachProjectionItsOwnRowScales()
    {
        const int innerDim = 4;
        using Tensor fused = Int8Weight(3 * innerDim, 256);
        using Tensor rowScale = RowScale(3 * innerDim);
        fused.QuantInfo = new QuantWeightInfo
        {
            Format = "int8_tensorwise",
            RowScale = rowScale,
            ConvRotGroupSize = 256,
        };

        Dictionary<string, Tensor> output = new();
        CheckpointConvertUtils.SplitQkvWeight(fused, innerDim, "blocks.0.attn", "to_q", "to_k", "to_v", output);
        try
        {
            foreach ((string name, int firstRow) in new[] { ("to_q", 0), ("to_k", innerDim), ("to_v", 2 * innerDim) })
            {
                Tensor split = output[$"blocks.0.attn.{name}.weight"];
                QuantWeightInfo info = Assert.IsType<QuantWeightInfo>(split.QuantInfo);
                Assert.Equal("int8_tensorwise", info.Format);
                // ConvRot rotates along the INPUT dimension, which a row split leaves alone.
                Assert.Equal(256, info.ConvRotGroupSize);
                Assert.Equal(innerDim, (int)info.RowScale!.Shape[0]);
                ReadOnlySpan<float> scales = info.RowScale.AsReadOnlySpan<float>();
                for (int row = 0; row < innerDim; row++)
                    Assert.Equal(0.001f * (firstRow + row + 1), scales[row], 6);
            }
        }
        finally
        {
            DisposeAll(output);
        }
    }

    [Fact]
    public void SplitQkvWeight_SharesAPerTensorScaleAcrossAllThree()
    {
        const int innerDim = 4;
        using Tensor fused = Int8Weight(3 * innerDim, 256);
        using Tensor perTensor = new Tensor(new TensorShape(1), DType.F32);
        perTensor.AsSpan<float>()[0] = 0.0125f;
        fused.QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = perTensor };

        Dictionary<string, Tensor> output = new();
        CheckpointConvertUtils.SplitQkvWeight(fused, innerDim, "blocks.0.attn", "to_q", "to_k", "to_v", output);
        try
        {
            foreach (string name in new[] { "to_q", "to_k", "to_v" })
                Assert.Same(perTensor, output[$"blocks.0.attn.{name}.weight"].QuantInfo!.RowScale);
        }
        finally
        {
            DisposeAll(output);
        }
    }

    [Fact]
    public void SplitQkvWeight_CarriesBothFp8ScalarsOntoEverySplit()
    {
        const int innerDim = 4;
        using Tensor fused = new Tensor(new TensorShape(3 * innerDim, 64), DType.F8E4M3)
        {
            Fp8ScaleFactor = 0.0195f,
            Fp8InputScaleFactor = 0.0625f,
        };

        Dictionary<string, Tensor> output = new();
        CheckpointConvertUtils.SplitQkvWeight(fused, innerDim, "blocks.0.attn", "to_q", "to_k", "to_v", output);
        try
        {
            foreach (string name in new[] { "to_q", "to_k", "to_v" })
            {
                Tensor split = output[$"blocks.0.attn.{name}.weight"];
                Assert.Equal(0.0195f, split.Fp8ScaleFactor);
                // Without this the split silently reverts to a per-call activation absmax.
                Assert.Equal(0.0625f, split.Fp8InputScaleFactor);
            }
        }
        finally
        {
            DisposeAll(output);
        }
    }

    [Fact]
    public void SplitQkvWeight_RefusesAnNvfp4FusedWeightByName()
    {
        const int innerDim = 4;
        using Tensor fused = Int8Weight(3 * innerDim, 256);
        using Tensor blockScale = new Tensor(new TensorShape(128, 16), DType.F8E4M3);
        using Tensor globalScale = new Tensor(new TensorShape(1), DType.F32);
        fused.QuantInfo = new QuantWeightInfo { Format = "nvfp4", BlockScale = blockScale, GlobalScale = globalScale };

        Dictionary<string, Tensor> output = new();
        NotSupportedException error = Assert.Throws<NotSupportedException>(
            () => CheckpointConvertUtils.SplitQkvWeight(fused, innerDim, "blocks.0.attn", "to_q", "to_k", "to_v", output));
        Assert.Contains("blocks.0.attn.to_q.weight", error.Message);
        DisposeAll(output);
    }

    [Fact]
    public void SwapScaleShiftHalves_RefusesAWeightWithPerRowScales()
    {
        using Tensor table = Int8Weight(8, 256);
        using Tensor rowScale = RowScale(8);
        table.QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = rowScale };

        NotSupportedException error = Assert.Throws<NotSupportedException>(
            () => CheckpointConvertUtils.SwapScaleShiftHalves(table));
        Assert.Contains("int8_tensorwise", error.Message);
    }
}
