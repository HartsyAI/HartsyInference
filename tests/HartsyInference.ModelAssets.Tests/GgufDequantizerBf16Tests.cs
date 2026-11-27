using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>The BF16 decode of a GGUF block quant: every value is the F32 decode rounded to nearest, ties to even, across the parallel ranges.</summary>
public sealed class GgufDequantizerBf16Tests
{
    [Theory]
    [InlineData("Q2_K")]
    [InlineData("Q3_K")]
    [InlineData("Q4_0")]
    [InlineData("Q4_1")]
    [InlineData("Q4_K")]
    [InlineData("Q5_0")]
    [InlineData("Q5_1")]
    [InlineData("Q5_K")]
    [InlineData("Q6_K")]
    [InlineData("Q8_0")]
    public void EveryValueIsTheF32DecodeRoundedToNearest(string name)
    {
        DType[] types = [DType.Q2_K, DType.Q3_K, DType.Q4_0, DType.Q4_1, DType.Q4_K, DType.Q5_0, DType.Q5_1, DType.Q5_K, DType.Q6_K, DType.Q8_0];
        DType dtype = types.Single(candidate => candidate.Name == name);
        // More blocks than two parallel ranges and not a whole number of them. Random bytes decode to some values, NaN scales included.
        const int Blocks = 2500;
        using Tensor packed = new(new TensorShape(Blocks, dtype.BlockElementCount), dtype);
        new Random(name.Length * 31 + dtype.BlockByteSize).NextBytes(packed.AsSpan<byte>());

        using Tensor wide = GgufDequantizer.Dequantize(packed, DType.F32);
        using Tensor narrow = GgufDequantizer.Dequantize(packed, DType.BF16);

        ReadOnlySpan<float> values = wide.AsReadOnlySpan<float>();
        ReadOnlySpan<ushort> bits = narrow.AsReadOnlySpan<ushort>();
        Assert.Equal(values.Length, bits.Length);
        for (int i = 0; i < values.Length; i++)
            Assert.Equal(BFloat16Bits.FromFloat(values[i]), bits[i]);
    }
}
