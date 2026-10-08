using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.ModelAssets.BlockScale;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>A weight kept in its stored form must give exactly what the same weight widened to F32 gives.</summary>
public sealed class DeepSeekV41WeightTests : IDisposable
{
    private readonly List<Tensor> _owned = [];

    public void Dispose()
    {
        foreach (Tensor tensor in _owned) tensor.Dispose();
    }

    // random finite E4M3 bytes (NaN excluded) with in-range E8M0 scales, laid out like the V4.1 dense weights
    private (Tensor Weight, QuantWeightInfo Quant) Fp8(int rows, int cols, int seed)
    {
        Random rng = new(seed);
        Tensor weight = new(new TensorShape(rows, cols), DType.F8E4M3);
        Span<byte> bytes = weight.AsSpan<byte>();
        for (int i = 0; i < bytes.Length; i++) bytes[i] = (byte)(rng.Next(256) & 0xF7);
        BlockGeometry geometry = new(32, 32);
        (long scaleRows, long scaleCols) = geometry.ScaleShape(rows, cols);
        Tensor scale = new(new TensorShape(scaleRows, scaleCols), DType.F8E8M0);
        Span<byte> scales = scale.AsSpan<byte>();
        for (int i = 0; i < scales.Length; i++) scales[i] = (byte)rng.Next(110, 130);
        _owned.Add(weight);
        _owned.Add(scale);
        QuantRecipe recipe = new() { Encoding = QuantEncoding.Fp8E4M3BlockE8M0, Geometry = geometry, ScaleDType = scale.DType, LogicalRows = rows, LogicalCols = cols, Scale = scale };
        return (weight, new QuantWeightInfo { Format = recipe.FormatName, Recipe = recipe });
    }

    private static float[] Input(int count, int seed)
    {
        Random rng = new(seed);
        float[] x = new float[count];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() - 0.5);
        return x;
    }

    [Theory]
    [InlineData(40, 96, 3)]
    [InlineData(1100, 4096, 2)] // more than one decode window
    public void StoredFp8_LinearEqualsTheWidenedProductBitForBit(int rows, int cols, int tokens)
    {
        (Tensor weight, QuantWeightInfo quant) = Fp8(rows, cols, 5);
        float[] x = Input(tokens * cols, 6);
        DeepSeekV41Weight stored = DeepSeekV41Weight.FromStored("w", weight, quant, rows, cols);
        DeepSeekV41Weight widened = DeepSeekV41Weight.FromF32(WeightDequantizer.ToF32(weight, quant));

        Assert.False(stored.IsWidened);
        Assert.Equal(widened.Linear(x, tokens, cols, rows), stored.Linear(x, tokens, cols, rows));
        Assert.True(stored.ResidentBytes < widened.ResidentBytes / 3);
    }

    [Fact]
    public void StoredBf16_LinearRowsAndCopyRowMatchTheWidenedValues()
    {
        const int rows = 70, cols = 64;
        Random rng = new(9);
        Tensor weight = new(new TensorShape(rows, cols), DType.BF16);
        _owned.Add(weight);
        Span<ushort> bits = weight.AsSpan<ushort>();
        for (int i = 0; i < bits.Length; i++) bits[i] = (ushort)(BitConverter.SingleToUInt32Bits((float)(rng.NextDouble() - 0.5)) >> 16);
        float[] widenedValues = WeightDequantizer.ToF32(weight, null);
        DeepSeekV41Weight stored = DeepSeekV41Weight.FromStored("w", weight, null, rows, cols), widened = widenedValues;
        float[] x = Input(2 * cols, 10);

        Assert.Equal(widened.Linear(x, 2, cols, rows), stored.Linear(x, 2, cols, rows));
        float[] row = new float[cols];
        stored.CopyRow(33, row);
        Assert.Equal(widenedValues.AsSpan(33 * cols, cols).ToArray(), row);
        float[] scratch = new float[5 * cols];
        Assert.Equal(widenedValues.AsSpan(60 * cols, 5 * cols).ToArray(), stored.ReadRows(60, 5, cols, scratch).ToArray());
    }

    [Fact]
    public void GroupedProjection_OnAStoredWeightEqualsTheWidenedProjection()
    {
        const int groups = 4, rank = 40, groupDim = 96, tokens = 2;
        (Tensor weight, QuantWeightInfo quant) = Fp8(groups * rank, groupDim, 12);
        float[] x = Input(tokens * groups * groupDim, 13);
        float[] viaStored = new float[tokens * groups * rank], viaWidened = new float[viaStored.Length];

        DeepSeekV41GroupedProjection.Apply(x, DeepSeekV41Weight.FromStored("w", weight, quant, groups * rank, groupDim), tokens, groups, rank, groupDim, viaStored);
        DeepSeekV41GroupedProjection.Apply(x, WeightDequantizer.ToF32(weight, quant), tokens, groups, rank, groupDim, viaWidened);

        Assert.Equal(viaWidened, viaStored);
    }

    [Fact]
    public void AStoredWeightOfTheWrongShapeIsRefusedByName()
    {
        (Tensor weight, QuantWeightInfo quant) = Fp8(64, 64, 14);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => DeepSeekV41Weight.FromStored("layers.0.attn.wkv.weight", weight, quant, 64, 128));

        Assert.Contains("layers.0.attn.wkv.weight", error.Message);
        Assert.Contains("[64, 64]", error.Message);
    }

    [Fact]
    public void ANullFloatArrayConvertsToANullWeight()
    {
        float[]? none = null;
        DeepSeekV41Weight? weight = none;

        Assert.Null(weight);
    }

    [Fact]
    public void AStoredWeightRefusesAProductOfAnotherShape()
    {
        (Tensor weight, QuantWeightInfo quant) = Fp8(64, 64, 15);
        DeepSeekV41Weight stored = DeepSeekV41Weight.FromStored("w", weight, quant, 64, 64);

        Assert.Throws<ArgumentException>(() => stored.Linear(new float[32], 1, 32, 64));
    }
}
