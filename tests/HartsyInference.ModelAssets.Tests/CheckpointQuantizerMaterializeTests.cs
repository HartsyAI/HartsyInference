using System.Reflection;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.ModelAssets.Quant;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Regression gate for <see cref="CheckpointQuantizer"/>'s widening of block-scaled weights to F32.
/// <para>Each expected value is computed here from the format definition, not from the codec's own tables. A
/// block-scaled weight must decode through its recipe; widening it as its raw stored dtype is silent corruption.</para></summary>
public sealed class CheckpointQuantizerMaterializeTests
{
    private static readonly float[] Fp4Magnitudes = [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f];

    private static readonly MethodInfo MaterializeMethod = typeof(CheckpointQuantizer).GetMethod(
        "MaterializeF32", BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new InvalidOperationException("CheckpointQuantizer.MaterializeF32 not found.");

    private static double ScalarFp4(int nibble) =>
        nibble == 8 ? 0.0 : ((nibble & 8) != 0 ? -1.0 : 1.0) * Fp4Magnitudes[nibble & 7];

    private static double ScalarE4M3(byte b)
    {
        int exp = (b >> 3) & 0xF, man = b & 7;
        double magnitude = exp == 0 ? man / 8.0 * Math.Pow(2, -6) : (1 + man / 8.0) * Math.Pow(2, exp - 7);
        return (b & 0x80) != 0 ? -magnitude : magnitude;
    }

    private static Tensor Materialize(Tensor weight, string key, List<Tensor> owned)
    {
        try
        {
            return (Tensor)MaterializeMethod.Invoke(null, [weight, key, owned])!;
        }
        catch (TargetInvocationException ex) when (ex.InnerException is not null)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
            throw;
        }
    }

    [Fact]
    public void Mxfp4_Block_Weight_Widens_Through_Its_Recipe()
    {
        const int rows = 3, cols = 64;
        Random rng = new(21);
        using Tensor scale = new(new TensorShape(rows, cols / 32), DType.F8E8M0);
        Span<byte> scaleBytes = scale.AsSpan<byte>();
        for (int i = 0; i < scaleBytes.Length; i++) scaleBytes[i] = (byte)rng.Next(118, 130);
        using Tensor weight = new(new TensorShape(rows, cols / 2), DType.I8);
        rng.NextBytes(weight.AsSpan<byte>());
        QuantRecipe recipe = new()
        {
            Encoding = QuantEncoding.Mxfp4E8M0, Geometry = new BlockGeometry(1, 32), ScaleDType = scale.DType,
            LogicalRows = rows, LogicalCols = cols, Scale = scale,
        };
        weight.QuantInfo = new QuantWeightInfo { Format = recipe.FormatName, Recipe = recipe };
        List<Tensor> owned = new();

        try
        {
            Tensor wide = Materialize(weight, "experts.0.w1.weight", owned);

            Assert.Equal(DType.F32, wide.DType);
            ReadOnlySpan<byte> packed = weight.AsReadOnlySpan<byte>();
            ReadOnlySpan<float> values = wide.AsReadOnlySpan<float>();
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                {
                    byte b = packed[r * cols / 2 + c / 2];
                    int nibble = c % 2 == 0 ? b & 0xF : b >> 4;
                    double expected = ScalarFp4(nibble) * Math.Pow(2, scaleBytes[r * (cols / 32) + c / 32] - 127);
                    Assert.Equal((float)expected, values[r * cols + c]);
                }
        }
        finally
        {
            foreach (Tensor t in owned) t.Dispose();
        }
    }

    [Fact]
    public void Fp8_Block_Weight_Widens_Through_Its_Recipe_Not_As_Raw_E4M3()
    {
        const int rows = 4, cols = 64, blockRows = 2, blockCols = 32;
        Random rng = new(22);
        using Tensor scale = new(new TensorShape(rows / blockRows, cols / blockCols), DType.F8E8M0);
        Span<byte> scaleBytes = scale.AsSpan<byte>();
        for (int i = 0; i < scaleBytes.Length; i++) scaleBytes[i] = (byte)rng.Next(120, 128);
        using Tensor weight = new(new TensorShape(rows, cols), DType.F8E4M3);
        rng.NextBytes(weight.AsSpan<byte>());
        QuantRecipe recipe = new()
        {
            Encoding = QuantEncoding.Fp8E4M3BlockE8M0, Geometry = new BlockGeometry(blockRows, blockCols),
            ScaleDType = scale.DType, LogicalRows = rows, LogicalCols = cols, Scale = scale,
        };
        weight.QuantInfo = new QuantWeightInfo { Format = recipe.FormatName, Recipe = recipe };
        List<Tensor> owned = new();

        try
        {
            Tensor wide = Materialize(weight, "dense.weight", owned);

            Assert.Equal(DType.F32, wide.DType);
            ReadOnlySpan<byte> bytes = weight.AsReadOnlySpan<byte>();
            ReadOnlySpan<float> values = wide.AsReadOnlySpan<float>();
            int scaleCols = cols / blockCols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                {
                    double blockScale = Math.Pow(2, scaleBytes[(r / blockRows) * scaleCols + c / blockCols] - 127);
                    double expected = ScalarE4M3(bytes[r * cols + c]) * blockScale;
                    Assert.Equal((float)expected, values[r * cols + c], 6);
                }
        }
        finally
        {
            foreach (Tensor t in owned) t.Dispose();
        }
    }

    [Fact]
    public void Plain_F32_Weight_Passes_Through_Unchanged()
    {
        using Tensor weight = new(new TensorShape(2, 3), DType.F32);
        weight.AsSpan<float>().Fill(1.5f);
        List<Tensor> owned = new();

        Tensor wide = Materialize(weight, "plain.weight", owned);

        Assert.Same(weight, wide);
        Assert.Empty(owned);
    }

    [Fact]
    public void Bf16_Weight_Widens_To_F32_Exactly()
    {
        using Tensor weight = new(new TensorShape(1, 2), DType.BF16);
        ushort[] bits = [0x3FC0, 0xBF80];
        bits.AsSpan().CopyTo(weight.AsSpan<ushort>());
        List<Tensor> owned = new();
        try
        {
            Tensor wide = Materialize(weight, "bf16.weight", owned);

            Assert.Equal(DType.F32, wide.DType);
            Assert.Equal(1.5f, wide.AsReadOnlySpan<float>()[0]);
            Assert.Equal(-1.0f, wide.AsReadOnlySpan<float>()[1]);
        }
        finally
        {
            foreach (Tensor t in owned) t.Dispose();
        }
    }
}
