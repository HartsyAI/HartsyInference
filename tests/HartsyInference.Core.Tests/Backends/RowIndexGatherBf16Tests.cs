using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>GatherRows with a BF16 table: the rows are widened to F32 exactly, the F32 path is unchanged, and the contract refuses a non-F32
/// output or an unsupported input.</summary>
public sealed class RowIndexGatherBf16Tests
{
    [Fact]
    public void Bf16Table_GathersRowsWidenedExactly()
    {
        const int rows = 6, width = 5;
        ushort[] table = new ushort[rows * width];
        for (int i = 0; i < table.Length; i++) table[i] = TensorCasts.F32ToBf16Bits(0.25f * i - 3f);
        int[] index = [4, 0, 4, 5];
        using Tensor input = new(new TensorShape(rows, width), DType.BF16);
        for (int i = 0; i < table.Length; i++) input.AsSpan<ushort>()[i] = table[i];
        using Tensor output = new(new TensorShape(index.Length, width), DType.F32);
        using IBackend backend = new CpuBackend();

        backend.GatherRows(output, input, index);

        for (int m = 0; m < index.Length; m++)
            for (int j = 0; j < width; j++)
            {
                float expected = BFloat16Bits.ToFloat(table[index[m] * width + j]);
                Assert.Equal(expected, output.AsSpan<float>()[m * width + j]);
            }
    }

    [Fact]
    public void F32Table_IsUnchanged()
    {
        const int rows = 4, width = 3;
        float[] table = Enumerable.Range(0, rows * width).Select(i => i * 0.5f).ToArray();
        int[] index = [3, 1];
        using Tensor input = new(new TensorShape(rows, width), DType.F32);
        table.AsSpan().CopyTo(input.AsSpan<float>());
        using Tensor output = new(new TensorShape(index.Length, width), DType.F32);
        using IBackend backend = new CpuBackend();

        backend.GatherRows(output, input, index);

        for (int m = 0; m < index.Length; m++)
            for (int j = 0; j < width; j++)
                Assert.Equal(table[index[m] * width + j], output.AsSpan<float>()[m * width + j]);
    }

    [Fact]
    public void Bf16Output_IsRefused()
    {
        using Tensor input = new(new TensorShape(2, 4), DType.BF16);
        using Tensor output = new(new TensorShape(1, 4), DType.BF16);
        using Tensor index = IndexTensor.Rows([1]);
        using IBackend backend = new CpuBackend();

        Assert.Throws<NotSupportedException>(() => backend.GatherRows(output, input, index));
    }

    [Fact]
    public void Fp8Input_IsRefused()
    {
        using Tensor input = new(new TensorShape(2, 4), DType.F8E4M3);
        using Tensor output = new(new TensorShape(1, 4), DType.F32);
        using Tensor index = IndexTensor.Rows([1]);
        using IBackend backend = new CpuBackend();

        Assert.Throws<NotSupportedException>(() => backend.GatherRows(output, input, index));
    }
}
