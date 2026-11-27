using HartsyInference.Core.Backends;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>CUDA GatherRows with a BF16 table: the rows come back widened to F32 exactly (the host reference runs, never the F32 kernel on the
/// raw bytes), and TryGatherRowsResident refuses a BF16 table so the caller gathers on the host. Reported as skipped when no CUDA device opens.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class RowIndexGatherBf16CudaTests
{
    [CudaFact]
    public void Cuda_GatherRows_Bf16Table_WidensRowsExactly()
    {
        const int rows = 6, width = 5;
        int[] index = [4, 0, 4, 5];
        ushort[] table = new ushort[rows * width];
        for (int i = 0; i < table.Length; i++) table[i] = TensorCasts.F32ToBf16Bits(0.25f * i - 3f);
        using Tensor input = new(new TensorShape(rows, width), DType.BF16);
        table.AsSpan().CopyTo(input.AsSpan<ushort>());
        using Tensor output = new(new TensorShape(index.Length, width), DType.F32);
        using IBackend backend = new CudaBackend(0, CudaPtxDirectory.Resolve());

        backend.GatherRows(output, input, index);

        for (int m = 0; m < index.Length; m++)
            for (int j = 0; j < width; j++)
                Assert.Equal(BFloat16Bits.ToFloat(table[index[m] * width + j]), output.AsSpan<float>()[m * width + j]);
    }

    [CudaFact]
    public void Cuda_TryGatherRowsResident_RefusesBf16Table()
    {
        using Tensor input = new(new TensorShape(4, 3), DType.BF16);
        using Tensor output = new(new TensorShape(1, 3), DType.F32);
        using IBackend backend = new CudaBackend(0, CudaPtxDirectory.Resolve());

        Assert.False(backend.TryGatherRowsResident(output, input, [1]));
    }
}
