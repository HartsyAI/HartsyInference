using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Tests.Transformer;

public sealed class BlockDiagonalProjectionTests
{
    // HcaAttention projects wo_a through HostOps.BlockDiagonal; on the CPU it must keep the host grouped projection's exact bits.
    [Fact]
    public void BlockDiagonal_OnCpu_EqualsTheHostGroupedProjectionBitForBit()
    {
        const int tokens = 5, groups = 8, rank = 16, groupDim = 24;
        Random rng = new(3);
        float[] x = Enumerable.Range(0, tokens * groups * groupDim).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        float[] w = Enumerable.Range(0, groups * rank * groupDim).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        using Tensor weight = new(new TensorShape(groups * rank, groupDim), DType.F32);
        w.CopyTo(weight.AsSpan<float>());
        float[] want = new float[tokens * groups * rank];
        GroupedOutputProjection.Apply(x, w, tokens, groups, rank, groupDim, want);

        float[] got = HostOps.BlockDiagonal(new CpuBackend(), x, tokens, groups, weight, null);

        Assert.Equal(want.Select(BitConverter.SingleToInt32Bits), got.Select(BitConverter.SingleToInt32Bits));
    }

    [Fact]
    public void BlockDiagonal_RefusesPartialRows()
    {
        using Tensor weight = new(new TensorShape(4, 3), DType.F32);
        Assert.Throws<ArgumentException>(() => HostOps.BlockDiagonal(new CpuBackend(), new float[7], 2, 2, weight, null));
        Assert.Throws<ArgumentException>(() => HostOps.BlockDiagonal(new CpuBackend(), new float[6], 0, 2, weight, null));
    }
}
