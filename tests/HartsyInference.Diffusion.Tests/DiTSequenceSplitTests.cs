using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The split of a joint text/image sequence takes each batch item's own rows. One contiguous row block, which is
/// what the joint blocks used, hands batch 0's rows to every item once B &gt; 1.</summary>
public sealed class DiTSequenceSplitTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void EachBatchItemGetsItsOwnRows(int batch)
    {
        const int seq = 7, width = 5, offset = 2, rows = 4;
        using CpuBackend backend = new();
        using Tensor joint = new(new TensorShape(batch, seq, width), DType.F32);
        Span<float> j = joint.AsSpan<float>();
        for (int i = 0; i < j.Length; i++) j[i] = i;
        using Tensor part = new(new TensorShape(batch, rows, width), DType.F32);
        DiTUtils.SliceSeq(backend, part, joint, batch, seq, offset);
        ReadOnlySpan<float> p = part.AsReadOnlySpan<float>();
        for (int b = 0; b < batch; b++)
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < width; c++)
                    Assert.Equal(((b * seq) + offset + r) * width + c, p[((b * rows) + r) * width + c]);
    }
}
