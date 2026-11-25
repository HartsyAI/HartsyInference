using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The refusal rules for a biased H3 forward: the shape and dtype check, and the route check that refuses sparse and chunked
/// attention with a reason. Self-consistency only; it does not run a full forward.</summary>
public sealed class MiniMaxH3BiasRouteTests
{
    [Fact]
    public void BiasMustBeF32SquareOfTheSequenceLength()
    {
        using Tensor ok = new(new TensorShape(5, 5), DType.F32);
        MiniMaxH3Transformer.ValidateAttentionBias(ok, 5);

        using Tensor rectangular = new(new TensorShape(5, 4), DType.F32);
        Assert.Throws<ArgumentException>(() => MiniMaxH3Transformer.ValidateAttentionBias(rectangular, 5));

        using Tensor wrongLength = new(new TensorShape(4, 4), DType.F32);
        Assert.Throws<ArgumentException>(() => MiniMaxH3Transformer.ValidateAttentionBias(wrongLength, 5));

        using Tensor half = new(new TensorShape(5, 5), DType.F16);
        Assert.Throws<ArgumentException>(() => MiniMaxH3Transformer.ValidateAttentionBias(half, 5));
    }

    [Fact]
    public void DenseUnchunkedRouteIsAllowed()
    {
        MiniMaxH3Transformer.CheckBiasedRoute(seq: 8, chunkRows: int.MaxValue, freeBytes: 1L << 30, sparse: false);
    }

    [Fact]
    public void SparseRouteIsRefused()
    {
        NotSupportedException error = Assert.Throws<NotSupportedException>(() =>
            MiniMaxH3Transformer.CheckBiasedRoute(seq: 8, chunkRows: int.MaxValue, freeBytes: 1L << 30, sparse: true));
        Assert.Contains("sparse", error.Message);
    }

    [Fact]
    public void ChunkedRouteIsRefusedWithVramAndChunkRows()
    {
        NotSupportedException error = Assert.Throws<NotSupportedException>(() =>
            MiniMaxH3Transformer.CheckBiasedRoute(seq: 4096, chunkRows: 1024, freeBytes: 3_000_000_000L, sparse: false));
        Assert.Contains("chunks at 1024 rows", error.Message);
        Assert.Contains("seq=4096", error.Message);
        Assert.Contains("free VRAM 3000000000 bytes", error.Message);
    }
}
