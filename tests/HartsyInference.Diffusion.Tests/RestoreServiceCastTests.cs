using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The CPU-backend F32 promotion SeedVR2's fp16 checkpoints go through: a wrong pass-through leaves a half
/// tensor for an F32-only kernel (silent until the first GEMM), and a copy of an already-F32 view leaks memory the
/// size of the checkpoint.</summary>
public sealed class RestoreServiceCastTests
{
    [Fact]
    public void CastAllToF32_ConvertsHalfTensorsAndKeepsF32Views()
    {
        using Tensor half = new(new TensorShape(2, 3), DType.F16);
        using Tensor full = new(new TensorShape(4), DType.F32);
        half.AsSpan<Half>().Fill((Half)1.5f);
        full.AsSpan<float>().Fill(2f);
        Dictionary<string, Tensor> weights = new() { ["a.bias"] = half, ["b.weight"] = full };
        List<Tensor> owned = [];

        Dictionary<string, Tensor> cast = RestoreService.CastAllToF32(weights, owned);

        Assert.Equal(DType.F32, cast["a.bias"].DType);
        Assert.Equal(1.5f, cast["a.bias"].AsSpan<float>()[5]);
        Assert.Same(full, cast["b.weight"]);
        Tensor single = Assert.Single(owned);
        Assert.Same(cast["a.bias"], single);
        single.Dispose();
    }
}
