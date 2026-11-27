using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The offloaded frame cache: a written slot moves to host memory and a join stages it back, so the joined keys and values are the
/// same as a resident cache's. The output of a clip does not depend on where the frame cache lives.</summary>
public sealed class FastFramesKvOffloadTests
{
    private static Tensor Slot(int tokens, int channels, float fill)
    {
        Tensor t = new(new TensorShape([(long)tokens, channels]), DType.F32);
        for (int i = 0; i < t.AsSpan<float>().Length; i++)
            t.AsSpan<float>()[i] = fill + i * 0.25f;
        return t;
    }

    [Fact]
    public void AnOffloadedCacheJoinsTheSameKeysAndValuesAsAResidentOne()
    {
        using CpuBackend backend = new();
        using FrameKvCache resident = new(backend, layers: 2, maxFrames: 4);
        using FrameKvCache offloaded = new(backend, layers: 2, maxFrames: 4, offload: true);
        for (int layer = 0; layer < 2; layer++)
            for (int f = 0; f < 4; f++)
            {
                resident.Set(layer, f, Slot(3, 8, layer * 10 + f), Slot(3, 8, -(layer * 10 + f)));
                offloaded.Set(layer, f, Slot(3, 8, layer * 10 + f), Slot(3, 8, -(layer * 10 + f)));
            }

        for (int layer = 0; layer < 2; layer++)
        {
            (Tensor rk, Tensor rv) = resident.JoinPrefix(layer, 3);
            (Tensor ok, Tensor ov) = offloaded.JoinPrefix(layer, 3);
            using (rk) using (rv) using (ok) using (ov)
            {
                Assert.Equal(rk.Shape, ok.Shape);
                Assert.Equal(rk.AsReadOnlySpan<float>().ToArray(), ok.AsReadOnlySpan<float>().ToArray());
                Assert.Equal(rv.AsReadOnlySpan<float>().ToArray(), ov.AsReadOnlySpan<float>().ToArray());
            }
        }
    }

    [Fact]
    public void AReWrittenOffloadedSlotJoinsItsNewValues()
    {
        using CpuBackend backend = new();
        using FrameKvCache offloaded = new(backend, layers: 1, maxFrames: 2, offload: true);
        offloaded.Set(0, 0, Slot(2, 4, 1f), Slot(2, 4, -1f));
        offloaded.Set(0, 1, Slot(2, 4, 5f), Slot(2, 4, -5f));
        offloaded.Set(0, 0, Slot(2, 4, 9f), Slot(2, 4, -9f));

        (Tensor keys, Tensor values) = offloaded.JoinPrefix(0, 1);
        using (keys) using (values)
        {
            Assert.Equal(9f, keys.AsReadOnlySpan<float>()[0]);
            Assert.Equal(-9f, values.AsReadOnlySpan<float>()[0]);
            Assert.Equal(5f, keys.AsReadOnlySpan<float>()[8]);
        }
    }

    [Fact]
    public void OffloadIsOffByDefault()
    {
        using CpuBackend backend = new();
        using FrameKvCache cache = new(backend, layers: 1, maxFrames: 1);
        Assert.False(cache.Offloaded);
    }
}
