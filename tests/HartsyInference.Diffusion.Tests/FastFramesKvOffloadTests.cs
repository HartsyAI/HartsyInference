using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Video.Pipelines;
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

/// <summary>The auto mode's boundary and the host refusal: a resident cache is chosen only when the same check after the placement passes, and an
/// offloaded cache the host cannot hold is refused before a frame runs.</summary>
public sealed class FastFramesOffloadBoundaryTests
{
    private const long Mb = 1L << 20;

    [Fact]
    public void AResidentCacheFitsExactlyBesideTheWindowAndTheActivationAllowanceAndNotOneByteLess()
    {
        long need = 100 * Mb, window = 50 * Mb, room = need + window + FastFramesPipeline.DefaultStreamingHeadroomBytes;
        Assert.True(FastFramesPipeline.ResidentCacheFits(freeBytes: room, windowBytes: window, kvNeedBytes: need));
        Assert.False(FastFramesPipeline.ResidentCacheFits(freeBytes: room - 1, windowBytes: window, kvNeedBytes: need));
    }

    [Fact]
    public void ThePlacementHeadroomIsTheCacheNeedPlusTheActivationAllowancePlusTheWindowUnlessOverridden()
    {
        const long need = 100 * Mb, window = 50 * Mb;
        Assert.Equal(need + FastFramesPipeline.DefaultStreamingHeadroomBytes + window,
            FastFramesPipeline.PlacementHeadroomBytes(need, window, overrideBytes: null));
        Assert.Equal(1L << 30, FastFramesPipeline.DefaultStreamingHeadroomBytes);
        Assert.Equal(7 * Mb, FastFramesPipeline.PlacementHeadroomBytes(kvNeedBytes: 1, windowBytes: 0, overrideBytes: 7 * Mb));
    }

    [Fact]
    public void TheHostNeedIsTheWholeCachePlusOneBlocksJoinStaging()
    {
        Assert.Equal(300 * Mb, FastFramesPipeline.HostNeedBytes(cacheBytes: 280 * Mb, oneBlockJoinBytes: 20 * Mb));
        // A host with exactly the cache but not the staging is refused.
        Assert.Throws<InvalidOperationException>(() =>
            FastFramesPipeline.CheckHostFits("FastFrames L", FastFramesPipeline.HostNeedBytes(280 * Mb, 20 * Mb), 21, availableBytes: 280 * Mb));
    }

    [Fact]
    public void AutoChoosesOffloadWhenOnlyTheCacheAndWindowWouldFitWithoutTheAllowance()
    {
        // The cache and the window fit the free VRAM, but not the activation allowance the placement keeps as well.
        Assert.False(FastFramesPipeline.ResidentCacheFits(freeBytes: 150 * Mb, windowBytes: 50 * Mb, kvNeedBytes: 100 * Mb));
    }

    [Fact]
    public void AFreeAmountThatFitsTheCacheButNotTheWindowForcesOffload()
    {
        // The cache fits the free VRAM on its own, but the window the placement uploads after it does not: a resident choice would then fail the check.
        Assert.False(FastFramesPipeline.ResidentCacheFits(freeBytes: 6000 * Mb, windowBytes: 3000 * Mb, kvNeedBytes: 5000 * Mb));
    }

    [Fact]
    public void AHostThatHoldsTheOffloadedCacheExactlyPasses()
    {
        FastFramesPipeline.CheckHostFits("FastFrames L", needBytes: 26 * 1024 * Mb, latentFrames: 21, availableBytes: 26 * 1024 * Mb);
    }

    [Fact]
    public void AHostOneByteShortRefusesTheOffloadedCacheBeforeAFrame()
    {
        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            FastFramesPipeline.CheckHostFits("FastFrames L", needBytes: 26 * 1024 * Mb, latentFrames: 21, availableBytes: 26 * 1024 * Mb - 1));
        Assert.Contains("host memory", error.Message, StringComparison.Ordinal);
        Assert.Contains("21 latent frames", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AHostWithUnknownMemoryIsNotChecked()
    {
        FastFramesPipeline.CheckHostFits("FastFrames L", needBytes: long.MaxValue, latentFrames: 21, availableBytes: null);
    }
}
