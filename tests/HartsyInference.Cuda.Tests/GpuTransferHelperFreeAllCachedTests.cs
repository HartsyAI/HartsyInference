using HartsyInference.Core.Tensors;
using HartsyInference.Gpu;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>Every route to "release everything" on the CUDA transfer cache must leave nothing held.
///
/// <para>CUDA's release is more than the shared cache's sweep. An activation can carry a Q8_1 sidecar — three more
/// device buffers — that the shared sweep never learns about, because it hands buffers back without calling the
/// per-activation eviction hook. The wrapper that also releases those lived only on a static entry point
/// (<c>GpuTransferHelper.FreeAllCached(State)</c>), which teardown and <c>EvictGpuCache</c> call.
/// <see cref="GpuBackendBase.FreeAllDeviceMemory"/> — the sweep an engine makes at a model-swap boundary — holds the
/// cache only as <see cref="IGpuResidency"/>, so it called the shared sweep directly and went around the wrapper:
/// every sidecar still alive at that moment stayed on the card.</para>
///
/// <para>Needs no device and no registered backend. Every pointer here is zero, which each free path treats as
/// "nothing to free" before it reaches the driver, so what is left to check is the bookkeeping: that the release
/// reached the sidecars at all.</para></summary>
[Trait("Category", "GpuIntegration")]
public sealed class GpuTransferHelperFreeAllCachedTests
{
    private const int Width = 32;

    /// <summary>A backend that is nothing but a residency cache, so <see cref="GpuBackendBase.FreeAllDeviceMemory"/>
    /// runs against a real CUDA <see cref="GpuTransferHelper.State"/> without a context behind it.</summary>
    private sealed class ResidencyOnlyBackend(IGpuResidency residency) : GpuBackendBase
    {
        protected override IGpuResidency Residency => residency;

        public override (long FreeBytes, long TotalBytes) GetVramInfo() => (0, 0);

        protected override void TrimMemoryPoolCore() { }

        protected override void DisposeCore() { }
    }

    [Fact]
    public void FreeAllDeviceMemory_Leaves_No_Sidecar_Behind()
    {
        AssertReleasesEverything(state => new ResidencyOnlyBackend(state).FreeAllDeviceMemory());
    }

    [Fact]
    public void The_Residency_Interface_Leaves_No_Sidecar_Behind()
    {
        AssertReleasesEverything(state => ((IGpuResidency)state).FreeAllCached());
    }

    /// <summary>The control: the entry point that already ran the wrapper, and must keep running the same
    /// cleanup as the other two.</summary>
    [Fact]
    public void The_Static_Entry_Point_Leaves_No_Sidecar_Behind()
    {
        AssertReleasesEverything(GpuTransferHelper.FreeAllCached);
    }

    private static void AssertReleasesEverything(Action<GpuTransferHelper.State> release)
    {
        using GpuTransferHelper.State state = new();
        using Tensor first = new(new TensorShape(Width), DType.F32);
        using Tensor second = new(new TensorShape(Width), DType.F32);
        foreach (Tensor tensor in new[] { first, second })
        {
            state.CacheActivation(tensor, 0, Width * sizeof(float));
            state.SidecarCache[tensor] = (0, 0, 0, Width);
        }
        Assert.Equal(2, state.ActivationCount);
        Assert.Equal(2, state.SidecarCache.Count);

        release(state);

        Assert.Empty(state.SidecarCache);
        Assert.Equal(0, state.ActivationCount);
        Assert.Equal(0, state.CachedBufferCount);
    }
}
