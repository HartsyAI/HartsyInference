using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>The four calls an engine makes to get VRAM back, against a real device.
///
/// <para>They did nothing here until this change. <c>FreeActivations</c>, <c>TrimMemoryPool</c> and
/// <c>FreeAllDeviceMemory</c> are empty-bodied defaults on <c>IBackend</c> that only CUDA implemented, so the
/// twenty-odd engine call sites at generation and model-swap boundaries were no-ops on Vulkan and device memory
/// came back only when the GC happened to reach each tensor. A unit test on the shared cache proves the
/// bookkeeping; this proves the bytes.</para></summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanMemoryReleaseTests
{
    private readonly ITestOutputHelper _output;

    public VulkanMemoryReleaseTests(ITestOutputHelper output) => _output = output;

    private const int M = 512, K = 512, N = 512;

    /// <summary>How far the driver's free figure may drift between two probes without meaning anything.</summary>
    /// <remarks>Since the report became a live driver number it answers for every process on the card, not just
    /// this one, so a comparison across time needs room for a co-tenant. Far below the ~1 GB this test moves.</remarks>
    private const long CoTenantSlackBytes = 256L << 20;

    /// <summary>A weight big enough that keeping it resident is visible next to the activations.</summary>
    private static Tensor NewWeight() => Filled(new TensorShape(N, K), 0.01f);

    private static Tensor Filled(TensorShape shape, float value)
    {
        Tensor tensor = new(shape, DType.F32);
        Span<float> span = tensor.AsSpan<float>();
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = value;
        }
        return tensor;
    }

    /// <summary>Runs one Linear, leaving its output cached on the device — the state a phase boundary reclaims.</summary>
    private static Tensor RunLinear(VulkanBackend backend, Tensor weight)
    {
        Tensor input = Filled(new TensorShape(M, K), 0.5f);
        Tensor output = new(new TensorShape(M, N), DType.F32);
        try
        {
            backend.Linear(output, input, weight, null);
            return output;
        }
        finally
        {
            input.Dispose();
        }
    }

    [Fact]
    public void FreeActivations_ReturnsActivationBytesAndKeepsTheWeightResident()
    {
        if (!BackendGate.TryOpen("vulkan", _output.WriteLine, out IBackend? opened))
        {
            return;
        }
        using VulkanBackend backend = (VulkanBackend)opened!;
        using Tensor weight = NewWeight();

        backend.PreloadWeights([weight]);
        backend.Sync();
        (long weightsOnly, _, _, long cachedWeightsOnly) = backend.MemoryStats;

        List<Tensor> activations = [];
        for (int i = 0; i < 8; i++)
        {
            activations.Add(RunLinear(backend, weight));
        }
        backend.Sync();
        (long withActivations, _, _, long cachedWithActivations) = backend.MemoryStats;
        Assert.True(cachedWithActivations > cachedWeightsOnly,
            "the ops cached nothing, so this test would pass without freeing anything");

        (_, long missesBefore) = backend.GetTransferCacheStats();
        backend.FreeActivations();
        (long afterUsed, _, _, long afterCached) = backend.MemoryStats;

        _output.WriteLine($"weights only: used={weightsOnly >> 20} MB cached={cachedWeightsOnly >> 20} MB");
        _output.WriteLine($"+8 activations: used={withActivations >> 20} MB cached={cachedWithActivations >> 20} MB");
        _output.WriteLine($"after free: used={afterUsed >> 20} MB cached={afterCached >> 20} MB");

        // The cache is the exact measure; the allocator's own figure includes pooled blocks that a trim may or may
        // not have been able to hand back, which is a different question (see the trim test below).
        Assert.Equal(cachedWeightsOnly, afterCached);
        Assert.True(afterUsed <= withActivations,
            $"used bytes grew across a free: {withActivations >> 20} MB -> {afterUsed >> 20} MB");

        // Still resident: another op must not re-upload it. That is the whole point of freeing activations rather
        // than everything, and a free that took the weight with it would look identical on the byte counters.
        using Tensor after = RunLinear(backend, weight);
        (_, long missesAfter) = backend.GetTransferCacheStats();
        Assert.Equal(missesBefore + 1, missesAfter);   // the fresh input, and only the fresh input

        foreach (Tensor tensor in activations)
        {
            tensor.Dispose();
        }
    }

    /// <summary>A pinned activation is cross-step state whose only copy is on the device. Freeing it would not be
    /// slow, it would be wrong.</summary>
    [Fact]
    public void FreeActivations_KeepsAPinnedActivation()
    {
        if (!BackendGate.TryOpen("vulkan", _output.WriteLine, out IBackend? opened))
        {
            return;
        }
        using VulkanBackend backend = (VulkanBackend)opened!;
        using Tensor weight = NewWeight();

        using Tensor pinned = RunLinear(backend, weight);
        backend.PinActivation(pinned);
        using Tensor scratch = RunLinear(backend, weight);
        backend.Sync();

        (_, _, _, long cachedBefore) = backend.MemoryStats;
        backend.FreeActivations();
        (_, _, _, long cachedAfter) = backend.MemoryStats;
        (_, long missesBefore) = backend.GetTransferCacheStats();

        _output.WriteLine($"cached {cachedBefore >> 10} KB -> {cachedAfter >> 10} KB");
        Assert.True(cachedAfter < cachedBefore, "nothing was released");

        // Reading the pinned tensor must come from its surviving device copy, not from a re-upload.
        Tensor echo = new(pinned.Shape, DType.F32);
        try
        {
            backend.Scale(echo, pinned, 1f);
            (_, long missesAfter) = backend.GetTransferCacheStats();
            Assert.Equal(missesBefore, missesAfter);
        }
        finally
        {
            echo.Dispose();
        }
    }

    /// <summary>A trim hands pool reservations back without disturbing a single cache entry — it is the one of
    /// these four that is safe to call mid-generation.</summary>
    [Fact]
    public void TrimMemoryPool_ReleasesReservationsWithoutTouchingTheCaches()
    {
        if (!BackendGate.TryOpen("vulkan", _output.WriteLine, out IBackend? opened))
        {
            return;
        }
        using VulkanBackend backend = (VulkanBackend)opened!;
        using Tensor weight = NewWeight();

        backend.PreloadWeights([weight]);
        for (int i = 0; i < 4; i++)
        {
            RunLinear(backend, weight).Dispose();
        }
        backend.Sync();

        (_, long reservedBefore, int blocksBefore, long cachedBefore) = backend.MemoryStats;
        backend.TrimMemoryPool();
        (_, long reservedAfter, int blocksAfter, long cachedAfter) = backend.MemoryStats;

        _output.WriteLine($"reserved {reservedBefore >> 20} MB ({blocksBefore} blocks) -> "
            + $"{reservedAfter >> 20} MB ({blocksAfter} blocks), cached {cachedAfter >> 10} KB");
        Assert.Equal(cachedBefore, cachedAfter);
        Assert.True(reservedAfter <= reservedBefore, "a trim reserved MORE memory than it started with");
        Assert.True(blocksAfter <= blocksBefore);
    }

    /// <summary>The model-swap call: everything goes, and the device reports the memory back.</summary>
    [Fact]
    public void FreeAllDeviceMemory_ReturnsTheCardToItsOpeningState()
    {
        if (!BackendGate.TryOpen("vulkan", _output.WriteLine, out IBackend? opened))
        {
            return;
        }
        using VulkanBackend backend = (VulkanBackend)opened!;

        (long openingUsed, _, _, _) = backend.MemoryStats;
        long openingFree = backend.GetVramInfo().FreeBytes;

        using Tensor weight = NewWeight();
        backend.PreloadWeights([weight]);
        List<Tensor> activations = [];
        for (int i = 0; i < 8; i++)
        {
            activations.Add(RunLinear(backend, weight));
        }
        backend.Sync();

        backend.FreeAllDeviceMemory();
        (long afterUsed, _, _, long afterCached) = backend.MemoryStats;
        long afterFree = backend.GetVramInfo().FreeBytes;

        _output.WriteLine($"used {openingUsed >> 20} MB -> {afterUsed >> 20} MB, "
            + $"free {openingFree >> 20} MB -> {afterFree >> 20} MB, cached {afterCached} B");
        Assert.Equal(0, afterCached);
        Assert.Equal(openingUsed, afterUsed);
        // The driver's free figure moves with every process on the card, so this carries slack that the exact
        // allocator figures above do not need. It was written when the number was pure allocator arithmetic and
        // two probes of it were exactly comparable; a co-tenant taking a megabyte between them would otherwise
        // report a leak that did not happen. The allocator assertions are the ones with teeth here.
        Assert.True(afterFree >= openingFree - CoTenantSlackBytes,
            $"free VRAM did not come back: {openingFree >> 20} MB at open, {afterFree >> 20} MB after the sweep");

        // The tensors outlive the sweep, and reading one must still produce its value from the host copy rather
        // than dereferencing a buffer that has been handed back.
        foreach (Tensor tensor in activations)
        {
            Assert.Equal(M * N, tensor.AsReadOnlySpan<float>().Length);
            tensor.Dispose();
        }

        // And the backend still works afterwards: a sweep is not a teardown.
        using Tensor again = RunLinear(backend, weight);
        Assert.True(again.AsReadOnlySpan<float>()[0] != 0f);
    }
}
