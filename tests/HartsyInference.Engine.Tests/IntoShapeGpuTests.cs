using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Engine.Tests.Services;
using HartsyInference.Vulkan;

namespace HartsyInference.Engine.Tests;

/// <summary><see cref="Tensor.IntoShape"/> on a real device: the buffer and its pin move with no host read.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class IntoShapeGpuTests
{
    private static readonly Lazy<string?> CudaFailure = new(static () => Probe(static () => Cuda().Dispose()));
    private static readonly Lazy<string?> VulkanFailure = new(static () => Probe(static () => new VulkanBackend().Dispose()));

    private static CudaBackend Cuda() => new(0, Ernie5ModelGpuParityTests.PtxDirectory());

    [CudaRun]
    public void Cuda_IntoShape_KeepsTheResultOnTheDevice() => MoveStaysOnDevice(Cuda());

    [VulkanRun]
    public void Vulkan_IntoShape_KeepsTheResultOnTheDevice() => MoveStaysOnDevice(new VulkanBackend());

    private void MoveStaysOnDevice(IBackend device)
    {
        using (device)
        using (Tensor host = Filled(new Random(2), 6, 8))
        {
            float[] expected = host.AsReadOnlySpan<float>().ToArray();
            Tensor produced = new(host.Shape, DType.F32);
            device.Scale(produced, host, 1f);
            device.ResetD2hSyncCount();
            using Tensor moved = produced.IntoShape(new TensorShape(2, 24));
            using Tensor twice = new(moved.Shape, DType.F32);
            device.Scale(twice, moved, 2f);
            Assert.Equal(0, device.GetD2hSyncCount());
            Assert.False(moved.HostBytesCurrent);
            Assert.Equal(expected.Select(v => v * 2f), twice.AsReadOnlySpan<float>().ToArray());

            // The sweep frees a moved tensor like any other activation, unless the pin it came with keeps it.
            Tensor kept = new(host.Shape, DType.F32);
            device.Scale(kept, host, 3f);
            device.PinActivation(kept);
            using Tensor pinned = kept.IntoShape(new TensorShape(48));
            device.FreeActivations();
            Assert.Equal(expected.Select(v => v * 3f), pinned.AsReadOnlySpan<float>().ToArray());
            device.UnpinActivation(pinned);
            device.Scale(twice, host, 2f);
            Assert.Equal(expected.Select(v => v * 2f), twice.AsReadOnlySpan<float>().ToArray());
        }
    }

    private static Tensor Filled(Random rng, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>()) v = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    private static string? Probe(Action open)
    {
        try
        {
            open();
            return null;
        }
        catch (Exception ex)
        {
            return $"{ex.GetType().Name}: {ex.Message}";
        }
    }

    private sealed class CudaRunAttribute : FactAttribute
    {
        public CudaRunAttribute()
        {
            if (CudaFailure.Value is { } failure) Skip = $"SKIPPED: no usable CUDA device ({failure}).";
        }
    }

    private sealed class VulkanRunAttribute : FactAttribute
    {
        public VulkanRunAttribute()
        {
            if (VulkanFailure.Value is { } failure) Skip = $"SKIPPED: no usable Vulkan device ({failure}).";
        }
    }
}
