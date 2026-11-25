using HartsyInference.Cuda;
using HartsyInference.Engine.Tests.Services;
using HartsyInference.Vulkan;

namespace HartsyInference.Engine.Tests;

/// <summary>Whether a CUDA or Vulkan backend opens on this machine, probed once per run so test discovery never opens a device twice.</summary>
internal static class GpuDevices
{
    private static readonly Lazy<string?> Cuda = new(static () => Probe(static () => OpenCuda().Dispose()));
    private static readonly Lazy<string?> Vulkan = new(static () => Probe(static () => new VulkanBackend().Dispose()));

    /// <summary>Why CUDA cannot run, or null when it can.</summary>
    public static string? CudaFailure => Cuda.Value;

    /// <summary>Why Vulkan cannot run, or null when it can.</summary>
    public static string? VulkanFailure => Vulkan.Value;

    /// <summary>A CUDA backend on device 0 over the PTX beside the tests or in the repository.</summary>
    public static CudaBackend OpenCuda() => new(0, Ernie5ModelGpuParityTests.PtxDirectory());

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
}

/// <summary>A fact reported as skipped, never passed, when no CUDA device opens.</summary>
internal sealed class CudaDeviceFactAttribute : FactAttribute
{
    public CudaDeviceFactAttribute()
    {
        if (GpuDevices.CudaFailure is { } failure) Skip = $"SKIPPED: no usable CUDA device ({failure}).";
    }
}

/// <summary>A fact reported as skipped, never passed, when no Vulkan device opens.</summary>
internal sealed class VulkanDeviceFactAttribute : FactAttribute
{
    public VulkanDeviceFactAttribute()
    {
        if (GpuDevices.VulkanFailure is { } failure) Skip = $"SKIPPED: no usable Vulkan device ({failure}).";
    }
}
