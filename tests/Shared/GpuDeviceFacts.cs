using HartsyInference.Core.Backends;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.Tests.Shared;

/// <summary>Whether a CUDA or Vulkan backend opens on this machine, probed once per run so test discovery never opens a device twice.</summary>
internal static class GpuDevices
{
    private static readonly Lazy<string?> Cuda = new(static () => Probe(static () => OpenCuda().Dispose()));
    private static readonly Lazy<string?> Vulkan = new(static () => Probe(static () =>
    {
        using VulkanBackend backend = OpenVulkan();
        _vulkanF16 = backend.Capabilities.SupportsF16;
    }));
    private static bool _vulkanF16;

    /// <summary>Whether a Vulkan device opens and has F16.</summary>
    public static bool VulkanSupportsF16 => Vulkan.Value is null && _vulkanF16;

    /// <summary>Why CUDA cannot run, or null when it can.</summary>
    public static string? CudaFailure => Cuda.Value;

    /// <summary>Why Vulkan cannot run, or null when it can.</summary>
    public static string? VulkanFailure => Vulkan.Value;

    /// <summary>A CUDA backend on device 0 over the PTX beside the tests or in the repository.</summary>
    public static CudaBackend OpenCuda() => new(0, PtxDirectory());

    /// <summary>A Vulkan backend on the default device.</summary>
    public static VulkanBackend OpenVulkan() => new();

    private static string PtxDirectory()
    {
        string beside = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (Directory.Exists(beside)) return beside;
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }
        throw new DirectoryNotFoundException("No CUDA PTX directory beside the test assembly or in the repository.");
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

/// <summary>A fact reported as skipped, never passed, when no Vulkan device with F16 opens.</summary>
internal sealed class VulkanF16DeviceFactAttribute : FactAttribute
{
    public VulkanF16DeviceFactAttribute()
    {
        if (GpuDevices.VulkanFailure is { } failure) Skip = $"SKIPPED: no usable Vulkan device ({failure}).";
        else if (!GpuDevices.VulkanSupportsF16) Skip = "SKIPPED: this Vulkan device has no F16.";
    }
}
