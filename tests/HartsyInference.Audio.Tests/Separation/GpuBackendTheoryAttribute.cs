using HartsyInference.Cuda;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.Audio.Tests.Separation;

/// <summary>Skips unless the named GPU backend has a device.</summary>
public sealed class GpuBackendTheoryAttribute : TheoryAttribute
{
    public GpuBackendTheoryAttribute(string backend)
    {
        bool available = backend == "cuda" ? CudaContext.IsAvailable() : VulkanContext.GetDeviceCount() > 0;
        if (!available) Skip = $"SKIPPED: no {backend} device.";
    }
}
