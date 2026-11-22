using Xunit;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>A theory that runs only where <see cref="DeviceProbe"/> finds a usable Vulkan device; elsewhere it is reported as skipped.</summary>
public sealed class VulkanTheoryAttribute : TheoryAttribute
{
    public VulkanTheoryAttribute()
    {
        if (!DeviceProbe.VulkanReady) Skip = $"SKIPPED: no usable Vulkan device or block-op SPIR-V ({DeviceProbe.VulkanFailure}); GPU check did not run.";
    }
}
