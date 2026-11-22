using Xunit;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>A theory that runs only on a Vulkan device with VK_NV_cooperative_matrix2; elsewhere it is reported as skipped.</summary>
public sealed class VulkanCoopMat2TheoryAttribute : TheoryAttribute
{
    public VulkanCoopMat2TheoryAttribute()
    {
        if (!DeviceProbe.VulkanReady) Skip = $"SKIPPED: no usable Vulkan device or block-op SPIR-V ({DeviceProbe.VulkanFailure}); GPU check did not run.";
        else if (!DeviceProbe.VulkanCoopMat2Ready) Skip = "SKIPPED: the Vulkan device has no VK_NV_cooperative_matrix2; the coopmat2 check did not run.";
    }
}
