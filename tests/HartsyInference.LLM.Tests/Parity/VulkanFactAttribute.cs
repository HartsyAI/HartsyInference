using Xunit;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>A <see cref="FactAttribute"/> reported as skipped (never passed) when no Vulkan device opens or the block-op SPIR-V is not built.</summary>
public sealed class VulkanFactAttribute : FactAttribute
{
    public VulkanFactAttribute()
    {
        if (!DeviceProbe.VulkanReady) Skip = $"SKIPPED: no usable Vulkan device or block-op SPIR-V ({DeviceProbe.VulkanFailure}); GPU check did not run.";
    }
}
