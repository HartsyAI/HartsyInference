using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.Vulkan.Tests;

/// <summary>Pins the sync2 access-flag and result constants to their official Vulkan spec values. These are CPU-only (no device needed). A regression here previously left <see cref="VkAccessFlags2.ShaderStorageRead"/>/<c>ShaderStorageWrite</c> off by one bit, silently weakening the compute-compute barrier in <c>RecordGlobalComputeBarrier</c> — exactly the kind of wrong-enum-constant bug that produces intermittent wrong results with no crash.</summary>
public sealed class VulkanConstantTests
{
    [Fact]
    public void Sync2AccessFlags_MatchVulkanSpec()
    {
        // Values from VK_ACCESS_2_* in vulkan_core.h (KhronosGroup/Vulkan-Headers).
        Assert.Equal(0x00000008UL, VkAccessFlags2.UniformRead);          // was wrongly 0x40 (collided with ShaderWrite)
        Assert.Equal(0x00000020UL, VkAccessFlags2.ShaderRead);
        Assert.Equal(0x00000040UL, VkAccessFlags2.ShaderWrite);
        Assert.Equal(0x100000000UL, VkAccessFlags2.ShaderSampledRead);
        Assert.Equal(0x200000000UL, VkAccessFlags2.ShaderStorageRead);  // was wrongly 0x100000000 (=SampledRead)
        Assert.Equal(0x400000000UL, VkAccessFlags2.ShaderStorageWrite); // was wrongly 0x200000000 (=StorageRead)
    }

    [Fact]
    public void ComputeBarrierUsesDistinctStorageReadWrite()
    {
        // The whole point of the barrier: storage write must be made available before storage
        // read is made visible. If these two ever collapse to the same value, the barrier is a no-op.
        Assert.NotEqual(VkAccessFlags2.ShaderStorageWrite, VkAccessFlags2.ShaderStorageRead);
    }

    [Fact]
    public void OutOfPoolMemoryResult_MatchesVulkanSpec()
    {
        Assert.Equal(-1000069000, (int)VkResult.ErrorOutOfPoolMemory);
    }
}
