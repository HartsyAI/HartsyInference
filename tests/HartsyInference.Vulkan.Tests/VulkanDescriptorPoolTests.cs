using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>A descriptor pool may only be reset once the GPU has passed every submission that could still bind its sets. Fills both pools with one recording open the whole time, so the flip back to the first pool has to submit that recording and wait for it; the old flip reset the pool with that work still queued.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanDescriptorPoolTests
{
    private readonly ITestOutputHelper _out;
    public VulkanDescriptorPoolTests(ITestOutputHelper o) { _out = o; }

    [Fact]
    public void FlipWaitsForTheWorkThatBoundTheRetiringPool()
    {
        VulkanLibraryResolver.Register();
        using VulkanInstance instance = new();
        if (instance.EnumeratePhysicalDevices().Length == 0)
        {
            if (Environment.GetEnvironmentVariable("HARTSY_REQUIRE_BACKENDS") == "1")
                throw new InvalidOperationException("HARTSY_REQUIRE_BACKENDS=1 but no Vulkan device is present");
            _out.WriteLine("SKIPPED: no Vulkan device");
            return;
        }
        using VulkanDevice device = VulkanDevice.Create(instance);
        using VulkanCommandStream stream = new(device.Handle, device.ComputeQueue, device.Capabilities.ComputeQueueFamilyIndex);
        using VulkanDescriptorManager pools = new(device.Handle, stream);
        ulong layout = pools.GetSetLayout(1);

        stream.AcquireRecording();
        Assert.Equal(0UL, stream.LastSubmitted);

        // Fill pool 0, then the allocation that flips to pool 1: pool 0 now retires at the tick the open recording
        // will carry, and pool 1 has never been used, so nothing is waited for yet.
        for (int i = 0; i < VulkanDescriptorManager.MaxSetsPerPool; i++) pools.AllocateSet(layout);
        pools.AllocateSet(layout);
        Assert.Equal(0UL, stream.LastSubmitted);

        // Fill pool 1 and flip back: pool 0's tick is still recording, so the flip must submit it and reach it
        // before the reset. Without that wait this call returns with the tick unsubmitted.
        for (int i = 0; i < VulkanDescriptorManager.MaxSetsPerPool - 1; i++) pools.AllocateSet(layout);
        pools.AllocateSet(layout);
        Assert.Equal(1UL, stream.LastSubmitted);
        Assert.True(stream.TimelineReached(1));
    }
}
