using System.Runtime.InteropServices;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>Probe the device features via separate per-struct queries to isolate the chain bug.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanFeatureProbe
{
    private readonly ITestOutputHelper _out;
    public VulkanFeatureProbe(ITestOutputHelper o) { _out = o; }

    [Fact]
    public unsafe void IsolateFeatureQuery()
    {
        VulkanLibraryResolver.Register();
        using VulkanInstance instance = new();
        nint[] phys = instance.EnumeratePhysicalDevices();
        Assert.NotEmpty(phys);
        nint pd = phys[0];

        // Print apiVersion from properties
        VulkanApi.vkGetPhysicalDeviceProperties(pd, out VkPhysicalDeviceProperties props);
        uint api = props.apiVersion;
        _out.WriteLine($"apiVersion = {(api >> 22) & 0x7F}.{(api >> 12) & 0x3FF}.{api & 0xFFF}");
        _out.WriteLine($"deviceName = {VulkanUtilities.ReadFixedString(props.deviceName, 256)}");

        // Query each features struct alone (no chain) to see which one returns the right values.
        // 1) plain v1.0 features
        VulkanApi.vkGetPhysicalDeviceFeatures(pd, out VkPhysicalDeviceFeatures f10);
        _out.WriteLine($"v1.0 shaderInt16={f10.shaderInt16}  shaderInt64={f10.shaderInt64}  shaderFloat64={f10.shaderFloat64}");

        // 2) v1.1 alone
        VkPhysicalDeviceVulkan11Features f11 = new() { sType = VkStructureType.PhysicalDeviceVulkan11Features };
        VkPhysicalDeviceFeatures2 feat2_11 = new() { sType = VkStructureType.PhysicalDeviceFeatures2, pNext = (nint)(&f11) };
        VulkanApi.vkGetPhysicalDeviceFeatures2(pd, ref feat2_11);
        _out.WriteLine($"v1.1 ALONE: store16={f11.storageBuffer16BitAccess}");

        // 3) v1.2 alone
        VkPhysicalDeviceVulkan12Features f12 = new() { sType = VkStructureType.PhysicalDeviceVulkan12Features };
        VkPhysicalDeviceFeatures2 feat2_12 = new() { sType = VkStructureType.PhysicalDeviceFeatures2, pNext = (nint)(&f12) };
        VulkanApi.vkGetPhysicalDeviceFeatures2(pd, ref feat2_12);
        _out.WriteLine($"v1.2 ALONE: shaderFloat16={f12.shaderFloat16}  shaderInt8={f12.shaderInt8}  timelineSemaphore={f12.timelineSemaphore}  bufferDeviceAddress={f12.bufferDeviceAddress}");

        // 4) v1.3 alone
        VkPhysicalDeviceVulkan13Features f13 = new() { sType = VkStructureType.PhysicalDeviceVulkan13Features };
        VkPhysicalDeviceFeatures2 feat2_13 = new() { sType = VkStructureType.PhysicalDeviceFeatures2, pNext = (nint)(&f13) };
        VulkanApi.vkGetPhysicalDeviceFeatures2(pd, ref feat2_13);
        _out.WriteLine($"v1.3 ALONE: subgroupSizeControl={f13.subgroupSizeControl}  computeFullSubgroups={f13.computeFullSubgroups}  synchronization2={f13.synchronization2}  maintenance4={f13.maintenance4}");

        // Print struct sizes
        _out.WriteLine($"sizeof v1.0 = {sizeof(VkPhysicalDeviceFeatures)} (should be 220)");
        _out.WriteLine($"sizeof v1.1 = {sizeof(VkPhysicalDeviceVulkan11Features)} (should be 64)");
        _out.WriteLine($"sizeof v1.2 = {sizeof(VkPhysicalDeviceVulkan12Features)} (should be 208)");
        _out.WriteLine($"sizeof v1.3 = {sizeof(VkPhysicalDeviceVulkan13Features)} (should be 84)");

        // Memory layout
        VulkanApi.vkGetPhysicalDeviceMemoryProperties(pd, out VkPhysicalDeviceMemoryProperties mp);
        _out.WriteLine($"--- memory ---");
        for (int i = 0; i < mp.memoryHeapCount; i++)
        {
            VkMemoryHeap h = mp.GetMemoryHeap(i);
            _out.WriteLine($"  heap {i}: size={h.size / (1L<<20)} MB  flags={h.flags}");
        }
        for (int i = 0; i < mp.memoryTypeCount; i++)
        {
            VkMemoryType t = mp.GetMemoryType(i);
            _out.WriteLine($"  type {i}: heap={t.heapIndex}  props={t.propertyFlags}");
        }
    }
}
