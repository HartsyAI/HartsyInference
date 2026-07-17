using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

[Trait("Category", "GpuIntegration")]
public sealed class VulkanDeviceInfoTest
{
    private readonly ITestOutputHelper _out;
    public VulkanDeviceInfoTest(ITestOutputHelper output) { _out = output; }

    [Fact]
    public void PrintSelectedDevice()
    {
        try
        {
            using VulkanInstance instance = new();
            nint[] phys = instance.EnumeratePhysicalDevices();
            _out.WriteLine($"Physical devices: {phys.Length}");
            using VulkanDevice device = VulkanDevice.Create(instance);
            VulkanCapabilities c = device.Capabilities;
            _out.WriteLine($"Selected: {c}");
            _out.WriteLine($"  vendor=0x{c.VendorId:X4} ({c.VendorString})  type={c.DeviceType}");
            _out.WriteLine($"  VRAM={c.TotalVramBytes / (1L << 20)} MB");
            _out.WriteLine($"  subgroup={c.SubgroupSize} [{c.MinSubgroupSize}-{c.MaxSubgroupSize}]");
            _out.WriteLine($"  fp16={c.SupportsFp16}  store16={c.Storage16Bit}  sgsControl={c.SubgroupSizeControl}  sync2={c.Synchronization2}  timeline={c.TimelineSemaphore}");
            _out.WriteLine($"  ReBAR={c.HasReBar}  pushDesc={c.HasPushDescriptor}  memBudget={c.HasMemoryBudget}  coopMat={c.HasCooperativeMatrix}  int8dot={c.HasInt8DotProduct}");
            _out.WriteLine($"  maxComputeShared={c.MaxComputeSharedMemoryBytes}  maxWGInvocations={c.MaxComputeWorkGroupInvocations}");
            _out.WriteLine($"  computeQueueFamily={c.ComputeQueueFamilyIndex}  asyncCompute={c.IsAsyncComputeQueue}");
        }
        catch (Exception e)
        {
            _out.WriteLine("No Vulkan: " + e.Message);
        }
    }
}
