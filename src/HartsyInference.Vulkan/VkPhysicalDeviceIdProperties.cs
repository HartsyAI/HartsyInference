using System.Runtime.InteropServices;
namespace HartsyInference.Vulkan;
/// <summary>Vulkan 1.1 device identity properties; UUID is independent of the pipeline cache.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceIdProperties
{
    public VkStructureType SType;
    public nint Next;
    public fixed byte DeviceUuid[16];
    public fixed byte DriverUuid[16];
    public fixed byte DeviceLuid[8];
    public uint DeviceNodeMask;
    public uint DeviceLuidValid;
}
