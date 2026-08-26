using HartsyInference.Core.Configuration;
using System.Runtime.InteropServices;
using static HartsyInference.Vulkan.VulkanUtilities;

namespace HartsyInference.Vulkan;

/// <summary>Wraps a VkInstance with deterministic disposal.</summary>
/// <remarks>Headless compute — no surface/swapchain extensions. Validation layer is opt-in via the
/// HARTSYINFERENCE_VK_VALIDATION env var.</remarks>
public sealed class VulkanInstance : IDisposable
{
    private nint _instance;
    private PinnedStringArray? _layers;
    private PinnedStringArray? _extensions;
    private PinnedUtf8String? _appName;
    private PinnedUtf8String? _engineName;

    public nint Handle
    {
        get
        {
            nint h = _instance;
            if (h == 0) throw new ObjectDisposedException(nameof(VulkanInstance));
            return h;
        }
    }

    /// <summary>Creates a Vulkan 1.3 instance. Validation layer enabled if HARTSYINFERENCE_VK_VALIDATION=1.</summary>
    public VulkanInstance(bool? enableValidation = null)
    {
        VulkanLibraryResolver.Register();

        bool enableVal = enableValidation ?? EngineKnobs.VkValidation.Value;

        List<string> layers = new();
        if (enableVal && IsLayerAvailable("VK_LAYER_KHRONOS_validation"))
        {
            layers.Add("VK_LAYER_KHRONOS_validation");
        }

        // Headless compute needs no instance extensions in Vulkan 1.3.
        List<string> extensions = new();

        _appName = new PinnedUtf8String("HartsyInference");
        _engineName = new PinnedUtf8String("HartsyInference.Vulkan");
        _layers = new PinnedStringArray(layers);
        _extensions = new PinnedStringArray(extensions);

        unsafe
        {
            VkApplicationInfo appInfo = new()
            {
                sType = VkStructureType.ApplicationInfo,
                pNext = 0,
                pApplicationName = _appName.Pointer,
                applicationVersion = MakeVersion(1, 0, 0),
                pEngineName = _engineName.Pointer,
                engineVersion = MakeVersion(1, 0, 0),
                apiVersion = VkApiVersion13,
            };

            VkInstanceCreateInfo ci = new()
            {
                sType = VkStructureType.InstanceCreateInfo,
                pNext = 0,
                flags = 0,
                pApplicationInfo = (nint)(&appInfo),
                enabledLayerCount = _layers.Count,
                ppEnabledLayerNames = _layers.Pointers,
                enabledExtensionCount = _extensions.Count,
                ppEnabledExtensionNames = _extensions.Pointers,
            };

            VulkanApi.vkCreateInstance(in ci, 0, out _instance).ThrowOnError("vkCreateInstance");
        }
    }

    private static unsafe bool IsLayerAvailable(string name)
    {
        uint count = 0;
        if (VulkanApi.vkEnumerateInstanceLayerProperties(ref count, 0) != VkResult.Success) return false;
        if (count == 0) return false;

        int sz = sizeof(VkLayerProperties);
        nint block = Marshal.AllocHGlobal((int)count * sz);
        try
        {
            if (VulkanApi.vkEnumerateInstanceLayerProperties(ref count, block) != VkResult.Success) return false;
            byte* p = (byte*)block;
            for (uint i = 0; i < count; i++)
            {
                VkLayerProperties* lp = (VkLayerProperties*)(p + i * sz);
                string n = ReadFixedString(lp->layerName, 256);
                if (n == name) return true;
            }
            return false;
        }
        finally { Marshal.FreeHGlobal(block); }
    }

    /// <summary>Enumerates physical devices visible to this instance.</summary>
    public unsafe nint[] EnumeratePhysicalDevices()
    {
        uint count = 0;
        VulkanApi.vkEnumeratePhysicalDevices(Handle, ref count, 0).ThrowOnError("vkEnumeratePhysicalDevices count");
        if (count == 0) return Array.Empty<nint>();

        nint block = Marshal.AllocHGlobal((int)count * IntPtr.Size);
        try
        {
            VulkanApi.vkEnumeratePhysicalDevices(Handle, ref count, block).ThrowOnError("vkEnumeratePhysicalDevices fetch");
            nint[] result = new nint[count];
            nint* src = (nint*)block;
            for (uint i = 0; i < count; i++) result[i] = src[i];
            return result;
        }
        finally { Marshal.FreeHGlobal(block); }
    }

    public void Dispose()
    {
        nint h = Interlocked.Exchange(ref _instance, 0);
        if (h != 0) VulkanApi.vkDestroyInstance(h, 0);
        _layers?.Dispose(); _layers = null;
        _extensions?.Dispose(); _extensions = null;
        _appName?.Dispose(); _appName = null;
        _engineName?.Dispose(); _engineName = null;
        GC.SuppressFinalize(this);
    }

    ~VulkanInstance()
    {
        nint h = Interlocked.Exchange(ref _instance, 0);
        if (h != 0) VulkanApi.vkDestroyInstance(h, 0);
    }
}
