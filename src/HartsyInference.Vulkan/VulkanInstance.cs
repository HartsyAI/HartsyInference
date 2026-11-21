using HartsyInference.Core.Configuration;
using System.Runtime.InteropServices;
using static HartsyInference.Vulkan.VulkanUtilities;

namespace HartsyInference.Vulkan;

/// <summary>Wraps a VkInstance with deterministic disposal.</summary>
/// <remarks>Headless compute — no surface/swapchain extensions. Validation layer is opt-in via the
/// HARTSYINFERENCE_VK_VALIDATION env var.
/// <para>The first instance a process creates is preceded by one more that is never destroyed
/// (<see cref="HoldsProcessInstance"/>). On the NVIDIA Windows driver, destroying the device of a process's only
/// instance after cooperative-matrix attention ran on it can leave a driver helper thread starting on memory the
/// driver has already released, which ends the process with an access violation in <c>nvoglv64.dll</c>. With a
/// second instance alive that does not happen, and creating the next backend is several times faster.
/// <c>diagnostics.vkKeepInstance</c> turns it off.</para></remarks>
public sealed class VulkanInstance : IDisposable
{
    private static readonly object _heldLock = new();
    private static nint _held;

    /// <summary>Whether this process holds its idle instance: true from the first instance created with
    /// <c>diagnostics.vkKeepInstance</c> on, until the process ends.</summary>
    public static bool HoldsProcessInstance => Volatile.Read(ref _held) != 0;

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
        HoldProcessInstance();

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

        Create(_appName, _engineName, _layers, _extensions, out _instance).ThrowOnError("vkCreateInstance");
    }

    private static unsafe VkResult Create(PinnedUtf8String appName, PinnedUtf8String engineName, PinnedStringArray layers,
        PinnedStringArray extensions, out nint instance)
    {
        VkApplicationInfo appInfo = new()
        {
            sType = VkStructureType.ApplicationInfo,
            pNext = 0,
            pApplicationName = appName.Pointer,
            applicationVersion = MakeVersion(1, 0, 0),
            pEngineName = engineName.Pointer,
            engineVersion = MakeVersion(1, 0, 0),
            apiVersion = VkApiVersion13,
        };

        VkInstanceCreateInfo ci = new()
        {
            sType = VkStructureType.InstanceCreateInfo,
            pNext = 0,
            flags = 0,
            pApplicationInfo = (nint)(&appInfo),
            enabledLayerCount = layers.Count,
            ppEnabledLayerNames = layers.Pointers,
            enabledExtensionCount = extensions.Count,
            ppEnabledExtensionNames = extensions.Pointers,
        };

        return VulkanApi.vkCreateInstance(in ci, 0, out instance);
    }

    /// <summary>Creates the instance this process holds until it ends, once, before the first instance a caller asked
    /// for. It has no layers, no extensions and no device, and nothing destroys it: not Dispose, not a finalizer, not
    /// process exit, because destroying the last instance is the path being kept away from. A failure to create it is
    /// left for the caller's own instance to report.</summary>
    private static void HoldProcessInstance()
    {
        if (Volatile.Read(ref _held) != 0 || !EngineKnobs.VkKeepInstance.Value) return;
        lock (_heldLock)
        {
            if (_held != 0) return;
            using PinnedUtf8String appName = new("HartsyInference");
            using PinnedUtf8String engineName = new("HartsyInference.Vulkan");
            using PinnedStringArray none = new(new List<string>());
            if (Create(appName, engineName, none, none, out nint held) == VkResult.Success) Volatile.Write(ref _held, held);
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
