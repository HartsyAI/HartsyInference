using System.Reflection;
using System.Runtime.InteropServices;

namespace SharpInference.Vulkan;

/// <summary>Resolves the "vulkan-1" library name to the platform-specific Vulkan loader at runtime: libvulkan.so.1 (Linux), vulkan-1.dll (Windows), libvulkan.1.dylib / libMoltenVK.dylib (macOS).</summary>
public static class VulkanLibraryResolver
{
    private static int _registered;

    /// <summary>Registers the resolver for the SharpInference.Vulkan assembly. Safe to call multiple times.</summary>
    public static void Register()
    {
        if (Interlocked.CompareExchange(ref _registered, 1, 0) == 0)
        {
            NativeLibrary.SetDllImportResolver(typeof(VulkanLibraryResolver).Assembly, ResolveLibrary);
        }
    }

    private static nint ResolveLibrary(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName != "vulkan-1") return 0;

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            if (NativeLibrary.TryLoad("libvulkan.so.1", out nint h)) return h;
            if (NativeLibrary.TryLoad("libvulkan.so", out h)) return h;
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            if (NativeLibrary.TryLoad("vulkan-1.dll", out nint h)) return h;
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            if (NativeLibrary.TryLoad("libvulkan.1.dylib", out nint h)) return h;
            if (NativeLibrary.TryLoad("libMoltenVK.dylib", out h)) return h;
        }

        return 0;
    }
}
