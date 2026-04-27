using System.Runtime.InteropServices;

namespace SharpInference.Vulkan;

/// <summary>
/// Wraps a VkPipelineCache, persisting it to disk between runs.
/// Cache file lives at ~/.cache/sharpinference/vulkan/&lt;deviceUUID&gt;.pipeline_cache
/// (or %LOCALAPPDATA%\sharpinference\vulkan\... on Windows). The driver gracefully
/// ignores cache contents that don't match the current device, so the file can be
/// shared without breaking on driver/GPU upgrades.
/// </summary>
public sealed class VulkanPipelineCache : IDisposable
{
    private readonly nint _device;
    private ulong _handle;
    private readonly string _cachePath;

    public ulong Handle => _handle;

    public VulkanPipelineCache(nint device, VulkanCapabilities caps)
    {
        _device = device;
        _cachePath = ResolveCachePath(caps);
        byte[]? cacheData = TryReadCacheFile();
        InitCache(cacheData);
    }

    private unsafe void InitCache(byte[]? cacheData)
    {
        nint dataPtr = 0;
        nuint dataSize = 0;
        try
        {
            if (cacheData is { Length: > 0 })
            {
                dataPtr = Marshal.AllocHGlobal(cacheData.Length);
                Marshal.Copy(cacheData, 0, dataPtr, cacheData.Length);
                dataSize = (nuint)cacheData.Length;
            }

            VkPipelineCacheCreateInfo ci = new()
            {
                sType = VkStructureType.PipelineCacheCreateInfo,
                initialDataSize = dataSize,
                pInitialData = dataPtr,
            };
            VkResult r = VulkanApi.vkCreatePipelineCache(_device, in ci, 0, out _handle);
            if (r != VkResult.Success && cacheData != null)
            {
                // Cache contents may be incompatible — try without
                ci = new VkPipelineCacheCreateInfo { sType = VkStructureType.PipelineCacheCreateInfo };
                VulkanApi.vkCreatePipelineCache(_device, in ci, 0, out _handle).ThrowOnError("vkCreatePipelineCache");
            }
            else
            {
                r.ThrowOnError("vkCreatePipelineCache");
            }
        }
        finally
        {
            if (dataPtr != 0) Marshal.FreeHGlobal(dataPtr);
        }
    }

    private byte[]? TryReadCacheFile()
    {
        try
        {
            return File.Exists(_cachePath) ? File.ReadAllBytes(_cachePath) : null;
        }
        catch
        {
            return null;
        }
    }

    private static string ResolveCachePath(VulkanCapabilities caps)
    {
        string root = Environment.OSVersion.Platform == PlatformID.Win32NT
            ? Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData)
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache");
        string dir = Path.Combine(root, "sharpinference", "vulkan");
        Directory.CreateDirectory(dir);
        string fname = $"{caps.VendorString}-{caps.DeviceId:X4}.pipeline_cache".Replace(':', '_');
        return Path.Combine(dir, fname);
    }

    /// <summary>Persists the current cache contents to disk.</summary>
    public unsafe void Persist()
    {
        if (_handle == 0) return;
        nuint sz = 0;
        VulkanApi.vkGetPipelineCacheData(_device, _handle, ref sz, 0).ThrowOnError("vkGetPipelineCacheData size");
        if (sz == 0) return;

        byte[] data = new byte[(int)sz];
        nint p = Marshal.AllocHGlobal((int)sz);
        try
        {
            VulkanApi.vkGetPipelineCacheData(_device, _handle, ref sz, p).ThrowOnError("vkGetPipelineCacheData fetch");
            Marshal.Copy(p, data, 0, (int)sz);
        }
        finally { Marshal.FreeHGlobal(p); }

        try { File.WriteAllBytes(_cachePath, data); } catch { /* best effort */ }
    }

    public void Dispose()
    {
        Persist();
        if (_handle != 0) { VulkanApi.vkDestroyPipelineCache(_device, _handle, 0); _handle = 0; }
        GC.SuppressFinalize(this);
    }

    ~VulkanPipelineCache()
    {
        if (_handle != 0) VulkanApi.vkDestroyPipelineCache(_device, _handle, 0);
    }
}
