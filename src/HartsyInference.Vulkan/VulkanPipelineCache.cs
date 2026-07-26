using System.Runtime.InteropServices;
using HartsyInference.Core.Logging;

namespace HartsyInference.Vulkan;

/// <summary>Wraps a VkPipelineCache and persists it between runs at a per-device cache file.</summary>
/// <remarks>The driver gracefully ignores cache contents that don't match the current device, so the file
/// survives driver/GPU upgrades.</remarks>
public sealed class VulkanPipelineCache : IDisposable
{
    private readonly nint _device;
    private ulong _handle;
    private readonly string _cachePath;

    public ulong Handle => _handle;

    /// <summary>Filesystem path of the on-disk pipeline-cache blob (set at construction). Exposed so
    /// tests can verify persist+reload behavior across backend instances.</summary>
    public string CachePath => _cachePath;

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
                Logs.Warning($"VulkanPipelineCache: cached data at {_cachePath} was rejected by the driver " +
                    $"(result={r}); falling back to an empty cache.");
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
        catch (Exception ex)
        {
            Logs.Warning($"VulkanPipelineCache: failed to read cache file {_cachePath}, starting with an empty cache: {ex.Message}");
            return null;
        }
    }

    private static string ResolveCachePath(VulkanCapabilities caps)
    {
        string root = Environment.OSVersion.Platform == PlatformID.Win32NT
            ? Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData)
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache");
        string dir = Path.Combine(root, "hartsyinference", "vulkan");
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

        try { File.WriteAllBytes(_cachePath, data); }
        catch (Exception ex) { Logs.Warning($"VulkanPipelineCache: failed to persist cache to {_cachePath}: {ex.Message}"); }
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
