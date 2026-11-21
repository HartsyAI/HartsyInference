using System.Security;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;

namespace HartsyInference.Vulkan;

/// <summary>Wraps a VkPipelineCache and persists it between runs at a per-device cache file.</summary>
/// <remarks>The file is checked before the driver sees it (<see cref="VulkanPipelineCacheFile"/>): a cut-short or
/// altered file, or one from another device or driver build, starts an empty cache instead.</remarks>
public sealed class VulkanPipelineCache : IDisposable
{
    private readonly nint _device;
    private ulong _handle;
    private readonly string _cachePath;

    public VulkanPipelineCache(nint device, VulkanCapabilities caps) : this(device, caps, null)
    {
    }

    /// <summary>As the public constructor, with the cache file in <paramref name="cacheDirectory"/> when one is given.</summary>
    internal VulkanPipelineCache(nint device, VulkanCapabilities caps, string? cacheDirectory)
    {
        _device = device;
        _cachePath = ResolveCachePath(caps, cacheDirectory);
        VulkanPipelineCacheLoad load = VulkanPipelineCacheFile.Load(_cachePath,
            new VulkanPipelineCacheIdentity(caps.VendorId, caps.DeviceId, caps.PipelineCacheUuid));
        if (load.IsForAnotherDevice)
        {
            Logs.Info($"VulkanPipelineCache: {_cachePath} was written for another device or driver build "
                + $"({load.Status}); starting with an empty cache.");
        }
        else if (load.Status == VulkanPipelineCacheLoadStatus.Unreadable)
        {
            Logs.Warning($"VulkanPipelineCache: could not read {_cachePath} ({load.Detail}); starting with an empty cache.");
        }
        else if (load.Status is not (VulkanPipelineCacheLoadStatus.Loaded or VulkanPipelineCacheLoadStatus.Missing))
        {
            Logs.Warning($"VulkanPipelineCache: {_cachePath} failed its integrity check ({load.Status}); "
                + "starting with an empty cache.");
        }
        InitialDataBytes = load.Blob.Length;
        InitCache(load.Blob);
    }

    public ulong Handle => _handle;

    /// <summary>Filesystem path of the on-disk pipeline-cache file (set at construction). Exposed so tests can verify
    /// persist+reload behavior across backend instances.</summary>
    public string CachePath => _cachePath;

    /// <summary>Bytes handed to the driver as initial cache data at construction, 0 when nothing was read. A driver
    /// that persists no pipelines leaves the blob at its 32-byte header, so the reload cannot be observed from the
    /// file's size alone — this is what makes that assertion real on such a driver.</summary>
    public int InitialDataBytes { get; }

    /// <summary>Persists the current cache contents to disk; a failed write is logged and leaves the previous file.</summary>
    public unsafe void Persist()
    {
        if (_handle == 0) return;
        nuint sz = 0;
        VulkanApi.vkGetPipelineCacheData(_device, _handle, ref sz, 0).ThrowOnError("vkGetPipelineCacheData size");
        if (sz == 0) return;

        byte[] data = new byte[(int)sz];
        fixed (byte* p = data)
        {
            VulkanApi.vkGetPipelineCacheData(_device, _handle, ref sz, (nint)p).ThrowOnError("vkGetPipelineCacheData fetch");
        }

        try
        {
            VulkanPipelineCacheFile.WriteAtomic(_cachePath, data.AsSpan(0, (int)sz));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or NotSupportedException or SecurityException)
        {
            Logs.Warning($"VulkanPipelineCache: failed to persist cache to {_cachePath}: {ex.Message}");
        }
    }

    private unsafe void InitCache(ReadOnlyMemory<byte> blob)
    {
        fixed (byte* data = blob.Span)
        {
            VkPipelineCacheCreateInfo ci = new()
            {
                sType = VkStructureType.PipelineCacheCreateInfo,
                initialDataSize = (nuint)blob.Length,
                pInitialData = (nint)data,
            };
            VkResult r = VulkanApi.vkCreatePipelineCache(_device, in ci, 0, out _handle);
            if (r != VkResult.Success && blob.Length > 0)
            {
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
    }

    private static string ResolveCachePath(VulkanCapabilities caps, string? cacheDirectory)
    {
        string? dir = cacheDirectory ?? EngineKnobs.VkPipelineCacheDir.Value;
        if (string.IsNullOrWhiteSpace(dir))
        {
            string root = Environment.OSVersion.Platform == PlatformID.Win32NT
                ? Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData)
                : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache");
            dir = Path.Combine(root, "hartsyinference", "vulkan");
        }
        Directory.CreateDirectory(dir);
        string fname = $"{caps.VendorString}-{caps.DeviceId:X4}{VulkanPipelineCacheFile.Extension}".Replace(':', '_');
        return Path.Combine(dir, fname);
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
