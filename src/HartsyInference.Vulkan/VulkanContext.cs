namespace HartsyInference.Vulkan;

/// <summary>Answers "is there a Vulkan GPU worth running on here" without building a backend, mirroring the
/// availability half of <c>HartsyInference.Cuda.CudaContext</c> so <c>auto</c> backend resolution can consider
/// Vulkan at the same cost it already considers CUDA.</summary>
public static class VulkanContext
{
    private static readonly object _lock = new();
    private static int? _deviceCount;

    /// <summary>Why <see cref="IsAvailable"/> last answered false; <c>null</c> when a device was found.</summary>
    public static string? LastUnavailableReason { get; private set; }

    /// <summary>Vulkan devices that are actual GPUs.
    ///
    /// <para>Cached: unlike CUDA's device-count query, which is a driver call, this has to start the loader and
    /// create an instance, and <see cref="BackendFactory.Resolve"/> is called from banner and cache-key paths that
    /// assume they are cheap. The answer cannot change without a restart, so the first call pays for all of them.</para>
    ///
    /// <para>A <see cref="VkPhysicalDeviceType.Cpu"/> device is excluded rather than counted. That type is a
    /// software rasterizer (Mesa lavapipe, SwiftShader), which is a CPU implementation of Vulkan and therefore
    /// slower than running on <c>HartsyInference.Cpu</c> directly. Counting it would make <c>auto</c> prefer the
    /// slow path over the fast one, and would do it silently on exactly the machines least able to afford it:
    /// Linux CI images ship lavapipe routinely.</para></summary>
    public static int GetDeviceCount()
    {
        lock (_lock)
        {
            if (_deviceCount is int cached)
            {
                return cached;
            }
            _deviceCount = CountGpus(out string? reason);
            LastUnavailableReason = reason;
            return _deviceCount.Value;
        }
    }

    /// <summary>Whether <see cref="GetDeviceCount"/> found a GPU; see <see cref="LastUnavailableReason"/> when false.</summary>
    public static bool IsAvailable() => GetDeviceCount() > 0;

    private static unsafe int CountGpus(out string? reason)
    {
        try
        {
            using VulkanInstance instance = new();
            nint[] devices = instance.EnumeratePhysicalDevices();
            if (devices.Length == 0)
            {
                reason = "the Vulkan loader started but exposed no physical devices (no ICD installed, or no GPU).";
                return 0;
            }
            int gpus = 0;
            int software = 0;
            foreach (nint pd in devices)
            {
                VulkanApi.vkGetPhysicalDeviceProperties(pd, out VkPhysicalDeviceProperties props);
                switch (props.deviceType)
                {
                    case VkPhysicalDeviceType.DiscreteGpu:
                    case VkPhysicalDeviceType.IntegratedGpu:
                    case VkPhysicalDeviceType.VirtualGpu:
                        gpus++;
                        break;
                    case VkPhysicalDeviceType.Cpu:
                        software++;
                        break;
                }
            }
            reason = gpus > 0 ? null
                : software > 0
                    ? $"the only Vulkan device(s) present are software rasterizers ({software}), which are slower "
                        + "than the CPU backend."
                    : $"none of the {devices.Length} Vulkan device(s) present reported itself as a GPU.";
            return gpus;
        }
        catch (DllNotFoundException ex)
        {
            reason = $"the Vulkan loader could not be loaded ({ex.Message}), so no Vulkan runtime is installed.";
            return 0;
        }
        catch (Exception ex)
        {
            reason = $"{ex.GetType().Name}: {ex.Message}";
            return 0;
        }
    }
}
