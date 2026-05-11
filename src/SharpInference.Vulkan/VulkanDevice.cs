using System.Runtime.InteropServices;
using static SharpInference.Vulkan.VulkanUtilities;

namespace SharpInference.Vulkan;

/// <summary>Picks a physical device, queries its features/limits/queue families, builds the Vulkan 1.3 feature pNext chain (FP16, subgroupSizeControl, synchronization2, timelineSemaphore, bufferDeviceAddress), and creates the logical device + compute queue. Captures full <see cref="VulkanCapabilities"/>.</summary>
public sealed class VulkanDevice : IDisposable
{
    private readonly VulkanInstance _instance;
    private nint _physicalDevice;
    private nint _device;
    private nint _queue;

    public nint PhysicalDevice => _physicalDevice;
    public nint Handle
    {
        get
        {
            nint h = _device;
            if (h == 0) throw new ObjectDisposedException(nameof(VulkanDevice));
            return h;
        }
    }
    public nint ComputeQueue => _queue;
    public required VulkanCapabilities Capabilities { get; init; }
    public required VkPhysicalDeviceMemoryProperties MemoryProperties { get; init; }

    private VulkanDevice(VulkanInstance instance) { _instance = instance; }

    /// <summary>Selects a physical device by ranking score and creates the logical device. Throws if no compatible device is found.</summary>
    public static VulkanDevice Create(VulkanInstance instance, int? deviceOrdinal = null)
    {
        ArgumentNullException.ThrowIfNull(instance);

        nint[] phys = instance.EnumeratePhysicalDevices();
        if (phys.Length == 0)
            throw new VulkanException(-9, "No Vulkan-capable physical devices visible to the loader (loader/ICD config issue or no GPU).");

        nint chosen;
        VulkanCapabilities caps;
        VkPhysicalDeviceMemoryProperties memProps;

        if (deviceOrdinal is int ord)
        {
            if ((uint)ord >= (uint)phys.Length)
                throw new ArgumentOutOfRangeException(nameof(deviceOrdinal), $"Vulkan device ordinal {ord} out of range; {phys.Length} device(s) present.");
            chosen = phys[ord];
            (caps, memProps) = QueryAll(chosen);
        }
        else
        {
            (chosen, caps, memProps) = PickBest(phys);
        }

        nint device = CreateLogicalDevice(chosen, caps);
        VulkanApi.vkGetDeviceQueue(device, caps.ComputeQueueFamilyIndex, 0, out nint queue);

        VulkanDevice d = new(instance) { Capabilities = caps, MemoryProperties = memProps };
        d._physicalDevice = chosen;
        d._device = device;
        d._queue = queue;
        return d;
    }

    private static (nint, VulkanCapabilities, VkPhysicalDeviceMemoryProperties) PickBest(nint[] phys)
    {
        nint best = 0;
        int bestScore = int.MinValue;
        VulkanCapabilities? bestCaps = null;
        VkPhysicalDeviceMemoryProperties bestMem = default;
        Exception? lastReject = null;

        foreach (nint pd in phys)
        {
            try
            {
                (VulkanCapabilities caps, VkPhysicalDeviceMemoryProperties mem) = QueryAll(pd);
                int score = ScoreDevice(caps);
                if (score > bestScore)
                {
                    bestScore = score; best = pd; bestCaps = caps; bestMem = mem;
                }
            }
            catch (Exception e)
            {
                lastReject = e;
            }
        }

        if (best == 0 || bestCaps is null)
            throw new VulkanException(-8,
                $"No Vulkan device met SharpInference requirements (FP16 + arithmetic/shuffle subgroup ops + compute queue). Last reject: {lastReject?.Message}");
        return (best, bestCaps, bestMem);
    }

    private static int ScoreDevice(VulkanCapabilities c)
    {
        int score = 0;
        if (c.DeviceType == VkPhysicalDeviceType.DiscreteGpu) score += 4096;
        else if (c.DeviceType == VkPhysicalDeviceType.IntegratedGpu) score += 1024;
        if (c.SupportsFp16) score += 256;
        if (c.SubgroupSizeControl) score += 256;
        if (c.HasCooperativeMatrix) score += 128;
        if (c.TotalVramBytes > (8L << 30)) score += 64;
        score += (int)Math.Min(c.TotalVramBytes / (1L << 30), int.MaxValue);

        bool subgroupOk = (c.SubgroupOps & VkSubgroupFeatureFlags.Arithmetic) != 0
                       && (c.SubgroupOps & VkSubgroupFeatureFlags.Shuffle) != 0
                       && (c.SubgroupOps & VkSubgroupFeatureFlags.ShuffleRelative) != 0;
        if (!subgroupOk) score -= 1_000_000;
        return score;
    }

    private static unsafe (VulkanCapabilities, VkPhysicalDeviceMemoryProperties) QueryAll(nint pd)
    {
        VkPhysicalDeviceSubgroupProperties subgroupProps = new()
        {
            sType = VkStructureType.PhysicalDeviceSubgroupProperties,
            pNext = 0,
        };
        VkPhysicalDeviceSubgroupSizeControlProperties sgsControlProps = new()
        {
            sType = VkStructureType.PhysicalDeviceSubgroupSizeControlProperties,
            pNext = (nint)(&subgroupProps),
        };
        VkPhysicalDeviceProperties2 props2 = new()
        {
            sType = VkStructureType.PhysicalDeviceProperties2,
            pNext = (nint)(&sgsControlProps),
        };
        VulkanApi.vkGetPhysicalDeviceProperties2(pd, ref props2);

        VkPhysicalDeviceVulkan11Features f11 = new() { sType = VkStructureType.PhysicalDeviceVulkan11Features };
        VkPhysicalDeviceVulkan12Features f12 = new() { sType = VkStructureType.PhysicalDeviceVulkan12Features, pNext = (nint)(&f11) };
        VkPhysicalDeviceVulkan13Features f13 = new() { sType = VkStructureType.PhysicalDeviceVulkan13Features, pNext = (nint)(&f12) };
        VkPhysicalDeviceFeatures2 feat2 = new() { sType = VkStructureType.PhysicalDeviceFeatures2, pNext = (nint)(&f13) };
        VulkanApi.vkGetPhysicalDeviceFeatures2(pd, ref feat2);

        VulkanApi.vkGetPhysicalDeviceMemoryProperties(pd, out VkPhysicalDeviceMemoryProperties memProps);

        uint qfCount = 0;
        VulkanApi.vkGetPhysicalDeviceQueueFamilyProperties(pd, ref qfCount, 0);
        nint qfBlock = Marshal.AllocHGlobal((int)qfCount * sizeof(VkQueueFamilyProperties));
        VkQueueFamilyProperties[] qfs;
        try
        {
            VulkanApi.vkGetPhysicalDeviceQueueFamilyProperties(pd, ref qfCount, qfBlock);
            qfs = new VkQueueFamilyProperties[qfCount];
            VkQueueFamilyProperties* src = (VkQueueFamilyProperties*)qfBlock;
            for (uint i = 0; i < qfCount; i++) qfs[i] = src[i];
        }
        finally { Marshal.FreeHGlobal(qfBlock); }

        uint queueFamily = uint.MaxValue;
        bool isAsyncCompute = false;
        for (uint i = 0; i < qfs.Length; i++)
        {
            if ((qfs[i].queueFlags & VkQueueFlags.Compute) != 0
                && (qfs[i].queueFlags & VkQueueFlags.Graphics) == 0)
            { queueFamily = i; isAsyncCompute = true; break; }
        }
        if (queueFamily == uint.MaxValue)
        {
            for (uint i = 0; i < qfs.Length; i++)
            {
                if ((qfs[i].queueFlags & VkQueueFlags.Compute) != 0)
                { queueFamily = i; break; }
            }
        }
        if (queueFamily == uint.MaxValue)
            throw new VulkanException(-8, "Physical device has no queue family with COMPUTE bit.");

        ulong vram = 0;
        bool hasReBar = false;
        for (int i = 0; i < memProps.memoryTypeCount; i++)
        {
            VkMemoryType mt = memProps.GetMemoryType(i);
            if ((mt.propertyFlags & VkMemoryPropertyFlags.DeviceLocal) != 0
                && (mt.propertyFlags & VkMemoryPropertyFlags.HostVisible) != 0)
            { hasReBar = true; }
        }
        for (int i = 0; i < memProps.memoryHeapCount; i++)
        {
            VkMemoryHeap h = memProps.GetMemoryHeap(i);
            if ((h.flags & VkMemoryHeapFlags.DeviceLocal) != 0) vram += h.size;
        }

        (bool hasMemoryBudget, bool hasPushDescriptor, bool hasCoopMatrix) = QueryDeviceExtensions(pd);

        // Some drivers (notably older NVIDIA Linux blobs) advertise apiVersion 1.3+ but
        // return zeros for shaderFloat16 / timelineSemaphore / synchronization2 / etc.
        // when queried through the v1.2/v1.3 *promoted* feature structs. The features
        // are still real and accepted on device-create. Trust apiVersion as the source of
        // truth for guaranteed-promoted core features.
        uint api = props2.properties.apiVersion;
        uint apiMajor = (api >> 22) & 0x7F;
        uint apiMinor = (api >> 12) & 0x3FF;
        bool atLeast12 = apiMajor > 1 || (apiMajor == 1 && apiMinor >= 2);
        bool atLeast13 = apiMajor > 1 || (apiMajor == 1 && apiMinor >= 3);

        bool fp16  = f12.shaderFloat16 != 0 || (atLeast12 && props2.properties.vendorID is 0x10DE or 0x1002 or 0x8086);
        bool ts    = f12.timelineSemaphore != 0 || atLeast12;
        bool bda   = f12.bufferDeviceAddress != 0 || (atLeast12 && props2.properties.vendorID is 0x10DE or 0x1002 or 0x8086);
        bool sgs   = f13.subgroupSizeControl != 0 || atLeast13;
        bool cfs   = f13.computeFullSubgroups != 0 || atLeast13;
        bool sync2 = f13.synchronization2 != 0 || atLeast13;

        VulkanCapabilities caps = new()
        {
            DeviceName = ReadFixedString(props2.properties.deviceName, 256),
            VendorId = props2.properties.vendorID,
            DeviceId = props2.properties.deviceID,
            ApiVersion = props2.properties.apiVersion,
            DeviceType = props2.properties.deviceType,
            TotalVramBytes = vram,

            SubgroupSize = subgroupProps.subgroupSize,
            MinSubgroupSize = sgsControlProps.minSubgroupSize == 0 ? subgroupProps.subgroupSize : sgsControlProps.minSubgroupSize,
            MaxSubgroupSize = sgsControlProps.maxSubgroupSize == 0 ? subgroupProps.subgroupSize : sgsControlProps.maxSubgroupSize,
            SubgroupOps = subgroupProps.supportedOperations,

            MaxComputeSharedMemoryBytes = props2.properties.limits.maxComputeSharedMemorySize,
            MaxComputeWorkGroupInvocations = props2.properties.limits.maxComputeWorkGroupInvocations,
            MaxComputeWorkGroupSizeX = props2.properties.limits.maxComputeWorkGroupSizeX,
            MaxComputeWorkGroupSizeY = props2.properties.limits.maxComputeWorkGroupSizeY,
            MaxComputeWorkGroupSizeZ = props2.properties.limits.maxComputeWorkGroupSizeZ,
            MaxComputeWorkGroupCountX = props2.properties.limits.maxComputeWorkGroupCountX,
            MaxComputeWorkGroupCountY = props2.properties.limits.maxComputeWorkGroupCountY,
            MaxComputeWorkGroupCountZ = props2.properties.limits.maxComputeWorkGroupCountZ,
            MaxPushConstantsSize = props2.properties.limits.maxPushConstantsSize,
            MaxPerStageDescriptorStorageBuffers = props2.properties.limits.maxPerStageDescriptorStorageBuffers,

            MinStorageBufferOffsetAlignment = props2.properties.limits.minStorageBufferOffsetAlignment,
            NonCoherentAtomSize = props2.properties.limits.nonCoherentAtomSize,
            MinMemoryMapAlignment = props2.properties.limits.minMemoryMapAlignment,

            SupportsFp16 = fp16,
            Storage16Bit = f11.storageBuffer16BitAccess != 0,
            SubgroupSizeControl = sgs,
            ComputeFullSubgroups = cfs,
            Synchronization2 = sync2,
            TimelineSemaphore = ts,
            BufferDeviceAddress = bda,

            HasMemoryBudget = hasMemoryBudget,
            HasPushDescriptor = hasPushDescriptor,
            HasCooperativeMatrix = hasCoopMatrix,
            HasReBar = hasReBar,

            ComputeQueueFamilyIndex = queueFamily,
            IsAsyncComputeQueue = isAsyncCompute,
        };

        return (caps, memProps);
    }

    private static unsafe (bool memBudget, bool pushDesc, bool coopMatrix) QueryDeviceExtensions(nint pd)
    {
        uint count = 0;
        if (VulkanApi.vkEnumerateDeviceExtensionProperties(pd, 0, ref count, 0) != VkResult.Success || count == 0)
            return (false, false, false);

        int sz = sizeof(VkExtensionProperties);
        nint block = Marshal.AllocHGlobal((int)count * sz);
        try
        {
            if (VulkanApi.vkEnumerateDeviceExtensionProperties(pd, 0, ref count, block) != VkResult.Success)
                return (false, false, false);

            bool budget = false, push = false, coop = false;
            byte* p = (byte*)block;
            for (uint i = 0; i < count; i++)
            {
                VkExtensionProperties* ext = (VkExtensionProperties*)(p + i * sz);
                string name = ReadFixedString(ext->extensionName, 256);
                if (name == "VK_EXT_memory_budget") budget = true;
                else if (name == "VK_KHR_push_descriptor") push = true;
                else if (name == "VK_KHR_cooperative_matrix") coop = true;
            }
            return (budget, push, coop);
        }
        finally { Marshal.FreeHGlobal(block); }
    }

    private static unsafe nint CreateLogicalDevice(nint pd, VulkanCapabilities caps)
    {
        VkPhysicalDeviceVulkan11Features f11 = new()
        {
            sType = VkStructureType.PhysicalDeviceVulkan11Features,
            storageBuffer16BitAccess = caps.Storage16Bit ? 1u : 0u,
        };
        VkPhysicalDeviceVulkan12Features f12 = new()
        {
            sType = VkStructureType.PhysicalDeviceVulkan12Features,
            pNext = (nint)(&f11),
            shaderFloat16 = caps.SupportsFp16 ? 1u : 0u,
            timelineSemaphore = caps.TimelineSemaphore ? 1u : 0u,
            bufferDeviceAddress = 0u,
        };
        VkPhysicalDeviceVulkan13Features f13 = new()
        {
            sType = VkStructureType.PhysicalDeviceVulkan13Features,
            pNext = (nint)(&f12),
            subgroupSizeControl = caps.SubgroupSizeControl ? 1u : 0u,
            computeFullSubgroups = caps.ComputeFullSubgroups ? 1u : 0u,
            synchronization2 = caps.Synchronization2 ? 1u : 0u,
            maintenance4 = 1u,
        };
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR fCoop = new()
        {
            sType = VkStructureType.PhysicalDeviceCooperativeMatrixFeaturesKHR,
            pNext = (nint)(&f13),
            cooperativeMatrix = caps.HasCooperativeMatrix ? 1u : 0u,
        };
        VkPhysicalDeviceFeatures2 feat2 = new()
        {
            sType = VkStructureType.PhysicalDeviceFeatures2,
            pNext = caps.HasCooperativeMatrix ? (nint)(&fCoop) : (nint)(&f13),
        };

        float prio = 1.0f;
        VkDeviceQueueCreateInfo queueCi = new()
        {
            sType = VkStructureType.DeviceQueueCreateInfo,
            queueFamilyIndex = caps.ComputeQueueFamilyIndex,
            queueCount = 1,
            pQueuePriorities = (nint)(&prio),
        };

        List<string> extList = new();
        if (caps.HasMemoryBudget) extList.Add("VK_EXT_memory_budget");
        if (caps.HasPushDescriptor) extList.Add("VK_KHR_push_descriptor");
        if (caps.HasCooperativeMatrix) extList.Add("VK_KHR_cooperative_matrix");

        using PinnedStringArray ext = new(extList);

        VkDeviceCreateInfo devCi = new()
        {
            sType = VkStructureType.DeviceCreateInfo,
            pNext = (nint)(&feat2),
            queueCreateInfoCount = 1,
            pQueueCreateInfos = (nint)(&queueCi),
            enabledExtensionCount = ext.Count,
            ppEnabledExtensionNames = ext.Pointers,
            pEnabledFeatures = 0,
        };

        VulkanApi.vkCreateDevice(pd, in devCi, 0, out nint device).ThrowOnError("vkCreateDevice");
        return device;
    }

    /// <summary>Blocks until all GPU work submitted to any queue on this device has completed.</summary>
    public void WaitIdle() => VulkanApi.vkDeviceWaitIdle(Handle).ThrowOnError("vkDeviceWaitIdle");

    public void Dispose()
    {
        if (_device != 0)
        {
            try { VulkanApi.vkDeviceWaitIdle(_device); } catch { /* swallow during dispose */ }
            VulkanApi.vkDestroyDevice(_device, 0);
            _device = 0;
        }
        GC.SuppressFinalize(this);
    }

    ~VulkanDevice()
    {
        if (_device != 0) VulkanApi.vkDestroyDevice(_device, 0);
    }
}
