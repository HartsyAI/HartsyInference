using System.Runtime.InteropServices;
using HartsyInference.Core.Logging;
using static HartsyInference.Vulkan.VulkanUtilities;

namespace HartsyInference.Vulkan;

/// <summary>Picks a physical device, queries its features/limits/queue families, and creates the logical device + compute queue.</summary>
/// <remarks>Builds the Vulkan 1.3 feature pNext chain (FP16, subgroupSizeControl, synchronization2,
/// timelineSemaphore, bufferDeviceAddress) and captures full <see cref="VulkanCapabilities"/>.</remarks>
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
            (caps, memProps) = QueryAll(chosen, instance.Handle);
        }
        else
        {
            (chosen, caps, memProps) = PickBest(phys, instance.Handle);
        }

        nint device = CreateLogicalDevice(chosen, caps);
        VulkanApi.vkGetDeviceQueue(device, caps.ComputeQueueFamilyIndex, 0, out nint queue);

        VulkanDevice d = new(instance) { Capabilities = caps, MemoryProperties = memProps };
        d._physicalDevice = chosen;
        d._device = device;
        d._queue = queue;
        return d;
    }

    private static (nint, VulkanCapabilities, VkPhysicalDeviceMemoryProperties) PickBest(nint[] phys, nint instance)
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
                (VulkanCapabilities caps, VkPhysicalDeviceMemoryProperties mem) = QueryAll(pd, instance);
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
                $"No Vulkan device met HartsyInference requirements (FP16 + arithmetic/shuffle subgroup ops + compute queue). Last reject: {lastReject?.Message}");
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

    private static unsafe (VulkanCapabilities, VkPhysicalDeviceMemoryProperties) QueryAll(nint pd, nint instance)
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

        (bool hasMemoryBudget, bool hasPushDescriptor, bool hasCoopMatrix, bool hasCoopMatrix2Raw) = QueryDeviceExtensions(pd);

        // The extension being present is not enough: the matmul_coopmat shader hard-codes a
        // specific configuration (16x16x16, FP16 A/B, FP32 accumulate, subgroup scope). NVIDIA
        // always reports it, but AMD RDNA3 / Intel Arc advertise different supported sets, so
        // enumerate and confirm the exact shape before enabling the path. Otherwise pipeline
        // creation fails or the matmul silently miscomputes on non-NVIDIA hardware.
        if (hasCoopMatrix)
            hasCoopMatrix = CoopMatShapeSupported(instance, pd);

        // VK_NV_cooperative_matrix2 requires VK_KHR_cooperative_matrix as a dependency (spec),
        // so only probe it once the base coopmat1 shape is already confirmed. Like coopmat1,
        // presence of the extension doesn't guarantee a usable FP16/FP32 workgroup-scope
        // configuration exists — enumerate flexible-dimensions properties and confirm one does.
        uint coopMat2MGran = 0, coopMat2NGran = 0, coopMat2KGran = 0, coopMat2WgInvocations = 0;
        bool hasCoopMatrix2 = hasCoopMatrix && hasCoopMatrix2Raw
            && CoopMat2Supported(instance, pd, out coopMat2MGran, out coopMat2NGran, out coopMat2KGran, out coopMat2WgInvocations);

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
        // shaderIntegerDotProduct: like fp16/timeline/sync2, the NVIDIA Linux blob reports 0 through
        // the promoted Vulkan13Features struct but accepts the feature on device-create (deviation
        // #3). All NVIDIA/AMD/Intel GPUs exposing Vulkan 1.3 support integer dot product (DP4a HW),
        // so fall back to the vendor + apiVersion check, mirroring the fp16 detection above.
        bool int8dot = f13.shaderIntegerDotProduct != 0
            || (atLeast13 && props2.properties.vendorID is 0x10DE or 0x1002 or 0x8086);

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
            HasInt8DotProduct = int8dot,
            SubgroupSizeControl = sgs,
            ComputeFullSubgroups = cfs,
            Synchronization2 = sync2,
            TimelineSemaphore = ts,
            BufferDeviceAddress = bda,

            HasMemoryBudget = hasMemoryBudget,
            HasPushDescriptor = hasPushDescriptor,
            HasCooperativeMatrix = hasCoopMatrix,
            HasCooperativeMatrix2 = hasCoopMatrix2,
            CoopMat2MGranularity = coopMat2MGran,
            CoopMat2NGranularity = coopMat2NGran,
            CoopMat2KGranularity = coopMat2KGran,
            CoopMat2WorkgroupInvocations = coopMat2WgInvocations,
            HasReBar = hasReBar,

            ComputeQueueFamilyIndex = queueFamily,
            IsAsyncComputeQueue = isAsyncCompute,
            TimestampPeriod = props2.properties.limits.timestampPeriod,
        };

        return (caps, memProps);
    }

    private static unsafe (bool memBudget, bool pushDesc, bool coopMatrix, bool coopMatrix2) QueryDeviceExtensions(nint pd)
    {
        uint count = 0;
        if (VulkanApi.vkEnumerateDeviceExtensionProperties(pd, 0, ref count, 0) != VkResult.Success || count == 0)
            return (false, false, false, false);

        int sz = sizeof(VkExtensionProperties);
        nint block = Marshal.AllocHGlobal((int)count * sz);
        try
        {
            if (VulkanApi.vkEnumerateDeviceExtensionProperties(pd, 0, ref count, block) != VkResult.Success)
                return (false, false, false, false);

            bool budget = false, push = false, coop = false, coop2 = false;
            byte* p = (byte*)block;
            for (uint i = 0; i < count; i++)
            {
                VkExtensionProperties* ext = (VkExtensionProperties*)(p + i * sz);
                string name = ReadFixedString(ext->extensionName, 256);
                if (name == "VK_EXT_memory_budget") budget = true;
                else if (name == "VK_KHR_push_descriptor") push = true;
                else if (name == "VK_KHR_cooperative_matrix") coop = true;
                else if (name == "VK_NV_cooperative_matrix2") coop2 = true;
            }
            return (budget, push, coop, coop2);
        }
        finally { Marshal.FreeHGlobal(block); }
    }

    /// <summary>Confirms the device's enumerated cooperative-matrix configurations include the one the <c>matmul_coopmat</c> shader uses: 16x16x16, FP16 A and B, FP32 accumulate (C and Result), subgroup scope, non-saturating. The property query is a physical-device extension command, so it is loaded via <c>vkGetInstanceProcAddr</c>.</summary>
    private static unsafe bool CoopMatShapeSupported(nint instance, nint pd)
    {
        nint fp = VulkanApi.vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
        if (fp == 0) return false;
        delegate* unmanaged<nint, uint*, VkCooperativeMatrixPropertiesKHR*, VkResult> query =
            (delegate* unmanaged<nint, uint*, VkCooperativeMatrixPropertiesKHR*, VkResult>)fp;

        uint count = 0;
        if (query(pd, &count, null) != VkResult.Success || count == 0) return false;

        VkCooperativeMatrixPropertiesKHR[] props = new VkCooperativeMatrixPropertiesKHR[count];
        for (uint i = 0; i < count; i++) props[i].sType = VkStructureType.CooperativeMatrixPropertiesKHR;
        fixed (VkCooperativeMatrixPropertiesKHR* p = props)
        {
            if (query(pd, &count, p) != VkResult.Success) return false;
            for (uint i = 0; i < count; i++)
            {
                VkCooperativeMatrixPropertiesKHR e = p[i];
                if (e.MSize == 16 && e.NSize == 16 && e.KSize == 16
                    && e.AType == VkComponentTypeKHR.Float16 && e.BType == VkComponentTypeKHR.Float16
                    && e.CType == VkComponentTypeKHR.Float32 && e.ResultType == VkComponentTypeKHR.Float32
                    && e.scope == VkScopeKHR.Subgroup && e.saturatingAccumulation == 0)
                    return true;
            }
        }
        return false;
    }

    /// <summary>Confirms the device's enumerated <c>VK_NV_cooperative_matrix2</c> "flexible dimensions" configurations include a usable one: FP16 A and B, FP32 accumulate (C and Result), WORKGROUP scope (not subgroup — this is the architectural difference from coopmat1), non-saturating. Unlike coopmat1's fixed 16x16x16 shape, this reports granularities (M/N/K tile dims used by a kernel must be multiples of these) and the exact workgroup invocation count the driver expects for this configuration, both needed to size the <c>matmul_coopmat2</c> shader correctly.</summary>
    private static unsafe bool CoopMat2Supported(nint instance, nint pd, out uint mGranularity, out uint nGranularity, out uint kGranularity, out uint workgroupInvocations)
    {
        mGranularity = nGranularity = kGranularity = workgroupInvocations = 0;

        nint fp = VulkanApi.vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV");
        if (fp == 0) return false;
        delegate* unmanaged<nint, uint*, VkCooperativeMatrixFlexibleDimensionsPropertiesNV*, VkResult> query =
            (delegate* unmanaged<nint, uint*, VkCooperativeMatrixFlexibleDimensionsPropertiesNV*, VkResult>)fp;

        uint count = 0;
        if (query(pd, &count, null) != VkResult.Success || count == 0) return false;

        VkCooperativeMatrixFlexibleDimensionsPropertiesNV[] props = new VkCooperativeMatrixFlexibleDimensionsPropertiesNV[count];
        for (uint i = 0; i < count; i++) props[i].sType = VkStructureType.CooperativeMatrixFlexibleDimensionsPropertiesNV;
        fixed (VkCooperativeMatrixFlexibleDimensionsPropertiesNV* p = props)
        {
            if (query(pd, &count, p) != VkResult.Success) return false;
            bool found = false;
            for (uint i = 0; i < count; i++)
            {
                VkCooperativeMatrixFlexibleDimensionsPropertiesNV e = p[i];
                if (System.Environment.GetEnvironmentVariable("HARTSYINFERENCE_VK_DUMP_COOPMAT2") == "1")
                {
                    Logs.Warning($"coopmat2[{i}]: M/N/Kgran={e.MGranularity}/{e.NGranularity}/{e.KGranularity} A={e.AType} B={e.BType} C={e.CType} R={e.ResultType} sat={e.saturatingAccumulation} scope={e.scope} wgInvocations={e.workgroupInvocations}");
                }
                // Prefer the config with the largest workgroupInvocations among FP16-in/FP32-out,
                // workgroup-scope, non-saturating matches — a bigger workgroup does more MACs per
                // dispatch of the coopmat2 op, which is the whole point of workgroup (vs subgroup) scope.
                if (e.AType == VkComponentTypeKHR.Float16 && e.BType == VkComponentTypeKHR.Float16
                    && e.CType == VkComponentTypeKHR.Float32 && e.ResultType == VkComponentTypeKHR.Float32
                    && e.scope == VkScopeKHR.Workgroup && e.saturatingAccumulation == 0
                    && (!found || e.workgroupInvocations > workgroupInvocations))
                {
                    mGranularity = e.MGranularity;
                    nGranularity = e.NGranularity;
                    kGranularity = e.KGranularity;
                    workgroupInvocations = e.workgroupInvocations;
                    found = true;
                }
            }
            return found;
        }
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
            shaderIntegerDotProduct = caps.HasInt8DotProduct ? 1u : 0u,
            maintenance4 = 1u,
        };
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR fCoop = new()
        {
            sType = VkStructureType.PhysicalDeviceCooperativeMatrixFeaturesKHR,
            pNext = (nint)(&f13),
            cooperativeMatrix = caps.HasCooperativeMatrix ? 1u : 0u,
        };
        VkPhysicalDeviceCooperativeMatrix2FeaturesNV fCoop2 = new()
        {
            sType = VkStructureType.PhysicalDeviceCooperativeMatrix2FeaturesNV,
            pNext = (nint)(&fCoop),
            cooperativeMatrixWorkgroupScope = caps.HasCooperativeMatrix2 ? 1u : 0u,
            cooperativeMatrixFlexibleDimensions = caps.HasCooperativeMatrix2 ? 1u : 0u,
            cooperativeMatrixTensorAddressing = caps.HasCooperativeMatrix2 ? 1u : 0u,
        };
        VkPhysicalDeviceFeatures2 feat2 = new()
        {
            sType = VkStructureType.PhysicalDeviceFeatures2,
            pNext = caps.HasCooperativeMatrix2 ? (nint)(&fCoop2)
                  : caps.HasCooperativeMatrix ? (nint)(&fCoop)
                  : (nint)(&f13),
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
        if (caps.HasCooperativeMatrix2) extList.Add("VK_NV_cooperative_matrix2");

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
            try { VulkanApi.vkDeviceWaitIdle(_device); }
            catch (Exception ex) { Logs.Warning($"VulkanDevice.Dispose: vkDeviceWaitIdle failed on best-effort teardown: {ex.Message}"); }
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
