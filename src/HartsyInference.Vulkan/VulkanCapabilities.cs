namespace HartsyInference.Vulkan;

/// <summary>Captured at startup from the chosen physical device. Drives kernel selection (FP16 / cooperative-matrix path) and validation (push-constant size budget, tile-size limits).</summary>
public sealed class VulkanCapabilities
{
    /// <summary>Human-readable device name from VkPhysicalDeviceProperties.deviceName.</summary>
    public required string DeviceName { get; init; }

    /// <summary>0x10DE NVIDIA, 0x1002 AMD, 0x8086 Intel, 0x13B5 ARM (Mali), 0x5143 Qualcomm, 0x106B Apple, etc.</summary>
    public required uint VendorId { get; init; }

    /// <summary>Driver-internal device ID. Used in pipeline cache UUID composition and telemetry.</summary>
    public required uint DeviceId { get; init; }

    /// <summary>Vulkan API version (encoded VK_VERSION).</summary>
    public required uint ApiVersion { get; init; }

    /// <summary>Discrete / Integrated / CPU / Virtual.</summary>
    public required VkPhysicalDeviceType DeviceType { get; init; }

    /// <summary>Total VRAM aggregated across DEVICE_LOCAL heaps (informational; per-allocation budget is queried separately via VK_EXT_memory_budget).</summary>
    public required ulong TotalVramBytes { get; init; }

    // ── Subgroup ────────────────────────────────────────────────────────

    /// <summary>Default subgroup size on the device (32 NV/Intel-Arc, 32 or 64 AMD RDNA, 64 GCN, 8–32 Intel iGPU).</summary>
    public required uint SubgroupSize { get; init; }
    public required uint MinSubgroupSize { get; init; }
    public required uint MaxSubgroupSize { get; init; }
    public required VkSubgroupFeatureFlags SubgroupOps { get; init; }

    // ── Compute limits ──────────────────────────────────────────────────

    public required uint MaxComputeSharedMemoryBytes { get; init; }
    public required uint MaxComputeWorkGroupInvocations { get; init; }
    public required uint MaxComputeWorkGroupSizeX { get; init; }
    public required uint MaxComputeWorkGroupSizeY { get; init; }
    public required uint MaxComputeWorkGroupSizeZ { get; init; }
    public required uint MaxComputeWorkGroupCountX { get; init; }
    public required uint MaxComputeWorkGroupCountY { get; init; }
    public required uint MaxComputeWorkGroupCountZ { get; init; }
    public required uint MaxPushConstantsSize { get; init; }
    public required uint MaxPerStageDescriptorStorageBuffers { get; init; }

    // ── Memory alignment ────────────────────────────────────────────────

    public required ulong MinStorageBufferOffsetAlignment { get; init; }
    public required ulong NonCoherentAtomSize { get; init; }
    public required nuint MinMemoryMapAlignment { get; init; }

    // ── Features ────────────────────────────────────────────────────────

    public required bool SupportsFp16 { get; init; }
    public required bool Storage16Bit { get; init; }

    /// <summary>True if the device supports the integer dot-product feature (<c>shaderIntegerDotProduct</c>, core 1.3), enabling the INT8 GEMM path via <c>dotPacked4x8</c> — the cross-vendor DP4a/IMMA equivalent.</summary>
    public required bool HasInt8DotProduct { get; init; }

    public required bool SubgroupSizeControl { get; init; }
    public required bool ComputeFullSubgroups { get; init; }
    public required bool Synchronization2 { get; init; }
    public required bool TimelineSemaphore { get; init; }
    public required bool BufferDeviceAddress { get; init; }

    // ── Optional extensions ─────────────────────────────────────────────

    public required bool HasMemoryBudget { get; init; }
    public required bool HasPushDescriptor { get; init; }
    public required bool HasCooperativeMatrix { get; init; }

    /// <summary>True if any memory type advertises both DEVICE_LOCAL and HOST_VISIBLE — the ReBAR / Smart Access Memory fast path that skips staging copies.</summary>
    public required bool HasReBar { get; init; }

    /// <summary>Compute queue family index selected during device create.</summary>
    public required uint ComputeQueueFamilyIndex { get; init; }

    /// <summary>True if the chosen queue family is compute-only (no GRAPHICS bit). Better for async-compute on AMD/NV.</summary>
    public required bool IsAsyncComputeQueue { get; init; }

    /// <summary>Returns vendor string suitable for logs / cache file names.</summary>
    public string VendorString => VendorId switch
    {
        0x10DE => "NVIDIA",
        0x1002 => "AMD",
        0x8086 => "Intel",
        0x13B5 => "ARM",
        0x5143 => "Qualcomm",
        0x106B => "Apple",
        _ => $"vendor:0x{VendorId:X4}",
    };

    public override string ToString()
        => $"Vulkan({DeviceName}, {VendorString}, {DeviceType}, " +
           $"VRAM={TotalVramBytes / (1L << 30)} GB, subgroup={SubgroupSize} " +
           $"[{MinSubgroupSize}-{MaxSubgroupSize}], FP16={SupportsFp16}, " +
           $"ReBAR={HasReBar}, CoopMat={HasCooperativeMatrix})";
}
