namespace HartsyInference.Vulkan;

/// <summary>Captured at startup from the chosen physical device; drives kernel selection and validation.</summary>
/// <remarks>Used for FP16 / cooperative-matrix path selection and validation (push-constant size budget,
/// tile-size limits).</remarks>
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

    /// <summary>Smallest subgroup size the device can be configured for via subgroup-size control.</summary>
    public required uint MinSubgroupSize { get; init; }

    /// <summary>Largest subgroup size the device can be configured for via subgroup-size control.</summary>
    public required uint MaxSubgroupSize { get; init; }

    /// <summary>Subgroup operation categories the device's shaders can use (arithmetic, shuffle, ballot, etc.).</summary>
    public required VkSubgroupFeatureFlags SubgroupOps { get; init; }

    // ── Compute limits ──────────────────────────────────────────────────

    /// <summary>Maximum shared-memory (workgroup-local) allocation a compute shader may declare, in bytes.</summary>
    public required uint MaxComputeSharedMemoryBytes { get; init; }

    /// <summary>Maximum total invocations (threads) per workgroup.</summary>
    public required uint MaxComputeWorkGroupInvocations { get; init; }

    /// <summary>Maximum local workgroup size along X.</summary>
    public required uint MaxComputeWorkGroupSizeX { get; init; }

    /// <summary>Maximum local workgroup size along Y.</summary>
    public required uint MaxComputeWorkGroupSizeY { get; init; }

    /// <summary>Maximum local workgroup size along Z.</summary>
    public required uint MaxComputeWorkGroupSizeZ { get; init; }

    /// <summary>Maximum dispatch group count along X.</summary>
    public required uint MaxComputeWorkGroupCountX { get; init; }

    /// <summary>Maximum dispatch group count along Y.</summary>
    public required uint MaxComputeWorkGroupCountY { get; init; }

    /// <summary>Maximum dispatch group count along Z.</summary>
    public required uint MaxComputeWorkGroupCountZ { get; init; }

    /// <summary>Maximum total size of a pipeline's push-constant range, in bytes.</summary>
    public required uint MaxPushConstantsSize { get; init; }

    /// <summary>Maximum storage-buffer descriptors bindable to a single shader stage.</summary>
    public required uint MaxPerStageDescriptorStorageBuffers { get; init; }

    // ── Memory alignment ────────────────────────────────────────────────

    /// <summary>Required alignment, in bytes, for storage-buffer descriptor offsets.</summary>
    public required ulong MinStorageBufferOffsetAlignment { get; init; }

    /// <summary>Granularity, in bytes, at which non-coherent mapped-memory flush/invalidate ranges must be aligned.</summary>
    public required ulong NonCoherentAtomSize { get; init; }

    /// <summary>Minimum alignment the driver guarantees for a mapped-memory pointer.</summary>
    public required nuint MinMemoryMapAlignment { get; init; }

    // ── Features ────────────────────────────────────────────────────────

    /// <summary>Whether the device supports FP16 shader arithmetic (<c>shaderFloat16</c>).</summary>
    public required bool SupportsFp16 { get; init; }

    /// <summary>Whether storage buffers may use 16-bit types (<c>storageBuffer16BitAccess</c>).</summary>
    public required bool Storage16Bit { get; init; }

    /// <summary>True if the device supports <c>shaderIntegerDotProduct</c> (core 1.3), enabling the INT8 GEMM path via <c>dotPacked4x8</c>.</summary>
    public required bool HasInt8DotProduct { get; init; }

    /// <summary>Whether the device supports configuring a required subgroup size per shader stage.</summary>
    public required bool SubgroupSizeControl { get; init; }

    /// <summary>Whether the device supports the "full subgroups" compute shader execution mode.</summary>
    public required bool ComputeFullSubgroups { get; init; }

    /// <summary>Whether the device supports Vulkan 1.3 core synchronization2 (barrier2/submit2).</summary>
    public required bool Synchronization2 { get; init; }

    /// <summary>Whether the device supports timeline semaphores.</summary>
    public required bool TimelineSemaphore { get; init; }

    /// <summary>Whether the device supports buffer device address (GPU-visible buffer pointers).</summary>
    public required bool BufferDeviceAddress { get; init; }

    // ── Optional extensions ─────────────────────────────────────────────

    /// <summary>Whether <c>VK_EXT_memory_budget</c> is available for live heap-usage/budget queries.</summary>
    public required bool HasMemoryBudget { get; init; }

    /// <summary>Whether <c>VK_KHR_push_descriptor</c> (or Vulkan 1.4 core push descriptors) is available.</summary>
    public required bool HasPushDescriptor { get; init; }

    /// <summary>Whether <c>VK_KHR_cooperative_matrix</c> is available and exposes the shape this backend needs.</summary>
    public required bool HasCooperativeMatrix { get; init; }

    /// <summary>Whether <c>VK_NV_cooperative_matrix2</c> is available and exposes a usable FP16/FP32
    /// workgroup-scope "flexible dimensions" configuration. NVIDIA-only (revision 1 confirmed on both
    /// RTX 4090 and RTX 3060); architecturally distinct from <see cref="HasCooperativeMatrix"/> —
    /// workgroup scope + tensor-layout addressing rather than subgroup-scope fixed 16x16x16 fragments.</summary>
    public required bool HasCooperativeMatrix2 { get; init; }

    /// <summary>M-dimension granularity for the coopmat2 configuration selected in
    /// <see cref="HasCooperativeMatrix2"/> — kernel M tile size must be a multiple of this. 0 if unsupported.</summary>
    public required uint CoopMat2MGranularity { get; init; }

    /// <summary>N-dimension granularity for the selected coopmat2 configuration. 0 if unsupported.</summary>
    public required uint CoopMat2NGranularity { get; init; }

    /// <summary>K-dimension granularity for the selected coopmat2 configuration. 0 if unsupported.</summary>
    public required uint CoopMat2KGranularity { get; init; }

    /// <summary>Exact workgroup invocation count (local_size_x * y * z) the driver expects for this
    /// coopmat2 configuration's <c>gl_ScopeWorkgroup</c> matrices. 0 if unsupported.</summary>
    public required uint CoopMat2WorkgroupInvocations { get; init; }

    /// <summary>True if any memory type advertises both DEVICE_LOCAL and HOST_VISIBLE — the ReBAR / Smart Access Memory fast path that skips staging copies.</summary>
    public required bool HasReBar { get; init; }

    /// <summary>Compute queue family index selected during device create.</summary>
    public required uint ComputeQueueFamilyIndex { get; init; }

    /// <summary>True if the chosen queue family is compute-only (no GRAPHICS bit). Better for async-compute on AMD/NV.</summary>
    public required bool IsAsyncComputeQueue { get; init; }

    /// <summary>Nanoseconds per tick of <c>vkCmdWriteTimestamp2</c> results (<c>VkPhysicalDeviceLimits.timestampPeriod</c>).
    /// Used by <see cref="VulkanGpuTimer"/> to convert raw timestamp deltas into wall-clock GPU execution time.</summary>
    public required float TimestampPeriod { get; init; }

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
           $"ReBAR={HasReBar}, CoopMat={HasCooperativeMatrix}, CoopMat2={HasCooperativeMatrix2})";
}
