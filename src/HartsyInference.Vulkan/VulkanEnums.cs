// Vulkan 1.3 enum + flag values used by HartsyInference.Vulkan.
// Values match vulkan_core.h exactly. Only the enums actually consumed by this
// backend are defined; full enums (graphics state, image formats, etc.) are not
// included since this is a compute-only backend.
//
// See [VULKAN_COMPUTE_API.md] for the canonical enum tables.

namespace HartsyInference.Vulkan;

#pragma warning disable CA1707  // Underscore-style enum names mirror the C VkResult values for source-grep parity with the spec.

/// <summary>VkResult: negative = error, non-negative = success/partial.</summary>
public enum VkResult
{
    Success = 0,
    NotReady = 1,
    Timeout = 2,
    EventSet = 3,
    EventReset = 4,
    Incomplete = 5,
    ErrorOutOfHostMemory = -1,
    ErrorOutOfDeviceMemory = -2,
    ErrorInitializationFailed = -3,
    ErrorDeviceLost = -4,
    ErrorMemoryMapFailed = -5,
    ErrorLayerNotPresent = -6,
    ErrorExtensionNotPresent = -7,
    ErrorFeatureNotPresent = -8,
    ErrorIncompatibleDriver = -9,
    ErrorTooManyObjects = -10,
    ErrorFormatNotSupported = -11,
    ErrorFragmentedPool = -12,
    ErrorUnknown = -13,
    ErrorOutOfPoolMemory = -1000069000,
}

#pragma warning restore CA1707

/// <summary>VkStructureType — only the values used by the compute backend are listed.</summary>
/// <remarks>These are the values the Vulkan headers define, and several here were not. A wrong one does not fail
/// loudly: a driver that skips an unrecognized <c>pNext</c> struct simply ignores it, so a feature chain with the
/// wrong sType queries as all-zero and enables nothing, while the device keeps working because the driver allows
/// what was never requested. That is what made the 1.2/1.3 feature structs look like a driver that "reports 0" —
/// it was this. Synchronization validation names every one of them by VUID, which is how these were found.</remarks>
public enum VkStructureType
{
    ApplicationInfo = 0,
    InstanceCreateInfo = 1,
    DeviceQueueCreateInfo = 2,
    DeviceCreateInfo = 3,
    MemoryAllocateInfo = 5,
    MappedMemoryRange = 6,
    FenceCreateInfo = 8,
    SemaphoreCreateInfo = 9,
    QueryPoolCreateInfo = 11,
    BufferCreateInfo = 12,
    ShaderModuleCreateInfo = 16,
    PipelineCacheCreateInfo = 17,
    PipelineShaderStageCreateInfo = 18,
    ComputePipelineCreateInfo = 29,
    PipelineLayoutCreateInfo = 30,
    DescriptorSetLayoutCreateInfo = 32,
    DescriptorPoolCreateInfo = 33,
    DescriptorSetAllocateInfo = 34,
    WriteDescriptorSet = 35,
    CommandPoolCreateInfo = 39,
    CommandBufferAllocateInfo = 40,
    CommandBufferBeginInfo = 42,

    PhysicalDeviceFeatures2 = 1000059000,
    PhysicalDeviceProperties2 = 1000059001,
    PhysicalDeviceMemoryProperties2 = 1000059006,
    PhysicalDeviceSubgroupProperties = 1000094000,

    PhysicalDeviceVulkan11Features = 49,
    PhysicalDeviceVulkan12Features = 51,
    PhysicalDeviceVulkan13Features = 53,

    PipelineShaderStageRequiredSubgroupSizeCreateInfo = 1000225001,
    PhysicalDeviceSubgroupSizeControlProperties = 1000225000,

    SemaphoreTypeCreateInfo = 1000207002,
    SemaphoreWaitInfo = 1000207004,

    PhysicalDeviceMemoryBudgetProperties = 1000237000,

    MemoryBarrier2 = 1000314000,
    BufferMemoryBarrier2 = 1000314001,
    DependencyInfo = 1000314003,
    SubmitInfo2 = 1000314004,
    SemaphoreSubmitInfo = 1000314005,
    CommandBufferSubmitInfo = 1000314006,

    PhysicalDeviceCooperativeMatrixFeaturesKHR = 1000506000,
    CooperativeMatrixPropertiesKHR = 1000506001,

    PhysicalDeviceCooperativeMatrix2FeaturesNV = 1000593000,
    PhysicalDeviceShaderFloat8FeaturesEXT = 1000567000,
    CooperativeMatrixFlexibleDimensionsPropertiesNV = 1000593001,
}

/// <summary>VkBufferUsageFlags — bits we actually use.</summary>
[Flags]
public enum VkBufferUsageFlags : uint
{
    None = 0,
    TransferSrc = 0x0001,
    TransferDst = 0x0002,
    UniformTexelBuffer = 0x0004,
    StorageTexelBuffer = 0x0008,
    UniformBuffer = 0x0010,
    StorageBuffer = 0x0020,
    IndexBuffer = 0x0040,
    VertexBuffer = 0x0080,
    IndirectBuffer = 0x0100,
    ShaderDeviceAddress = 0x00020000,
}

[Flags]
public enum VkMemoryPropertyFlags : uint
{
    None = 0,
    DeviceLocal = 0x01,
    HostVisible = 0x02,
    HostCoherent = 0x04,
    HostCached = 0x08,
    LazilyAllocated = 0x10,
    Protected = 0x20,
    DeviceCoherentAmd = 0x40,
    DeviceUncachedAmd = 0x80,
}

[Flags]
public enum VkMemoryHeapFlags : uint
{
    None = 0,
    DeviceLocal = 0x1,
    MultiInstance = 0x2,
}

[Flags]
public enum VkQueueFlags : uint
{
    None = 0,
    Graphics = 0x1,
    Compute = 0x2,
    Transfer = 0x4,
    SparseBinding = 0x8,
    Protected = 0x10,
}

public enum VkPhysicalDeviceType
{
    Other = 0,
    IntegratedGpu = 1,
    DiscreteGpu = 2,
    VirtualGpu = 3,
    Cpu = 4,
}

/// <summary>VkDescriptorType — bits we use for compute (storage and uniform buffers).</summary>
public enum VkDescriptorType
{
    UniformBuffer = 6,
    StorageBuffer = 7,
}

[Flags]
public enum VkShaderStageFlags : uint
{
    None = 0,
    Compute = 0x20,
    All = 0x7FFFFFFF,
}

/// <summary>VkSubgroupFeatureFlags — feature bits returned in VkPhysicalDeviceSubgroupProperties.supportedOperations.</summary>
[Flags]
public enum VkSubgroupFeatureFlags : uint
{
    None = 0,
    Basic = 0x01,
    Vote = 0x02,
    Arithmetic = 0x04,
    Ballot = 0x08,
    Shuffle = 0x10,
    ShuffleRelative = 0x20,
    Clustered = 0x40,
    Quad = 0x80,
}

[Flags]
public enum VkCommandPoolCreateFlags : uint
{
    None = 0,
    Transient = 0x01,
    ResetCommandBuffer = 0x02,
    Protected = 0x04,
}

[Flags]
public enum VkCommandBufferUsageFlags : uint
{
    None = 0,
    OneTimeSubmit = 0x01,
    RenderPassContinue = 0x02,
    SimultaneousUse = 0x04,
}

public enum VkCommandBufferLevel
{
    Primary = 0,
    Secondary = 1,
}

public enum VkPipelineBindPoint
{
    Graphics = 0,
    Compute = 1,
}

[Flags]
public enum VkPipelineShaderStageCreateFlags : uint
{
    None = 0,
    AllowVaryingSubgroupSize = 0x01,
    RequireFullSubgroups = 0x02,
}

/// <summary>VkSemaphoreType — for timeline vs binary semaphores.</summary>
public enum VkSemaphoreType
{
    Binary = 0,
    Timeline = 1,
}

/// <summary>VkSharingMode — exclusive vs concurrent across queue families.</summary>
public enum VkSharingMode
{
    Exclusive = 0,
    Concurrent = 1,
}

/// <summary>VkPipelineStageFlags2 — sync2 64-bit pipeline-stage bitmask. Only compute/transfer/host bits used.</summary>
public static class VkPipelineStageFlags2
{
    public const ulong None = 0;
    public const ulong TopOfPipe = 0x00000001UL;
    public const ulong DrawIndirect = 0x00000002UL;
    public const ulong VertexInput = 0x00000004UL;
    public const ulong Transfer = 0x00001000UL;
    public const ulong ComputeShader = 0x00000800UL;
    public const ulong AllCommands = 0x00010000UL;
    public const ulong Host = 0x00004000UL;
    public const ulong Copy = 0x100000000UL;
    public const ulong Resolve = 0x200000000UL;
    public const ulong Blit = 0x400000000UL;
    public const ulong Clear = 0x800000000UL;
    public const ulong AllTransfer = Transfer;
}

/// <summary>VkAccessFlags2 — sync2 64-bit access bitmask.</summary>
public static class VkAccessFlags2
{
    public const ulong None = 0;
    public const ulong IndirectCommandRead = 0x00000001UL;
    public const ulong IndexRead = 0x00000002UL;
    public const ulong UniformRead = 0x00000008UL;
    public const ulong ShaderRead = 0x00000020UL;
    public const ulong ShaderWrite = 0x00000040UL;
    public const ulong TransferRead = 0x00000800UL;
    public const ulong TransferWrite = 0x00001000UL;
    public const ulong HostRead = 0x00002000UL;
    public const ulong HostWrite = 0x00004000UL;
    public const ulong MemoryRead = 0x00008000UL;
    public const ulong MemoryWrite = 0x00010000UL;
    public const ulong ShaderSampledRead = 0x100000000UL;
    public const ulong ShaderStorageRead = 0x200000000UL;
    public const ulong ShaderStorageWrite = 0x400000000UL;
}

/// <summary>VkComponentTypeKHR — cooperative-matrix element types (only the FP types used here).</summary>
public enum VkComponentTypeKHR
{
    Float16 = 0,
    Float32 = 1,
    Float64 = 2,
    Sint8 = 3,
    Sint16 = 4,
    Sint32 = 5,
    Sint64 = 6,
    Uint8 = 7,
    Uint16 = 8,
    Uint32 = 9,
    Uint64 = 10,
    Float8E4M3 = 1000491002,
    Float8E5M2 = 1000491003,
}

/// <summary>VkScopeKHR — execution scope a cooperative matrix spans. coopmat1 uses subgroup scope.</summary>
public enum VkScopeKHR
{
    Device = 1,
    Workgroup = 2,
    Subgroup = 3,
    QueueFamily = 5,
}

/// <summary>VK_WHOLE_SIZE constant for VkDeviceSize ranges.</summary>
public static class VkConstants
{
    public const ulong WholeSize = 0xFFFFFFFFFFFFFFFFUL;
    public const uint MaxMemoryTypes = 32;
    public const uint MaxMemoryHeaps = 16;
    public const uint MaxPhysicalDeviceNameSize = 256;
    public const uint UuidSize = 16;
    public const uint LuidSize = 8;
    public const uint MaxExtensionNameSize = 256;
    public const uint MaxDescriptionSize = 256;
    public const uint QueueFamilyIgnored = 0xFFFFFFFFu;
    public const uint True = 1;
    public const uint False = 0;
}
