// Vulkan 1.3 enum + flag values used by SharpInference.Vulkan.
// Values match vulkan_core.h exactly. Only the enums actually consumed by this
// backend are defined; full enums (graphics state, image formats, etc.) are not
// included since this is a compute-only backend.
//
// See [VULKAN_COMPUTE_API.md] for the canonical enum tables.

namespace SharpInference.Vulkan;

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
    ErrorOutOfPoolMemory = -1000257000,
}

#pragma warning restore CA1707

/// <summary>VkStructureType — only the values used by the compute backend are listed.</summary>
public enum VkStructureType
{
    ApplicationInfo = 0,
    InstanceCreateInfo = 1,
    DeviceQueueCreateInfo = 2,
    DeviceCreateInfo = 3,
    SubmitInfo = 4,
    MemoryAllocateInfo = 5,
    MappedMemoryRange = 6,
    BindSparseInfo = 7,
    FenceCreateInfo = 8,
    SemaphoreCreateInfo = 9,
    EventCreateInfo = 10,
    QueryPoolCreateInfo = 11,
    BufferCreateInfo = 12,
    BufferViewCreateInfo = 13,
    ImageCreateInfo = 14,
    ShaderModuleCreateInfo = 15,
    PipelineCacheCreateInfo = 17,
    PipelineShaderStageCreateInfo = 18,
    ComputePipelineCreateInfo = 29,
    PipelineLayoutCreateInfo = 30,
    DescriptorSetLayoutCreateInfo = 32,
    DescriptorPoolCreateInfo = 33,
    DescriptorSetAllocateInfo = 34,
    WriteDescriptorSet = 35,
    CopyDescriptorSet = 36,
    CommandPoolCreateInfo = 39,
    CommandBufferAllocateInfo = 40,
    CommandBufferBeginInfo = 42,
    BufferMemoryBarrier = 44,
    MemoryBarrier = 46,

    PhysicalDeviceFeatures2 = 1000059000,
    PhysicalDeviceProperties2 = 1000059001,
    PhysicalDeviceMemoryProperties2 = 1000059006,
    PhysicalDeviceSubgroupProperties = 1000094000,

    PhysicalDeviceVulkan11Features = 1000175000,
    PhysicalDeviceVulkan11Properties = 1000175001,
    PhysicalDeviceVulkan12Features = 1000196000,
    PhysicalDeviceVulkan12Properties = 1000196001,
    PhysicalDeviceVulkan13Features = 1000241000,
    PhysicalDeviceVulkan13Properties = 1000241001,

    PipelineShaderStageRequiredSubgroupSizeCreateInfo = 1000225001,
    PhysicalDeviceSubgroupSizeControlProperties = 1000225000,

    SemaphoreTypeCreateInfo = 1000207002,
    SemaphoreSignalInfo = 1000207005,
    SemaphoreWaitInfo = 1000207004,

    PhysicalDeviceMemoryBudgetProperties = 1000237000,

    BufferMemoryBarrier2 = 1000314000,
    MemoryBarrier2 = 1000314001,
    DependencyInfo = 1000314003,
    SubmitInfo2 = 1000314004,
    SemaphoreSubmitInfo = 1000314005,
    CommandBufferSubmitInfo = 1000314006,
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

/// <summary>VkMemoryPropertyFlags.</summary>
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

/// <summary>VkMemoryHeapFlags.</summary>
[Flags]
public enum VkMemoryHeapFlags : uint
{
    None = 0,
    DeviceLocal = 0x1,
    MultiInstance = 0x2,
}

/// <summary>VkQueueFlags.</summary>
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

/// <summary>VkPhysicalDeviceType.</summary>
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

/// <summary>VkShaderStageFlags.</summary>
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

/// <summary>VkCommandPoolCreateFlags.</summary>
[Flags]
public enum VkCommandPoolCreateFlags : uint
{
    None = 0,
    Transient = 0x01,
    ResetCommandBuffer = 0x02,
    Protected = 0x04,
}

/// <summary>VkCommandBufferUsageFlags.</summary>
[Flags]
public enum VkCommandBufferUsageFlags : uint
{
    None = 0,
    OneTimeSubmit = 0x01,
    RenderPassContinue = 0x02,
    SimultaneousUse = 0x04,
}

/// <summary>VkCommandBufferLevel.</summary>
public enum VkCommandBufferLevel
{
    Primary = 0,
    Secondary = 1,
}

/// <summary>VkPipelineBindPoint.</summary>
public enum VkPipelineBindPoint
{
    Graphics = 0,
    Compute = 1,
}

/// <summary>VkPipelineShaderStageCreateFlags.</summary>
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
    public const ulong UniformRead = 0x00000040UL;
    public const ulong ShaderRead = 0x00000020UL;
    public const ulong ShaderWrite = 0x00000040UL;
    public const ulong TransferRead = 0x00000800UL;
    public const ulong TransferWrite = 0x00001000UL;
    public const ulong HostRead = 0x00002000UL;
    public const ulong HostWrite = 0x00004000UL;
    public const ulong MemoryRead = 0x00008000UL;
    public const ulong MemoryWrite = 0x00010000UL;
    public const ulong ShaderStorageRead = 0x100000000UL;
    public const ulong ShaderStorageWrite = 0x200000000UL;
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
