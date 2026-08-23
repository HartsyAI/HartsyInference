using System.Runtime.InteropServices;

// Vulkan 1.3 struct layouts used by HartsyInference.Vulkan.
// Layouts match vulkan_core.h exactly (no implicit padding past pNext on x86_64).
// All structs are blittable; no marshalling on the P/Invoke boundary.
//
// See [VULKAN_COMPUTE_API.md] for field-by-field documentation.

namespace HartsyInference.Vulkan;

[StructLayout(LayoutKind.Sequential)]
public struct VkExtent3D
{
    public uint width;
    public uint height;
    public uint depth;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkApplicationInfo
{
    public VkStructureType sType;
    public nint pNext;
    public nint pApplicationName;
    public uint applicationVersion;
    public nint pEngineName;
    public uint engineVersion;
    public uint apiVersion;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkInstanceCreateInfo
{
    public VkStructureType sType;
    public nint pNext;
    public uint flags;
    public nint pApplicationInfo;
    public uint enabledLayerCount;
    public nint ppEnabledLayerNames;
    public uint enabledExtensionCount;
    public nint ppEnabledExtensionNames;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkPhysicalDeviceFeatures
{
    // Full struct is 55 VkBool32 fields; we keep it as a fixed-size opaque blob.
    // The chained Vulkan11/12/13 features structs are what we actually populate.
    public uint robustBufferAccess;
    public uint fullDrawIndexUint32;
    public uint imageCubeArray;
    public uint independentBlend;
    public uint geometryShader;
    public uint tessellationShader;
    public uint sampleRateShading;
    public uint dualSrcBlend;
    public uint logicOp;
    public uint multiDrawIndirect;
    public uint drawIndirectFirstInstance;
    public uint depthClamp;
    public uint depthBiasClamp;
    public uint fillModeNonSolid;
    public uint depthBounds;
    public uint wideLines;
    public uint largePoints;
    public uint alphaToOne;
    public uint multiViewport;
    public uint samplerAnisotropy;
    public uint textureCompressionETC2;
    public uint textureCompressionASTC_LDR;
    public uint textureCompressionBC;
    public uint occlusionQueryPrecise;
    public uint pipelineStatisticsQuery;
    public uint vertexPipelineStoresAndAtomics;
    public uint fragmentStoresAndAtomics;
    public uint shaderTessellationAndGeometryPointSize;
    public uint shaderImageGatherExtended;
    public uint shaderStorageImageExtendedFormats;
    public uint shaderStorageImageMultisample;
    public uint shaderStorageImageReadWithoutFormat;
    public uint shaderStorageImageWriteWithoutFormat;
    public uint shaderUniformBufferArrayDynamicIndexing;
    public uint shaderSampledImageArrayDynamicIndexing;
    public uint shaderStorageBufferArrayDynamicIndexing;
    public uint shaderStorageImageArrayDynamicIndexing;
    public uint shaderClipDistance;
    public uint shaderCullDistance;
    public uint shaderFloat64;
    public uint shaderInt64;
    public uint shaderInt16;
    public uint shaderResourceResidency;
    public uint shaderResourceMinLod;
    public uint sparseBinding;
    public uint sparseResidencyBuffer;
    public uint sparseResidencyImage2D;
    public uint sparseResidencyImage3D;
    public uint sparseResidency2Samples;
    public uint sparseResidency4Samples;
    public uint sparseResidency8Samples;
    public uint sparseResidency16Samples;
    public uint sparseResidencyAliased;
    public uint variableMultisampleRate;
    public uint inheritedQueries;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkPhysicalDeviceFeatures2
{
    public VkStructureType sType;
    public nint pNext;
    public VkPhysicalDeviceFeatures features;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkPhysicalDeviceLimits
{
    // Layout matches vulkan_core.h exactly. We only consume a subset; fields are
    // included as primitives so the struct is blittable for vkGetPhysicalDeviceProperties2.
    public uint maxImageDimension1D;
    public uint maxImageDimension2D;
    public uint maxImageDimension3D;
    public uint maxImageDimensionCube;
    public uint maxImageArrayLayers;
    public uint maxTexelBufferElements;
    public uint maxUniformBufferRange;
    public uint maxStorageBufferRange;
    public uint maxPushConstantsSize;
    public uint maxMemoryAllocationCount;
    public uint maxSamplerAllocationCount;
    public ulong bufferImageGranularity;
    public ulong sparseAddressSpaceSize;
    public uint maxBoundDescriptorSets;
    public uint maxPerStageDescriptorSamplers;
    public uint maxPerStageDescriptorUniformBuffers;
    public uint maxPerStageDescriptorStorageBuffers;
    public uint maxPerStageDescriptorSampledImages;
    public uint maxPerStageDescriptorStorageImages;
    public uint maxPerStageDescriptorInputAttachments;
    public uint maxPerStageResources;
    public uint maxDescriptorSetSamplers;
    public uint maxDescriptorSetUniformBuffers;
    public uint maxDescriptorSetUniformBuffersDynamic;
    public uint maxDescriptorSetStorageBuffers;
    public uint maxDescriptorSetStorageBuffersDynamic;
    public uint maxDescriptorSetSampledImages;
    public uint maxDescriptorSetStorageImages;
    public uint maxDescriptorSetInputAttachments;
    public uint maxVertexInputAttributes;
    public uint maxVertexInputBindings;
    public uint maxVertexInputAttributeOffset;
    public uint maxVertexInputBindingStride;
    public uint maxVertexOutputComponents;
    public uint maxTessellationGenerationLevel;
    public uint maxTessellationPatchSize;
    public uint maxTessellationControlPerVertexInputComponents;
    public uint maxTessellationControlPerVertexOutputComponents;
    public uint maxTessellationControlPerPatchOutputComponents;
    public uint maxTessellationControlTotalOutputComponents;
    public uint maxTessellationEvaluationInputComponents;
    public uint maxTessellationEvaluationOutputComponents;
    public uint maxGeometryShaderInvocations;
    public uint maxGeometryInputComponents;
    public uint maxGeometryOutputComponents;
    public uint maxGeometryOutputVertices;
    public uint maxGeometryTotalOutputComponents;
    public uint maxFragmentInputComponents;
    public uint maxFragmentOutputAttachments;
    public uint maxFragmentDualSrcAttachments;
    public uint maxFragmentCombinedOutputResources;
    public uint maxComputeSharedMemorySize;
    public uint maxComputeWorkGroupCountX;
    public uint maxComputeWorkGroupCountY;
    public uint maxComputeWorkGroupCountZ;
    public uint maxComputeWorkGroupInvocations;
    public uint maxComputeWorkGroupSizeX;
    public uint maxComputeWorkGroupSizeY;
    public uint maxComputeWorkGroupSizeZ;
    public uint subPixelPrecisionBits;
    public uint subTexelPrecisionBits;
    public uint mipmapPrecisionBits;
    public uint maxDrawIndexedIndexValue;
    public uint maxDrawIndirectCount;
    public float maxSamplerLodBias;
    public float maxSamplerAnisotropy;
    public uint maxViewports;
    public uint maxViewportDimensionsX;
    public uint maxViewportDimensionsY;
    public float viewportBoundsRangeMin;
    public float viewportBoundsRangeMax;
    public uint viewportSubPixelBits;
    public nuint minMemoryMapAlignment;
    public ulong minTexelBufferOffsetAlignment;
    public ulong minUniformBufferOffsetAlignment;
    public ulong minStorageBufferOffsetAlignment;
    public int minTexelOffset;
    public uint maxTexelOffset;
    public int minTexelGatherOffset;
    public uint maxTexelGatherOffset;
    public float minInterpolationOffset;
    public float maxInterpolationOffset;
    public uint subPixelInterpolationOffsetBits;
    public uint maxFramebufferWidth;
    public uint maxFramebufferHeight;
    public uint maxFramebufferLayers;
    public uint framebufferColorSampleCounts;
    public uint framebufferDepthSampleCounts;
    public uint framebufferStencilSampleCounts;
    public uint framebufferNoAttachmentsSampleCounts;
    public uint maxColorAttachments;
    public uint sampledImageColorSampleCounts;
    public uint sampledImageIntegerSampleCounts;
    public uint sampledImageDepthSampleCounts;
    public uint sampledImageStencilSampleCounts;
    public uint storageImageSampleCounts;
    public uint maxSampleMaskWords;
    public uint timestampComputeAndGraphics;
    public float timestampPeriod;
    public uint maxClipDistances;
    public uint maxCullDistances;
    public uint maxCombinedClipAndCullDistances;
    public uint discreteQueuePriorities;
    public float pointSizeRangeMin;
    public float pointSizeRangeMax;
    public float lineWidthRangeMin;
    public float lineWidthRangeMax;
    public float pointSizeGranularity;
    public float lineWidthGranularity;
    public uint strictLines;
    public uint standardSampleLocations;
    public ulong optimalBufferCopyOffsetAlignment;
    public ulong optimalBufferCopyRowPitchAlignment;
    public ulong nonCoherentAtomSize;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkPhysicalDeviceSparseProperties
{
    public uint residencyStandard2DBlockShape;
    public uint residencyStandard2DMultisampleBlockShape;
    public uint residencyStandard3DBlockShape;
    public uint residencyAlignedMipSize;
    public uint residencyNonResidentStrict;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkPhysicalDeviceProperties
{
    public uint apiVersion;
    public uint driverVersion;
    public uint vendorID;
    public uint deviceID;
    public VkPhysicalDeviceType deviceType;
    public fixed byte deviceName[256];
    public fixed byte pipelineCacheUUID[16];
    public VkPhysicalDeviceLimits limits;
    public VkPhysicalDeviceSparseProperties sparseProperties;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkPhysicalDeviceProperties2
{
    public VkStructureType sType;
    public nint pNext;
    public VkPhysicalDeviceProperties properties;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkPhysicalDeviceSubgroupProperties
{
    public VkStructureType sType;
    public nint pNext;
    public uint subgroupSize;
    public VkShaderStageFlags supportedStages;
    public VkSubgroupFeatureFlags supportedOperations;
    public uint quadOperationsInAllStages;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkPhysicalDeviceSubgroupSizeControlProperties
{
    public VkStructureType sType;
    public nint pNext;
    public uint minSubgroupSize;
    public uint maxSubgroupSize;
    public uint maxComputeWorkgroupSubgroups;
    public VkShaderStageFlags requiredSubgroupSizeStages;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkPhysicalDeviceVulkan11Features
{
    public VkStructureType sType;
    public nint pNext;
    public uint storageBuffer16BitAccess;
    public uint uniformAndStorageBuffer16BitAccess;
    public uint storagePushConstant16;
    public uint storageInputOutput16;
    public uint multiview;
    public uint multiviewGeometryShader;
    public uint multiviewTessellationShader;
    public uint variablePointersStorageBuffer;
    public uint variablePointers;
    public uint protectedMemory;
    public uint samplerYcbcrConversion;
    public uint shaderDrawParameters;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkPhysicalDeviceVulkan12Features
{
    public VkStructureType sType;
    public nint pNext;
    public uint samplerMirrorClampToEdge;
    public uint drawIndirectCount;
    public uint storageBuffer8BitAccess;
    public uint uniformAndStorageBuffer8BitAccess;
    public uint storagePushConstant8;
    public uint shaderBufferInt64Atomics;
    public uint shaderSharedInt64Atomics;
    public uint shaderFloat16;
    public uint shaderInt8;
    public uint descriptorIndexing;
    public uint shaderInputAttachmentArrayDynamicIndexing;
    public uint shaderUniformTexelBufferArrayDynamicIndexing;
    public uint shaderStorageTexelBufferArrayDynamicIndexing;
    public uint shaderUniformBufferArrayNonUniformIndexing;
    public uint shaderSampledImageArrayNonUniformIndexing;
    public uint shaderStorageBufferArrayNonUniformIndexing;
    public uint shaderStorageImageArrayNonUniformIndexing;
    public uint shaderInputAttachmentArrayNonUniformIndexing;
    public uint shaderUniformTexelBufferArrayNonUniformIndexing;
    public uint shaderStorageTexelBufferArrayNonUniformIndexing;
    public uint descriptorBindingUniformBufferUpdateAfterBind;
    public uint descriptorBindingSampledImageUpdateAfterBind;
    public uint descriptorBindingStorageImageUpdateAfterBind;
    public uint descriptorBindingStorageBufferUpdateAfterBind;
    public uint descriptorBindingUniformTexelBufferUpdateAfterBind;
    public uint descriptorBindingStorageTexelBufferUpdateAfterBind;
    public uint descriptorBindingUpdateUnusedWhilePending;
    public uint descriptorBindingPartiallyBound;
    public uint descriptorBindingVariableDescriptorCount;
    public uint runtimeDescriptorArray;
    public uint samplerFilterMinmax;
    public uint scalarBlockLayout;
    public uint imagelessFramebuffer;
    public uint uniformBufferStandardLayout;
    public uint shaderSubgroupExtendedTypes;
    public uint separateDepthStencilLayouts;
    public uint hostQueryReset;
    public uint timelineSemaphore;
    public uint bufferDeviceAddress;
    public uint bufferDeviceAddressCaptureReplay;
    public uint bufferDeviceAddressMultiDevice;
    public uint vulkanMemoryModel;
    public uint vulkanMemoryModelDeviceScope;
    public uint vulkanMemoryModelAvailabilityVisibilityChains;
    public uint shaderOutputViewportIndex;
    public uint shaderOutputLayer;
    public uint subgroupBroadcastDynamicId;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkPhysicalDeviceVulkan13Features
{
    public VkStructureType sType;
    public nint pNext;
    public uint robustImageAccess;
    public uint inlineUniformBlock;
    public uint descriptorBindingInlineUniformBlockUpdateAfterBind;
    public uint pipelineCreationCacheControl;
    public uint privateData;
    public uint shaderDemoteToHelperInvocation;
    public uint shaderTerminateInvocation;
    public uint subgroupSizeControl;
    public uint computeFullSubgroups;
    public uint synchronization2;
    public uint textureCompressionASTC_HDR;
    public uint shaderZeroInitializeWorkgroupMemory;
    public uint dynamicRendering;
    public uint shaderIntegerDotProduct;
    public uint maintenance4;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkMemoryType
{
    public VkMemoryPropertyFlags propertyFlags;
    public uint heapIndex;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkMemoryHeap
{
    public ulong size;
    public VkMemoryHeapFlags flags;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkPhysicalDeviceMemoryProperties
{
    public uint memoryTypeCount;
    // 32 * sizeof(VkMemoryType=8) = 256 bytes
    public VkMemoryType memoryTypes_0; public VkMemoryType memoryTypes_1; public VkMemoryType memoryTypes_2; public VkMemoryType memoryTypes_3;
    public VkMemoryType memoryTypes_4; public VkMemoryType memoryTypes_5; public VkMemoryType memoryTypes_6; public VkMemoryType memoryTypes_7;
    public VkMemoryType memoryTypes_8; public VkMemoryType memoryTypes_9; public VkMemoryType memoryTypes_10; public VkMemoryType memoryTypes_11;
    public VkMemoryType memoryTypes_12; public VkMemoryType memoryTypes_13; public VkMemoryType memoryTypes_14; public VkMemoryType memoryTypes_15;
    public VkMemoryType memoryTypes_16; public VkMemoryType memoryTypes_17; public VkMemoryType memoryTypes_18; public VkMemoryType memoryTypes_19;
    public VkMemoryType memoryTypes_20; public VkMemoryType memoryTypes_21; public VkMemoryType memoryTypes_22; public VkMemoryType memoryTypes_23;
    public VkMemoryType memoryTypes_24; public VkMemoryType memoryTypes_25; public VkMemoryType memoryTypes_26; public VkMemoryType memoryTypes_27;
    public VkMemoryType memoryTypes_28; public VkMemoryType memoryTypes_29; public VkMemoryType memoryTypes_30; public VkMemoryType memoryTypes_31;
    public uint memoryHeapCount;
    // 16 * sizeof(VkMemoryHeap=16) = 256 bytes
    public VkMemoryHeap memoryHeaps_0; public VkMemoryHeap memoryHeaps_1; public VkMemoryHeap memoryHeaps_2; public VkMemoryHeap memoryHeaps_3;
    public VkMemoryHeap memoryHeaps_4; public VkMemoryHeap memoryHeaps_5; public VkMemoryHeap memoryHeaps_6; public VkMemoryHeap memoryHeaps_7;
    public VkMemoryHeap memoryHeaps_8; public VkMemoryHeap memoryHeaps_9; public VkMemoryHeap memoryHeaps_10; public VkMemoryHeap memoryHeaps_11;
    public VkMemoryHeap memoryHeaps_12; public VkMemoryHeap memoryHeaps_13; public VkMemoryHeap memoryHeaps_14; public VkMemoryHeap memoryHeaps_15;

    /// <summary>Element-wise accessor for memoryTypes (avoids unsafe span tricks at call sites).</summary>
    public VkMemoryType GetMemoryType(int i) => i switch
    {
        0 => memoryTypes_0, 1 => memoryTypes_1, 2 => memoryTypes_2, 3 => memoryTypes_3,
        4 => memoryTypes_4, 5 => memoryTypes_5, 6 => memoryTypes_6, 7 => memoryTypes_7,
        8 => memoryTypes_8, 9 => memoryTypes_9, 10 => memoryTypes_10, 11 => memoryTypes_11,
        12 => memoryTypes_12, 13 => memoryTypes_13, 14 => memoryTypes_14, 15 => memoryTypes_15,
        16 => memoryTypes_16, 17 => memoryTypes_17, 18 => memoryTypes_18, 19 => memoryTypes_19,
        20 => memoryTypes_20, 21 => memoryTypes_21, 22 => memoryTypes_22, 23 => memoryTypes_23,
        24 => memoryTypes_24, 25 => memoryTypes_25, 26 => memoryTypes_26, 27 => memoryTypes_27,
        28 => memoryTypes_28, 29 => memoryTypes_29, 30 => memoryTypes_30, 31 => memoryTypes_31,
        _ => throw new ArgumentOutOfRangeException(nameof(i)),
    };

    public VkMemoryHeap GetMemoryHeap(int i) => i switch
    {
        0 => memoryHeaps_0, 1 => memoryHeaps_1, 2 => memoryHeaps_2, 3 => memoryHeaps_3,
        4 => memoryHeaps_4, 5 => memoryHeaps_5, 6 => memoryHeaps_6, 7 => memoryHeaps_7,
        8 => memoryHeaps_8, 9 => memoryHeaps_9, 10 => memoryHeaps_10, 11 => memoryHeaps_11,
        12 => memoryHeaps_12, 13 => memoryHeaps_13, 14 => memoryHeaps_14, 15 => memoryHeaps_15,
        _ => throw new ArgumentOutOfRangeException(nameof(i)),
    };
}

[StructLayout(LayoutKind.Sequential)]
public struct VkQueueFamilyProperties
{
    public VkQueueFlags queueFlags;
    public uint queueCount;
    public uint timestampValidBits;
    public VkExtent3D minImageTransferGranularity;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkDeviceQueueCreateInfo
{
    public VkStructureType sType;
    public nint pNext;
    public uint flags;
    public uint queueFamilyIndex;
    public uint queueCount;
    public nint pQueuePriorities;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkDeviceCreateInfo
{
    public VkStructureType sType;
    public nint pNext;
    public uint flags;
    public uint queueCreateInfoCount;
    public nint pQueueCreateInfos;
    public uint enabledLayerCount;
    public nint ppEnabledLayerNames;
    public uint enabledExtensionCount;
    public nint ppEnabledExtensionNames;
    public nint pEnabledFeatures;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkBufferCreateInfo
{
    public VkStructureType sType;
    public nint pNext;
    public uint flags;
    public ulong size;
    public VkBufferUsageFlags usage;
    public VkSharingMode sharingMode;
    public uint queueFamilyIndexCount;
    public nint pQueueFamilyIndices;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkMemoryRequirements
{
    public ulong size;
    public ulong alignment;
    public uint memoryTypeBits;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkMemoryAllocateInfo
{
    public VkStructureType sType;
    public nint pNext;
    public ulong allocationSize;
    public uint memoryTypeIndex;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkMappedMemoryRange
{
    public VkStructureType sType;
    public nint pNext;
    public ulong memory;
    public ulong offset;
    public ulong size;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkBufferCopy
{
    public ulong srcOffset;
    public ulong dstOffset;
    public ulong size;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkShaderModuleCreateInfo
{
    public VkStructureType sType;
    public nint pNext;
    public uint flags;
    public nuint codeSize;
    public nint pCode;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkPipelineCacheCreateInfo
{
    public VkStructureType sType;
    public nint pNext;
    public uint flags;
    public nuint initialDataSize;
    public nint pInitialData;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkSpecializationMapEntry
{
    public uint constantID;
    public uint offset;
    public nuint size;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkSpecializationInfo
{
    public uint mapEntryCount;
    public nint pMapEntries;
    public nuint dataSize;
    public nint pData;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkPipelineShaderStageCreateInfo
{
    public VkStructureType sType;
    public nint pNext;
    public VkPipelineShaderStageCreateFlags flags;
    public VkShaderStageFlags stage;
    public ulong module;
    public nint pName;
    public nint pSpecializationInfo;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkPipelineShaderStageRequiredSubgroupSizeCreateInfo
{
    public VkStructureType sType;
    public nint pNext;
    public uint requiredSubgroupSize;
}

/// <summary>Chained into <see cref="VkPhysicalDeviceFeatures2"/> at device-create when <c>VK_KHR_cooperative_matrix</c> is enabled.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct VkPhysicalDeviceCooperativeMatrixFeaturesKHR
{
    public VkStructureType sType;
    public nint pNext;
    public uint cooperativeMatrix;
    public uint cooperativeMatrixRobustBufferAccess;
}

/// <summary>One supported cooperative-matrix configuration, enumerated via <c>vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR</c>.</summary>
// The backend must confirm the shader's exact shape (16x16x16, F16 A/B, F32 accumulate, subgroup scope) is in
// this list before using coopmat — NVIDIA always reports it, but AMD RDNA3 / Intel Arc report different sets,
// so blindly assuming it miscomputes or fails to create the pipeline.
[StructLayout(LayoutKind.Sequential)]
public struct VkCooperativeMatrixPropertiesKHR
{
    public VkStructureType sType;
    public nint pNext;
    public uint MSize;
    public uint NSize;
    public uint KSize;
    public VkComponentTypeKHR AType;
    public VkComponentTypeKHR BType;
    public VkComponentTypeKHR CType;
    public VkComponentTypeKHR ResultType;
    public uint saturatingAccumulation; // VkBool32
    public VkScopeKHR scope;
}

/// <summary>One supported <c>VK_NV_cooperative_matrix2</c> "flexible dimensions" configuration — enumerated via <c>vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV</c>. Unlike coopmat1's fixed 16x16x16 shapes, coopmat2 reports GRANULARITIES (the M/N/K dims used must be multiples of these) plus <c>workgroupInvocations</c> (the exact workgroup size the driver expects for this configuration's <c>gl_ScopeWorkgroup</c> matrices).</summary>
[StructLayout(LayoutKind.Sequential)]
public struct VkCooperativeMatrixFlexibleDimensionsPropertiesNV
{
    public VkStructureType sType;
    public nint pNext;
    public uint MGranularity;
    public uint NGranularity;
    public uint KGranularity;
    public VkComponentTypeKHR AType;
    public VkComponentTypeKHR BType;
    public VkComponentTypeKHR CType;
    public VkComponentTypeKHR ResultType;
    public uint saturatingAccumulation; // VkBool32
    public VkScopeKHR scope;
    public uint workgroupInvocations;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkPhysicalDeviceCooperativeMatrix2FeaturesNV
{
    public VkStructureType sType;
    public nint pNext;
    public uint cooperativeMatrixWorkgroupScope;      // VkBool32 — the feature matmul_coopmat2 needs
    public uint cooperativeMatrixFlexibleDimensions;   // VkBool32 — lets M/N/K be chosen at pipeline-creation time
    public uint cooperativeMatrixReductions;
    public uint cooperativeMatrixConversions;
    public uint cooperativeMatrixPerElementOperations;
    public uint cooperativeMatrixTensorAddressing;     // VkBool32 — tensorLayoutNV / coopMatLoadTensorNV
    public uint cooperativeMatrixBlockLoads;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkPhysicalDeviceCooperativeMatrix2PropertiesNV
{
    public VkStructureType sType;
    public nint pNext;
    public uint cooperativeMatrixWorkgroupScopeMaxWorkgroupSize;
    public uint cooperativeMatrixFlexibleDimensionsMaxDimension;
    public uint cooperativeMatrixWorkgroupScopeReservedSharedMemory;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkComputePipelineCreateInfo
{
    public VkStructureType sType;
    public nint pNext;
    public uint flags;
    public VkPipelineShaderStageCreateInfo stage;
    public ulong layout;
    public ulong basePipelineHandle;
    public int basePipelineIndex;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkPushConstantRange
{
    public VkShaderStageFlags stageFlags;
    public uint offset;
    public uint size;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkPipelineLayoutCreateInfo
{
    public VkStructureType sType;
    public nint pNext;
    public uint flags;
    public uint setLayoutCount;
    public nint pSetLayouts;
    public uint pushConstantRangeCount;
    public nint pPushConstantRanges;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkDescriptorSetLayoutBinding
{
    public uint binding;
    public VkDescriptorType descriptorType;
    public uint descriptorCount;
    public VkShaderStageFlags stageFlags;
    public nint pImmutableSamplers;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkDescriptorSetLayoutCreateInfo
{
    public VkStructureType sType;
    public nint pNext;
    public uint flags;
    public uint bindingCount;
    public nint pBindings;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkDescriptorPoolSize
{
    public VkDescriptorType type;
    public uint descriptorCount;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkDescriptorPoolCreateInfo
{
    public VkStructureType sType;
    public nint pNext;
    public uint flags;
    public uint maxSets;
    public uint poolSizeCount;
    public nint pPoolSizes;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkDescriptorSetAllocateInfo
{
    public VkStructureType sType;
    public nint pNext;
    public ulong descriptorPool;
    public uint descriptorSetCount;
    public nint pSetLayouts;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkDescriptorBufferInfo
{
    public ulong buffer;
    public ulong offset;
    public ulong range;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkWriteDescriptorSet
{
    public VkStructureType sType;
    public nint pNext;
    public ulong dstSet;
    public uint dstBinding;
    public uint dstArrayElement;
    public uint descriptorCount;
    public VkDescriptorType descriptorType;
    public nint pImageInfo;
    public nint pBufferInfo;
    public nint pTexelBufferView;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkCommandPoolCreateInfo
{
    public VkStructureType sType;
    public nint pNext;
    public VkCommandPoolCreateFlags flags;
    public uint queueFamilyIndex;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkCommandBufferAllocateInfo
{
    public VkStructureType sType;
    public nint pNext;
    public ulong commandPool;
    public VkCommandBufferLevel level;
    public uint commandBufferCount;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkCommandBufferBeginInfo
{
    public VkStructureType sType;
    public nint pNext;
    public VkCommandBufferUsageFlags flags;
    public nint pInheritanceInfo;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkSemaphoreCreateInfo
{
    public VkStructureType sType;
    public nint pNext;
    public uint flags;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkSemaphoreTypeCreateInfo
{
    public VkStructureType sType;
    public nint pNext;
    public VkSemaphoreType semaphoreType;
    public ulong initialValue;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkSemaphoreSignalInfo
{
    public VkStructureType sType;
    public nint pNext;
    public ulong semaphore;
    public ulong value;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkSemaphoreWaitInfo
{
    public VkStructureType sType;
    public nint pNext;
    public uint flags;
    public uint semaphoreCount;
    public nint pSemaphores;
    public nint pValues;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkFenceCreateInfo
{
    public VkStructureType sType;
    public nint pNext;
    public uint flags;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkQueryPoolCreateInfo
{
    public VkStructureType sType;
    public nint pNext;
    public uint flags;
    public uint queryType;   // VkQueryType — 2 = VK_QUERY_TYPE_TIMESTAMP (the only value this backend uses)
    public uint queryCount;
    public uint pipelineStatistics;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkBufferMemoryBarrier2
{
    public VkStructureType sType;
    public nint pNext;
    public ulong srcStageMask;
    public ulong srcAccessMask;
    public ulong dstStageMask;
    public ulong dstAccessMask;
    public uint srcQueueFamilyIndex;
    public uint dstQueueFamilyIndex;
    public ulong buffer;
    public ulong offset;
    public ulong size;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkMemoryBarrier2
{
    public VkStructureType sType;
    public nint pNext;
    public ulong srcStageMask;
    public ulong srcAccessMask;
    public ulong dstStageMask;
    public ulong dstAccessMask;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkDependencyInfo
{
    public VkStructureType sType;
    public nint pNext;
    public uint dependencyFlags;
    public uint memoryBarrierCount;
    public nint pMemoryBarriers;
    public uint bufferMemoryBarrierCount;
    public nint pBufferMemoryBarriers;
    public uint imageMemoryBarrierCount;
    public nint pImageMemoryBarriers;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkSemaphoreSubmitInfo
{
    public VkStructureType sType;
    public nint pNext;
    public ulong semaphore;
    public ulong value;
    public ulong stageMask;
    public uint deviceIndex;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkCommandBufferSubmitInfo
{
    public VkStructureType sType;
    public nint pNext;
    public nint commandBuffer;
    public uint deviceMask;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkSubmitInfo2
{
    public VkStructureType sType;
    public nint pNext;
    public uint flags;
    public uint waitSemaphoreInfoCount;
    public nint pWaitSemaphoreInfos;
    public uint commandBufferInfoCount;
    public nint pCommandBufferInfos;
    public uint signalSemaphoreInfoCount;
    public nint pSignalSemaphoreInfos;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkExtensionProperties
{
    // 256-byte name + 4-byte uint = 260 bytes
    public unsafe fixed byte extensionName[256];
    public uint specVersion;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkLayerProperties
{
    public unsafe fixed byte layerName[256];
    public uint specVersion;
    public uint implementationVersion;
    public unsafe fixed byte description[256];
}
