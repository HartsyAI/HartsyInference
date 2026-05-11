using System.Runtime.InteropServices;

namespace SharpInference.Vulkan;

/// <summary>P/Invoke surface for Vulkan 1.3 compute. Library name "vulkan-1" is resolved at runtime by VulkanLibraryResolver to libvulkan.so.1 (Linux), vulkan-1.dll (Windows), libvulkan.1.dylib / libMoltenVK.dylib (macOS). All structs are blittable and passed by ref/out — no marshalling.</summary>
internal static partial class VulkanApi
{
    private const string Lib = "vulkan-1";

    // ── Instance / Loader ───────────────────────────────────────────────

    [LibraryImport(Lib)]
    internal static partial VkResult vkCreateInstance(in VkInstanceCreateInfo pCreateInfo, nint pAllocator, out nint pInstance);

    [LibraryImport(Lib)]
    internal static partial void vkDestroyInstance(nint instance, nint pAllocator);

    [LibraryImport(Lib)]
    internal static partial nint vkGetInstanceProcAddr(nint instance, [MarshalAs(UnmanagedType.LPStr)] string pName);

    [LibraryImport(Lib)]
    internal static partial nint vkGetDeviceProcAddr(nint device, [MarshalAs(UnmanagedType.LPStr)] string pName);

    [LibraryImport(Lib)]
    internal static partial VkResult vkEnumerateInstanceLayerProperties(ref uint pCount, nint pProperties);

    [LibraryImport(Lib)]
    internal static partial VkResult vkEnumerateInstanceExtensionProperties(nint pLayerName, ref uint pCount, nint pProperties);

    [LibraryImport(Lib)]
    internal static partial VkResult vkEnumerateDeviceExtensionProperties(nint physicalDevice, nint pLayerName, ref uint pCount, nint pProperties);

    // ── Physical Device Enumeration & Queries ───────────────────────────

    [LibraryImport(Lib)]
    internal static partial VkResult vkEnumeratePhysicalDevices(nint instance, ref uint pCount, nint pPhysicalDevices);

    [LibraryImport(Lib)]
    internal static partial void vkGetPhysicalDeviceProperties(nint physicalDevice, out VkPhysicalDeviceProperties props);

    [LibraryImport(Lib)]
    internal static partial void vkGetPhysicalDeviceProperties2(nint physicalDevice, ref VkPhysicalDeviceProperties2 props);

    [LibraryImport(Lib)]
    internal static partial void vkGetPhysicalDeviceFeatures(nint physicalDevice, out VkPhysicalDeviceFeatures features);

    [LibraryImport(Lib)]
    internal static partial void vkGetPhysicalDeviceFeatures2(nint physicalDevice, ref VkPhysicalDeviceFeatures2 features);

    [LibraryImport(Lib)]
    internal static partial void vkGetPhysicalDeviceMemoryProperties(nint physicalDevice, out VkPhysicalDeviceMemoryProperties memProps);

    [LibraryImport(Lib)]
    internal static partial void vkGetPhysicalDeviceQueueFamilyProperties(nint physicalDevice, ref uint pCount, nint pProperties);

    // ── Logical Device & Queue ──────────────────────────────────────────

    [LibraryImport(Lib)]
    internal static partial VkResult vkCreateDevice(nint physicalDevice, in VkDeviceCreateInfo pCreateInfo, nint pAllocator, out nint pDevice);

    [LibraryImport(Lib)]
    internal static partial void vkDestroyDevice(nint device, nint pAllocator);

    [LibraryImport(Lib)]
    internal static partial VkResult vkDeviceWaitIdle(nint device);

    [LibraryImport(Lib)]
    internal static partial void vkGetDeviceQueue(nint device, uint queueFamilyIndex, uint queueIndex, out nint pQueue);

    [LibraryImport(Lib)]
    internal static partial VkResult vkQueueWaitIdle(nint queue);

    // ── Buffers & Memory ────────────────────────────────────────────────

    [LibraryImport(Lib)]
    internal static partial VkResult vkCreateBuffer(nint device, in VkBufferCreateInfo pCreateInfo, nint pAllocator, out ulong pBuffer);

    [LibraryImport(Lib)]
    internal static partial void vkDestroyBuffer(nint device, ulong buffer, nint pAllocator);

    [LibraryImport(Lib)]
    internal static partial void vkGetBufferMemoryRequirements(nint device, ulong buffer, out VkMemoryRequirements pMemoryRequirements);

    [LibraryImport(Lib)]
    internal static partial VkResult vkAllocateMemory(nint device, in VkMemoryAllocateInfo pAllocateInfo, nint pAllocator, out ulong pMemory);

    [LibraryImport(Lib)]
    internal static partial void vkFreeMemory(nint device, ulong memory, nint pAllocator);

    [LibraryImport(Lib)]
    internal static partial VkResult vkBindBufferMemory(nint device, ulong buffer, ulong memory, ulong memoryOffset);

    [LibraryImport(Lib)]
    internal static partial VkResult vkMapMemory(nint device, ulong memory, ulong offset, ulong size, uint flags, out nint ppData);

    [LibraryImport(Lib)]
    internal static partial void vkUnmapMemory(nint device, ulong memory);

    [LibraryImport(Lib)]
    internal static partial VkResult vkFlushMappedMemoryRanges(nint device, uint rangeCount, nint pRanges);

    [LibraryImport(Lib)]
    internal static partial VkResult vkInvalidateMappedMemoryRanges(nint device, uint rangeCount, nint pRanges);

    // ── Shader Modules & Pipelines ──────────────────────────────────────

    [LibraryImport(Lib)]
    internal static partial VkResult vkCreateShaderModule(nint device, in VkShaderModuleCreateInfo pCreateInfo, nint pAllocator, out ulong pShaderModule);

    [LibraryImport(Lib)]
    internal static partial void vkDestroyShaderModule(nint device, ulong shaderModule, nint pAllocator);

    [LibraryImport(Lib)]
    internal static partial VkResult vkCreatePipelineLayout(nint device, in VkPipelineLayoutCreateInfo pCreateInfo, nint pAllocator, out ulong pPipelineLayout);

    [LibraryImport(Lib)]
    internal static partial void vkDestroyPipelineLayout(nint device, ulong pipelineLayout, nint pAllocator);

    [LibraryImport(Lib)]
    internal static partial VkResult vkCreatePipelineCache(nint device, in VkPipelineCacheCreateInfo pCreateInfo, nint pAllocator, out ulong pPipelineCache);

    [LibraryImport(Lib)]
    internal static partial void vkDestroyPipelineCache(nint device, ulong pipelineCache, nint pAllocator);

    [LibraryImport(Lib)]
    internal static partial VkResult vkGetPipelineCacheData(nint device, ulong pipelineCache, ref nuint pDataSize, nint pData);

    [LibraryImport(Lib)]
    internal static partial VkResult vkCreateComputePipelines(nint device, ulong pipelineCache, uint createInfoCount, nint pCreateInfos, nint pAllocator, nint pPipelines);

    [LibraryImport(Lib)]
    internal static partial void vkDestroyPipeline(nint device, ulong pipeline, nint pAllocator);

    // ── Descriptors ─────────────────────────────────────────────────────

    [LibraryImport(Lib)]
    internal static partial VkResult vkCreateDescriptorSetLayout(nint device, in VkDescriptorSetLayoutCreateInfo pCreateInfo, nint pAllocator, out ulong pSetLayout);

    [LibraryImport(Lib)]
    internal static partial void vkDestroyDescriptorSetLayout(nint device, ulong descriptorSetLayout, nint pAllocator);

    [LibraryImport(Lib)]
    internal static partial VkResult vkCreateDescriptorPool(nint device, in VkDescriptorPoolCreateInfo pCreateInfo, nint pAllocator, out ulong pDescriptorPool);

    [LibraryImport(Lib)]
    internal static partial void vkDestroyDescriptorPool(nint device, ulong descriptorPool, nint pAllocator);

    [LibraryImport(Lib)]
    internal static partial VkResult vkResetDescriptorPool(nint device, ulong descriptorPool, uint flags);

    [LibraryImport(Lib)]
    internal static partial VkResult vkAllocateDescriptorSets(nint device, in VkDescriptorSetAllocateInfo pAllocateInfo, nint pDescriptorSets);

    [LibraryImport(Lib)]
    internal static partial void vkUpdateDescriptorSets(nint device, uint descriptorWriteCount, nint pDescriptorWrites, uint descriptorCopyCount, nint pDescriptorCopies);

    // ── Command Pools / Buffers ─────────────────────────────────────────

    [LibraryImport(Lib)]
    internal static partial VkResult vkCreateCommandPool(nint device, in VkCommandPoolCreateInfo pCreateInfo, nint pAllocator, out ulong pCommandPool);

    [LibraryImport(Lib)]
    internal static partial void vkDestroyCommandPool(nint device, ulong commandPool, nint pAllocator);

    [LibraryImport(Lib)]
    internal static partial VkResult vkResetCommandPool(nint device, ulong commandPool, uint flags);

    [LibraryImport(Lib)]
    internal static partial VkResult vkAllocateCommandBuffers(nint device, in VkCommandBufferAllocateInfo pAllocateInfo, nint pCommandBuffers);

    [LibraryImport(Lib)]
    internal static partial void vkFreeCommandBuffers(nint device, ulong commandPool, uint count, nint pCommandBuffers);

    [LibraryImport(Lib)]
    internal static partial VkResult vkBeginCommandBuffer(nint commandBuffer, in VkCommandBufferBeginInfo pBeginInfo);

    [LibraryImport(Lib)]
    internal static partial VkResult vkEndCommandBuffer(nint commandBuffer);

    [LibraryImport(Lib)]
    internal static partial VkResult vkResetCommandBuffer(nint commandBuffer, uint flags);

    // ── Recording ───────────────────────────────────────────────────────

    [LibraryImport(Lib)]
    internal static partial void vkCmdBindPipeline(nint commandBuffer, VkPipelineBindPoint pipelineBindPoint, ulong pipeline);

    [LibraryImport(Lib)]
    internal static partial void vkCmdBindDescriptorSets(nint commandBuffer, VkPipelineBindPoint pipelineBindPoint, ulong layout, uint firstSet, uint descriptorSetCount, nint pDescriptorSets, uint dynamicOffsetCount, nint pDynamicOffsets);

    /// <summary>Records descriptor-set bindings directly into the command buffer without a descriptor pool. Requires Vulkan 1.4 core or the <c>VK_KHR_push_descriptor</c> extension (loader exposes both via the same unsuffixed entry point on modern builds). Saves the vkAllocateDescriptorSets + vkUpdateDescriptorSets round-trip per dispatch.</summary>
    [LibraryImport(Lib)]
    internal static partial void vkCmdPushDescriptorSet(nint commandBuffer, VkPipelineBindPoint pipelineBindPoint, ulong layout, uint set, uint descriptorWriteCount, nint pDescriptorWrites);

    [LibraryImport(Lib)]
    internal static partial void vkCmdPushConstants(nint commandBuffer, ulong layout, VkShaderStageFlags stageFlags, uint offset, uint size, nint pValues);

    [LibraryImport(Lib)]
    internal static partial void vkCmdDispatch(nint commandBuffer, uint groupCountX, uint groupCountY, uint groupCountZ);

    [LibraryImport(Lib)]
    internal static partial void vkCmdCopyBuffer(nint commandBuffer, ulong srcBuffer, ulong dstBuffer, uint regionCount, nint pRegions);

    [LibraryImport(Lib)]
    internal static partial void vkCmdFillBuffer(nint commandBuffer, ulong dstBuffer, ulong dstOffset, ulong size, uint data);

    [LibraryImport(Lib)]
    internal static partial void vkCmdPipelineBarrier2(nint commandBuffer, in VkDependencyInfo pDependencyInfo);

    // ── Sync ────────────────────────────────────────────────────────────

    [LibraryImport(Lib)]
    internal static partial VkResult vkCreateFence(nint device, in VkFenceCreateInfo pCreateInfo, nint pAllocator, out ulong pFence);

    [LibraryImport(Lib)]
    internal static partial void vkDestroyFence(nint device, ulong fence, nint pAllocator);

    [LibraryImport(Lib)]
    internal static partial VkResult vkResetFences(nint device, uint fenceCount, nint pFences);

    [LibraryImport(Lib)]
    internal static partial VkResult vkWaitForFences(nint device, uint fenceCount, nint pFences, uint waitAll, ulong timeout);

    [LibraryImport(Lib)]
    internal static partial VkResult vkGetFenceStatus(nint device, ulong fence);

    [LibraryImport(Lib)]
    internal static partial VkResult vkCreateSemaphore(nint device, in VkSemaphoreCreateInfo pCreateInfo, nint pAllocator, out ulong pSemaphore);

    [LibraryImport(Lib)]
    internal static partial void vkDestroySemaphore(nint device, ulong semaphore, nint pAllocator);

    [LibraryImport(Lib)]
    internal static partial VkResult vkSignalSemaphore(nint device, in VkSemaphoreSignalInfo pSignalInfo);

    [LibraryImport(Lib)]
    internal static partial VkResult vkWaitSemaphores(nint device, in VkSemaphoreWaitInfo pWaitInfo, ulong timeout);

    [LibraryImport(Lib)]
    internal static partial VkResult vkGetSemaphoreCounterValue(nint device, ulong semaphore, out ulong pValue);

    [LibraryImport(Lib)]
    internal static partial VkResult vkQueueSubmit2(nint queue, uint submitCount, nint pSubmits, ulong fence);
}
