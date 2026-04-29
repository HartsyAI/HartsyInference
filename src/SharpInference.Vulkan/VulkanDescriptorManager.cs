using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SharpInference.Vulkan;

/// <summary>
/// Manages a small set of canonical descriptor-set layouts, pipeline layouts,
/// and a descriptor-pool ring. Used by every kernel — by sharing layouts across
/// kernels with the same binding shape we keep total layout count to ~12.
///
/// Pool ring strategy: two pools, alternate per phase boundary. <c>FlipPool()</c>
/// resets the next pool. The active pool can hold up to <see cref="MaxSetsPerPool"/>
/// allocations before forcing a flip.
/// </summary>
public sealed class VulkanDescriptorManager : IDisposable
{
    private const int MaxSetsPerPool = 4096;
    private const int StorageBuffersPerSet = 8;   // upper bound on bindings per set
    private const int UniformBuffersPerSet = 2;

    private readonly nint _device;
    private readonly bool _pushDescriptor;
    private readonly Dictionary<int, ulong> _setLayouts = new();      // bindingCount -> VkDescriptorSetLayout
    private readonly Dictionary<int, ulong> _pipelineLayouts = new(); // bindingCount -> VkPipelineLayout (push-const = 128B compute stage)
    private readonly ulong[] _pools = new ulong[2];
    private int _activePool;
    private uint _setsInActive;

    /// <summary>True when VK_KHR_push_descriptor / Vulkan 1.4 core push descriptors are active.
    /// In this mode <see cref="PushSet"/> writes descriptors directly into the command buffer,
    /// bypassing the pool ring entirely. Saves a vkAllocateDescriptorSets + vkUpdateDescriptorSets
    /// round-trip per dispatch.</summary>
    public bool PushDescriptorActive => _pushDescriptor;

    public VulkanDescriptorManager(nint device, bool enablePushDescriptor = false)
    {
        _device = device;
        _pushDescriptor = enablePushDescriptor;
        if (!_pushDescriptor)
        {
            _pools[0] = CreatePool();
            _pools[1] = CreatePool();
        }
    }

    /// <summary>Returns (or creates) a layout for <c>n</c> storage buffers at bindings 0..n-1, all visible to the compute stage.</summary>
    public ulong GetSetLayout(int storageBufferCount)
    {
        if (_setLayouts.TryGetValue(storageBufferCount, out ulong cached)) return cached;
        cached = CreateSetLayout(storageBufferCount);
        _setLayouts[storageBufferCount] = cached;
        return cached;
    }

    /// <summary>Returns (or creates) a pipeline layout = (set layout for n SSBOs) + 128-byte compute push-constant range.</summary>
    public ulong GetPipelineLayout(int storageBufferCount)
    {
        if (_pipelineLayouts.TryGetValue(storageBufferCount, out ulong cached)) return cached;

        ulong setLayout = GetSetLayout(storageBufferCount);
        unsafe
        {
            VkPushConstantRange pcRange = new()
            {
                stageFlags = VkShaderStageFlags.Compute,
                offset = 0,
                size = 128,
            };
            VkPipelineLayoutCreateInfo ci = new()
            {
                sType = VkStructureType.PipelineLayoutCreateInfo,
                setLayoutCount = 1,
                pSetLayouts = (nint)(&setLayout),
                pushConstantRangeCount = 1,
                pPushConstantRanges = (nint)(&pcRange),
            };
            VulkanApi.vkCreatePipelineLayout(_device, in ci, 0, out cached).ThrowOnError("vkCreatePipelineLayout");
        }
        _pipelineLayouts[storageBufferCount] = cached;
        return cached;
    }

    private unsafe ulong CreateSetLayout(int n)
    {
        Span<VkDescriptorSetLayoutBinding> bindings = stackalloc VkDescriptorSetLayoutBinding[Math.Max(n, 1)];
        for (int i = 0; i < n; i++)
        {
            bindings[i] = new VkDescriptorSetLayoutBinding
            {
                binding = (uint)i,
                descriptorType = VkDescriptorType.StorageBuffer,
                descriptorCount = 1,
                stageFlags = VkShaderStageFlags.Compute,
            };
        }

        // VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR = 0x1.
        // Required when the descriptor set will be populated via vkCmdPushDescriptorSet rather
        // than allocated from a pool. Sets created with this flag cannot be allocated.
        const uint pushDescriptorFlag = 0x1u;

        fixed (VkDescriptorSetLayoutBinding* p = bindings)
        {
            VkDescriptorSetLayoutCreateInfo ci = new()
            {
                sType = VkStructureType.DescriptorSetLayoutCreateInfo,
                flags = _pushDescriptor ? pushDescriptorFlag : 0u,
                bindingCount = (uint)n,
                pBindings = (nint)p,
            };
            VulkanApi.vkCreateDescriptorSetLayout(_device, in ci, 0, out ulong layout).ThrowOnError("vkCreateDescriptorSetLayout");
            return layout;
        }
    }

    private unsafe ulong CreatePool()
    {
        Span<VkDescriptorPoolSize> sizes = stackalloc VkDescriptorPoolSize[2];
        sizes[0] = new VkDescriptorPoolSize { type = VkDescriptorType.StorageBuffer, descriptorCount = (uint)(MaxSetsPerPool * StorageBuffersPerSet) };
        sizes[1] = new VkDescriptorPoolSize { type = VkDescriptorType.UniformBuffer, descriptorCount = (uint)(MaxSetsPerPool * UniformBuffersPerSet) };
        fixed (VkDescriptorPoolSize* ps = sizes)
        {
            VkDescriptorPoolCreateInfo ci = new()
            {
                sType = VkStructureType.DescriptorPoolCreateInfo,
                maxSets = MaxSetsPerPool,
                poolSizeCount = 2,
                pPoolSizes = (nint)ps,
            };
            VulkanApi.vkCreateDescriptorPool(_device, in ci, 0, out ulong pool).ThrowOnError("vkCreateDescriptorPool");
            return pool;
        }
    }

    /// <summary>Allocates a descriptor set with the given layout from the active pool. Flips pools if exhausted.</summary>
    public unsafe ulong AllocateSet(ulong setLayout)
    {
        if (_setsInActive >= MaxSetsPerPool) FlipPool();

        VkDescriptorSetAllocateInfo ai = new()
        {
            sType = VkStructureType.DescriptorSetAllocateInfo,
            descriptorPool = _pools[_activePool],
            descriptorSetCount = 1,
            pSetLayouts = (nint)(&setLayout),
        };
        ulong outSet;
        VkResult r = VulkanApi.vkAllocateDescriptorSets(_device, in ai, (nint)(&outSet));
        if (r == (VkResult)(-1000257000) /* OUT_OF_POOL_MEMORY */ || r == VkResult.ErrorOutOfPoolMemory)
        {
            FlipPool();
            ai.descriptorPool = _pools[_activePool];
            VulkanApi.vkAllocateDescriptorSets(_device, in ai, (nint)(&outSet)).ThrowOnError("vkAllocateDescriptorSets (after flip)");
        }
        else
        {
            r.ThrowOnError("vkAllocateDescriptorSets");
        }
        _setsInActive++;
        return outSet;
    }

    /// <summary>Switches to the other pool, resetting it so its full capacity is available again. Call at phase boundaries.</summary>
    public void FlipPool()
    {
        _activePool = 1 - _activePool;
        VulkanApi.vkResetDescriptorPool(_device, _pools[_activePool], 0).ThrowOnError("vkResetDescriptorPool");
        _setsInActive = 0;
    }

    /// <summary>Records descriptor writes directly into the command buffer via
    /// <c>vkCmdPushDescriptorSet</c>. No descriptor set allocation, no pool, no
    /// <c>vkUpdateDescriptorSets</c> round-trip. Active when <see cref="PushDescriptorActive"/>
    /// is true (set at construction). Safe to call from any recorded dispatch — the descriptors
    /// are valid for the next dispatch only, which matches the per-op binding pattern we use.</summary>
    public unsafe void PushSet(nint commandBuffer, ulong pipelineLayout, ReadOnlySpan<ulong> bufferHandles)
    {
        int n = bufferHandles.Length;
        Span<VkDescriptorBufferInfo> infos = stackalloc VkDescriptorBufferInfo[n];
        Span<VkWriteDescriptorSet> writes = stackalloc VkWriteDescriptorSet[n];
        for (int i = 0; i < n; i++)
        {
            infos[i] = new VkDescriptorBufferInfo
            {
                buffer = bufferHandles[i],
                offset = 0,
                range = VkConstants.WholeSize,
            };
        }
        fixed (VkDescriptorBufferInfo* pInfos = infos)
        {
            for (int i = 0; i < n; i++)
            {
                writes[i] = new VkWriteDescriptorSet
                {
                    sType = VkStructureType.WriteDescriptorSet,
                    dstSet = 0,                                       // ignored for push descriptors
                    dstBinding = (uint)i,
                    descriptorCount = 1,
                    descriptorType = VkDescriptorType.StorageBuffer,
                    pBufferInfo = (nint)(pInfos + i),
                };
            }
            fixed (VkWriteDescriptorSet* pWrites = writes)
            {
                VulkanApi.vkCmdPushDescriptorSet(commandBuffer, VkPipelineBindPoint.Compute,
                    pipelineLayout, set: 0, descriptorWriteCount: (uint)n, pDescriptorWrites: (nint)pWrites);
            }
        }
    }

    /// <summary>Writes the bindings for a fresh set: one storage-buffer descriptor per <paramref name="bufferHandles"/>. Buffer offset = 0, range = WHOLE_SIZE.</summary>
    public unsafe void WriteSet(ulong dstSet, ReadOnlySpan<ulong> bufferHandles)
    {
        int n = bufferHandles.Length;
        Span<VkDescriptorBufferInfo> infos = stackalloc VkDescriptorBufferInfo[n];
        Span<VkWriteDescriptorSet> writes = stackalloc VkWriteDescriptorSet[n];
        for (int i = 0; i < n; i++)
        {
            infos[i] = new VkDescriptorBufferInfo
            {
                buffer = bufferHandles[i],
                offset = 0,
                range = VkConstants.WholeSize,
            };
        }
        fixed (VkDescriptorBufferInfo* pInfos = infos)
        {
            for (int i = 0; i < n; i++)
            {
                writes[i] = new VkWriteDescriptorSet
                {
                    sType = VkStructureType.WriteDescriptorSet,
                    dstSet = dstSet,
                    dstBinding = (uint)i,
                    descriptorCount = 1,
                    descriptorType = VkDescriptorType.StorageBuffer,
                    pBufferInfo = (nint)(pInfos + i),
                };
            }
            fixed (VkWriteDescriptorSet* pWrites = writes)
            {
                VulkanApi.vkUpdateDescriptorSets(_device, (uint)n, (nint)pWrites, 0, 0);
            }
        }
    }

    public void Dispose()
    {
        foreach ((_, ulong layout) in _pipelineLayouts) VulkanApi.vkDestroyPipelineLayout(_device, layout, 0);
        _pipelineLayouts.Clear();
        foreach ((_, ulong layout) in _setLayouts) VulkanApi.vkDestroyDescriptorSetLayout(_device, layout, 0);
        _setLayouts.Clear();
        if (!_pushDescriptor)
        {
            for (int i = 0; i < 2; i++)
                if (_pools[i] != 0) { VulkanApi.vkDestroyDescriptorPool(_device, _pools[i], 0); _pools[i] = 0; }
        }
        GC.SuppressFinalize(this);
    }
}
