using System.Runtime.InteropServices;

namespace SharpInference.Vulkan;

/// <summary>Low-level VkDeviceMemory wrapper analogous to CudaMemory. Selects the right memory type from the physical-device properties and binds it to a buffer. Higher-level allocator (<see cref="VulkanMemoryAllocator"/>) sub-allocates from these.</summary>
public static class VulkanMemoryHelpers
{
    /// <summary>Picks a memory type index that satisfies <c>memoryTypeBitsMask</c> and contains all <c>required</c> property flags. Among candidates, prefers ones with the most <c>preferred</c> flags set and the fewest extras.</summary>
    public static uint FindMemoryType(
        in VkPhysicalDeviceMemoryProperties memProps,
        uint memoryTypeBitsMask,
        VkMemoryPropertyFlags required,
        VkMemoryPropertyFlags preferred = 0)
    {
        uint best = uint.MaxValue;
        int bestScore = int.MinValue;
        for (uint i = 0; i < memProps.memoryTypeCount; i++)
        {
            if ((memoryTypeBitsMask & (1u << (int)i)) == 0) continue;
            VkMemoryType t = memProps.GetMemoryType((int)i);
            if ((t.propertyFlags & required) != required) continue;

            int score = BitsSet((uint)(t.propertyFlags & preferred)) * 4;
            // Penalize extras (e.g. don't pick HOST_CACHED when only HOST_VISIBLE was wanted)
            VkMemoryPropertyFlags extras = t.propertyFlags & ~required & ~preferred;
            score -= BitsSet((uint)extras);

            if ((required & VkMemoryPropertyFlags.DeviceLocal) != 0)
            {
                ulong heapSize = memProps.GetMemoryHeap((int)t.heapIndex).size;
                score += (int)Math.Min(heapSize >> 30, int.MaxValue);
            }

            if (score > bestScore) { bestScore = score; best = i; }
        }
        if (best == uint.MaxValue)
            throw new VulkanException(-8, $"No memory type satisfies typeBits=0x{memoryTypeBitsMask:X8} required={required}");
        return best;
    }

    private static int BitsSet(uint x)
    {
        int n = 0;
        while (x != 0) { n += (int)(x & 1u); x >>= 1; }
        return n;
    }
}

/// <summary>One sub-allocated region inside a <see cref="VulkanMemoryBlock"/>. Returned by <see cref="VulkanMemoryAllocator.Allocate"/>; freed via <see cref="VulkanMemoryAllocator.Free"/>.</summary>
public readonly record struct VulkanAllocation(
    ulong DeviceMemory,
    ulong Offset,
    ulong Size,
    nint MappedPointer,
    uint MemoryTypeIndex,
    int BlockId)
{
    /// <summary>Empty allocation sentinel; used for unsuccessful or never-bound allocations.</summary>
    public bool IsEmpty => DeviceMemory == 0 && Size == 0;
}

/// <summary>One backing <see cref="VkDeviceMemory"/> with a sorted free-list. Internal to <see cref="VulkanMemoryAllocator"/>.</summary>
internal sealed class VulkanMemoryBlock : IDisposable
{
    public ulong DeviceMemory;
    public ulong Size;
    public uint MemoryTypeIndex;
    public uint HeapIndex;
    public VkMemoryPropertyFlags PropertyFlags;
    public nint MappedPointer;
    public int BlockId;
    public int LiveAllocations;
    public List<(ulong Offset, ulong Size)> FreeList = new();
    public bool IsDedicated;     // single-allocation-per-block (for tensors larger than slab size)

    private nint _device;

    public VulkanMemoryBlock(nint device, ulong deviceMemory, ulong size, uint memoryTypeIndex,
                             uint heapIndex, VkMemoryPropertyFlags flags, nint mapped, int blockId, bool dedicated)
    {
        _device = device;
        DeviceMemory = deviceMemory;
        Size = size;
        MemoryTypeIndex = memoryTypeIndex;
        HeapIndex = heapIndex;
        PropertyFlags = flags;
        MappedPointer = mapped;
        BlockId = blockId;
        IsDedicated = dedicated;
        FreeList.Add((0, size));
    }

    /// <summary>First-fit allocation. Returns offset on success; ulong.MaxValue when no fit.</summary>
    public ulong TryAllocate(ulong size, ulong alignment)
    {
        for (int i = 0; i < FreeList.Count; i++)
        {
            (ulong off, ulong sz) = FreeList[i];
            ulong alignedOff = (off + alignment - 1) & ~(alignment - 1);
            ulong padding = alignedOff - off;
            if (sz >= padding + size)
            {
                ulong remainingTail = sz - padding - size;
                FreeList.RemoveAt(i);
                if (padding > 0) FreeList.Insert(i++, (off, padding));
                if (remainingTail > 0) FreeList.Insert(i, (alignedOff + size, remainingTail));
                LiveAllocations++;
                return alignedOff;
            }
        }
        return ulong.MaxValue;
    }

    public void Free(ulong offset, ulong size)
    {
        // Insert sorted, then merge adjacent
        int insertIdx = FreeList.BinarySearch((offset, size), Comparer<(ulong, ulong)>.Create(
            (a, b) => a.Item1.CompareTo(b.Item1)));
        if (insertIdx < 0) insertIdx = ~insertIdx;
        FreeList.Insert(insertIdx, (offset, size));
        Coalesce();
        LiveAllocations--;
    }

    private void Coalesce()
    {
        for (int i = 0; i < FreeList.Count - 1;)
        {
            (ulong o1, ulong s1) = FreeList[i];
            (ulong o2, ulong s2) = FreeList[i + 1];
            if (o1 + s1 == o2)
            {
                FreeList[i] = (o1, s1 + s2);
                FreeList.RemoveAt(i + 1);
            }
            else i++;
        }
    }

    public void Dispose()
    {
        if (DeviceMemory != 0)
        {
            if (MappedPointer != 0) VulkanApi.vkUnmapMemory(_device, DeviceMemory);
            VulkanApi.vkFreeMemory(_device, DeviceMemory, 0);
            DeviceMemory = 0;
        }
    }
}

/// <summary>Two-tier slab allocator. Sub-allocates <see cref="VkBuffer"/> regions from large pre-allocated <see cref="VkDeviceMemory"/> blocks (256 MB for big tensors/weights, 16 MB for small temporaries); allocations larger than the slab fall back to dedicated blocks. Not thread-safe — the engine uses one allocator per <see cref="VulkanBackend"/>. See <c>docs/Research/VULKAN_MEMORY_MANAGEMENT.md</c>.</summary>
public sealed class VulkanMemoryAllocator : IDisposable
{
    public const ulong SLAB_LARGE = 64UL * 1024 * 1024;
    public const ulong SLAB_SMALL = 8UL * 1024 * 1024;
    public const ulong DEDICATED_THRESHOLD = 16UL * 1024 * 1024;

    private readonly nint _device;
    private readonly VkPhysicalDeviceMemoryProperties _memProps;
    private readonly List<VulkanMemoryBlock> _blocks = new();
    private int _nextBlockId;
    private bool _disposed;

    /// <summary>Optional callback fired on OOM allocation — used by VulkanBackend to flush the
    /// command stream and drain the deferred-free list before retrying. Set after the stream is
    /// constructed (chicken-and-egg: allocator created before stream).</summary>
    public Action? OnOutOfMemory { get; set; }

    public VulkanMemoryAllocator(nint device, in VkPhysicalDeviceMemoryProperties memProps)
    {
        _device = device;
        _memProps = memProps;
    }

    public VulkanAllocation Allocate(
        ulong size, ulong alignment,
        uint memoryTypeBitsMask,
        VkMemoryPropertyFlags required,
        VkMemoryPropertyFlags preferred = 0)
    {
        if (size == 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Allocation size must be > 0");
        if (alignment == 0) alignment = 16;

        uint typeIdx = VulkanMemoryHelpers.FindMemoryType(in _memProps, memoryTypeBitsMask, required, preferred);

        if (size >= DEDICATED_THRESHOLD) return AllocateDedicated(size, typeIdx, required);

        // Try existing blocks of same type
        for (int i = 0; i < _blocks.Count; i++)
        {
            VulkanMemoryBlock b = _blocks[i];
            if (b.MemoryTypeIndex != typeIdx || b.IsDedicated) continue;
            ulong off = b.TryAllocate(size, alignment);
            if (off != ulong.MaxValue)
            {
                return new VulkanAllocation(b.DeviceMemory, off, size,
                    b.MappedPointer == 0 ? 0 : b.MappedPointer + (nint)off,
                    typeIdx, b.BlockId);
            }
        }

        // Need a new block
        ulong blockSize = size > SLAB_SMALL ? SLAB_LARGE : SLAB_SMALL;
        VulkanMemoryBlock newBlock = AllocateBlock(blockSize, typeIdx, required, dedicated: false);
        _blocks.Add(newBlock);
        ulong newOff = newBlock.TryAllocate(size, alignment);
        if (newOff == ulong.MaxValue)
            throw new VulkanException(-2, $"Sub-allocation of {size} bytes failed even in fresh block of {blockSize}");
        return new VulkanAllocation(newBlock.DeviceMemory, newOff, size,
            newBlock.MappedPointer == 0 ? 0 : newBlock.MappedPointer + (nint)newOff,
            typeIdx, newBlock.BlockId);
    }

    private VulkanAllocation AllocateDedicated(ulong size, uint typeIdx, VkMemoryPropertyFlags required)
    {
        VulkanMemoryBlock block = AllocateBlock(size, typeIdx, required, dedicated: true);
        _blocks.Add(block);
        ulong off = block.TryAllocate(size, 64);
        return new VulkanAllocation(block.DeviceMemory, off, size,
            block.MappedPointer == 0 ? 0 : block.MappedPointer + (nint)off,
            typeIdx, block.BlockId);
    }

    private VulkanMemoryBlock AllocateBlock(ulong size, uint typeIdx, VkMemoryPropertyFlags required, bool dedicated)
    {
        VkMemoryAllocateInfo ai = new()
        {
            sType = VkStructureType.MemoryAllocateInfo,
            allocationSize = size,
            memoryTypeIndex = typeIdx,
        };
        VkResult r = VulkanApi.vkAllocateMemory(_device, in ai, 0, out ulong mem);
        if (r == VkResult.ErrorOutOfDeviceMemory && OnOutOfMemory is not null)
        {
            // Drain the command-stream's deferred-free list and any pending command buffers.
            // Mirrors CUDA's cuMemAlloc OOM retry path.
            OnOutOfMemory();
            r = VulkanApi.vkAllocateMemory(_device, in ai, 0, out mem);
        }
        r.ThrowOnError($"vkAllocateMemory(size={size}, typeIdx={typeIdx})");

        nint mapped = 0;
        VkMemoryType mt = _memProps.GetMemoryType((int)typeIdx);
        if ((mt.propertyFlags & VkMemoryPropertyFlags.HostVisible) != 0)
        {
            VulkanApi.vkMapMemory(_device, mem, 0, VkConstants.WholeSize, 0, out mapped)
                .ThrowOnError("vkMapMemory");
        }

        return new VulkanMemoryBlock(_device, mem, size, typeIdx, mt.heapIndex, mt.propertyFlags, mapped, _nextBlockId++, dedicated);
    }

    public void Free(VulkanAllocation a)
    {
        if (a.IsEmpty) return;
        for (int i = 0; i < _blocks.Count; i++)
        {
            if (_blocks[i].BlockId == a.BlockId)
            {
                _blocks[i].Free(a.Offset, a.Size);
                if (_blocks[i].IsDedicated && _blocks[i].LiveAllocations == 0)
                {
                    _blocks[i].Dispose();
                    _blocks.RemoveAt(i);
                }
                return;
            }
        }
    }

    /// <summary>Releases every slab block that has zero live allocations. Called from the OOM
    /// retry path to reclaim memory held by empty slabs that haven't been touched recently.</summary>
    public int ReleaseEmptySlabs()
    {
        int released = 0;
        for (int i = _blocks.Count - 1; i >= 0; i--)
        {
            if (_blocks[i].LiveAllocations == 0)
            {
                _blocks[i].Dispose();
                _blocks.RemoveAt(i);
                released++;
            }
        }
        return released;
    }

    /// <summary>Returns total bytes currently sub-allocated across all blocks of the given heap.</summary>
    public ulong UsedBytes(uint heapIndex)
    {
        ulong used = 0;
        foreach (VulkanMemoryBlock b in _blocks)
        {
            if (b.HeapIndex != heapIndex) continue;
            ulong free = 0;
            foreach ((ulong _, ulong sz) in b.FreeList) free += sz;
            used += b.Size - free;
        }
        return used;
    }

    /// <summary>Returns total reserved bytes (block-level) on the given heap.</summary>
    public ulong ReservedBytes(uint heapIndex)
    {
        ulong total = 0;
        foreach (VulkanMemoryBlock b in _blocks)
            if (b.HeapIndex == heapIndex) total += b.Size;
        return total;
    }

    /// <summary>Number of underlying VkDeviceMemory allocations (counts against the device's 4096-allocation limit).</summary>
    public int BlockCount => _blocks.Count;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (VulkanMemoryBlock b in _blocks) b.Dispose();
        _blocks.Clear();
    }
}

/// <summary>A typed buffer view over a sub-allocated region of <see cref="VkDeviceMemory"/>. Owns its <see cref="VkBuffer"/> handle but not the underlying memory (the allocator owns the slab block).</summary>
public sealed class VulkanBuffer : IDisposable
{
    private readonly nint _device;
    private readonly VulkanMemoryAllocator _allocator;
    public ulong Handle { get; private set; }
    public VulkanAllocation Allocation { get; }
    public ulong Size { get; }
    public VkBufferUsageFlags Usage { get; }

    /// <summary>Returns the host-visible mapped pointer if this buffer is on a HOST_VISIBLE memory type; 0 otherwise.</summary>
    public nint MappedPointer => Allocation.MappedPointer;

    internal VulkanBuffer(nint device, VulkanMemoryAllocator alloc, ulong handle,
                          VulkanAllocation allocation, ulong size, VkBufferUsageFlags usage)
    {
        _device = device;
        _allocator = alloc;
        Handle = handle;
        Allocation = allocation;
        Size = size;
        Usage = usage;
    }

    public unsafe Span<T> AsSpan<T>() where T : unmanaged
    {
        if (Allocation.MappedPointer == 0)
            throw new InvalidOperationException("VulkanBuffer is not host-visible — cannot map. Use staging upload instead.");
        return new Span<T>((void*)Allocation.MappedPointer, (int)(Size / (ulong)sizeof(T)));
    }

    public void Dispose()
    {
        if (Handle != 0)
        {
            VulkanApi.vkDestroyBuffer(_device, Handle, 0);
            Handle = 0;
            _allocator.Free(Allocation);
        }
        GC.SuppressFinalize(this);
    }

    ~VulkanBuffer()
    {
        if (Handle != 0)
        {
            VulkanApi.vkDestroyBuffer(_device, Handle, 0);
            _allocator.Free(Allocation);
        }
    }
}

/// <summary>Static buffer-creation helper: creates a VkBuffer + binds it to allocator-provided memory.</summary>
public static class VulkanBufferFactory
{
    public static VulkanBuffer Create(
        nint device,
        VulkanMemoryAllocator allocator,
        in VkPhysicalDeviceMemoryProperties memProps,
        ulong size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags requiredProps,
        VkMemoryPropertyFlags preferredProps = 0)
    {
        VkBufferCreateInfo bci = new()
        {
            sType = VkStructureType.BufferCreateInfo,
            size = size,
            usage = usage,
            sharingMode = VkSharingMode.Exclusive,
        };
        VulkanApi.vkCreateBuffer(device, in bci, 0, out ulong buffer)
            .ThrowOnError("vkCreateBuffer");

        VulkanApi.vkGetBufferMemoryRequirements(device, buffer, out VkMemoryRequirements req);

        VulkanAllocation alloc = allocator.Allocate(req.size, req.alignment, req.memoryTypeBits, requiredProps, preferredProps);
        VulkanApi.vkBindBufferMemory(device, buffer, alloc.DeviceMemory, alloc.Offset)
            .ThrowOnError("vkBindBufferMemory");

        return new VulkanBuffer(device, allocator, buffer, alloc, size, usage);
    }
}
