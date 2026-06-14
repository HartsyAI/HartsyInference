using System.Runtime.InteropServices;
using System.Text;

namespace HartsyInference.Vulkan;

/// <summary>Identifies a kernel variant: shader file name + a hash of its specialization constants. Two variants of the same .spv with different spec values cache separately.</summary>
public readonly record struct KernelKey(string Name, ulong SpecHash);

/// <summary>One built compute pipeline. Owns its <see cref="Pipeline"/> handle (destroyed by registry on Dispose).</summary>
public sealed class VulkanKernel
{
    public string Name { get; init; } = "";
    public ulong Pipeline { get; init; }
    public ulong PipelineLayout { get; init; }
    public ulong DescriptorSetLayout { get; init; }
    public int StorageBufferCount { get; init; }
}

/// <summary>Loads SPIR-V kernels from disk, builds compute pipelines on first use, and caches them by (name, spec-hash). Pipelines are destroyed together on dispose; one registry per <see cref="VulkanBackend"/> for the device's lifetime, kernels are not unloaded individually.</summary>
public sealed class VulkanKernelRegistry : IDisposable
{
    private readonly nint _device;
    private readonly VulkanCapabilities _caps;
    private readonly VulkanPipelineCache _pipelineCache;
    private readonly VulkanDescriptorManager _descMgr;
    private readonly string _spvDir;

    private readonly Dictionary<KernelKey, VulkanKernel> _kernels = new();
    private readonly Dictionary<string, ulong> _modules = new(); // shader-name -> VkShaderModule (cached for re-spec)

    private static readonly byte[] _mainEntryUtf8 = Encoding.UTF8.GetBytes("main\0");
    private nint _mainEntry;

    public VulkanKernelRegistry(nint device, VulkanCapabilities caps, VulkanPipelineCache pipelineCache, VulkanDescriptorManager descMgr, string spvDir)
    {
        _device = device;
        _caps = caps;
        _pipelineCache = pipelineCache;
        _descMgr = descMgr;
        _spvDir = spvDir;

        _mainEntry = Marshal.AllocHGlobal(_mainEntryUtf8.Length);
        Marshal.Copy(_mainEntryUtf8, 0, _mainEntry, _mainEntryUtf8.Length);
    }

    /// <summary>Builds (or returns cached) a pipeline for the given kernel + spec constants.</summary>
    public VulkanKernel Get(string shaderName, int storageBufferCount, ReadOnlySpan<SpecConstant> specConstants)
    {
        ulong specHash = HashSpec(specConstants);
        KernelKey key = new(shaderName, specHash);
        if (_kernels.TryGetValue(key, out VulkanKernel? cached)) return cached;

        VulkanKernel built = Build(shaderName, storageBufferCount, specConstants);
        _kernels[key] = built;
        return built;
    }

    private VulkanKernel Build(string shaderName, int storageBufferCount, ReadOnlySpan<SpecConstant> specConstants)
    {
        ulong module = GetOrLoadModule(shaderName);
        ulong setLayout = _descMgr.GetSetLayout(storageBufferCount);
        ulong pipelineLayout = _descMgr.GetPipelineLayout(storageBufferCount);

        unsafe
        {
            // Pack spec data + map entries
            int specCount = specConstants.Length;
            int dataBytes = 0;
            for (int i = 0; i < specCount; i++) dataBytes += specConstants[i].SizeBytes;

            nint specDataPtr = dataBytes > 0 ? Marshal.AllocHGlobal(dataBytes) : 0;
            nint mapPtr = specCount > 0 ? Marshal.AllocHGlobal(specCount * sizeof(VkSpecializationMapEntry)) : 0;
            nint subgroupCi = 0;

            try
            {
                int off = 0;
                for (int i = 0; i < specCount; i++)
                {
                    SpecConstant sc = specConstants[i];
                    Span<byte> dst = new((byte*)specDataPtr + off, sc.SizeBytes);
                    sc.WriteTo(dst);
                    VkSpecializationMapEntry entry = new()
                    {
                        constantID = sc.ConstantId,
                        offset = (uint)off,
                        size = (nuint)sc.SizeBytes,
                    };
                    *((VkSpecializationMapEntry*)mapPtr + i) = entry;
                    off += sc.SizeBytes;
                }

                VkSpecializationInfo specInfo = new()
                {
                    mapEntryCount = (uint)specCount,
                    pMapEntries = mapPtr,
                    dataSize = (nuint)dataBytes,
                    pData = specDataPtr,
                };

                // Pin a required-subgroup-size struct only if the device supports it
                VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSizeCi = new()
                {
                    sType = VkStructureType.PipelineShaderStageRequiredSubgroupSizeCreateInfo,
                    requiredSubgroupSize = _caps.SubgroupSize,
                };
                if (_caps.SubgroupSizeControl)
                {
                    subgroupCi = Marshal.AllocHGlobal(Marshal.SizeOf<VkPipelineShaderStageRequiredSubgroupSizeCreateInfo>());
                    Marshal.StructureToPtr(subgroupSizeCi, subgroupCi, false);
                }

                VkPipelineShaderStageCreateInfo stage = new()
                {
                    sType = VkStructureType.PipelineShaderStageCreateInfo,
                    pNext = subgroupCi,
                    flags = _caps.ComputeFullSubgroups ? VkPipelineShaderStageCreateFlags.RequireFullSubgroups : VkPipelineShaderStageCreateFlags.None,
                    stage = VkShaderStageFlags.Compute,
                    module = module,
                    pName = _mainEntry,
                    pSpecializationInfo = (nint)(&specInfo),
                };

                VkComputePipelineCreateInfo ci = new()
                {
                    sType = VkStructureType.ComputePipelineCreateInfo,
                    stage = stage,
                    layout = pipelineLayout,
                    basePipelineIndex = -1,
                };

                ulong pipeline = 0;
                VkResult r = VulkanApi.vkCreateComputePipelines(
                    _device, _pipelineCache.Handle, 1, (nint)(&ci), 0, (nint)(&pipeline));
                r.ThrowOnError($"vkCreateComputePipelines({shaderName})");

                return new VulkanKernel
                {
                    Name = shaderName,
                    Pipeline = pipeline,
                    PipelineLayout = pipelineLayout,
                    DescriptorSetLayout = setLayout,
                    StorageBufferCount = storageBufferCount,
                };
            }
            finally
            {
                if (specDataPtr != 0) Marshal.FreeHGlobal(specDataPtr);
                if (mapPtr != 0) Marshal.FreeHGlobal(mapPtr);
                if (subgroupCi != 0) Marshal.FreeHGlobal(subgroupCi);
            }
        }
    }

    private ulong GetOrLoadModule(string name)
    {
        if (_modules.TryGetValue(name, out ulong cached)) return cached;
        string path = Path.Combine(_spvDir, name);
        if (!path.EndsWith(".spv", StringComparison.OrdinalIgnoreCase)) path += ".spv";
        ulong m = SpirVShaderLoader.LoadModuleFromFile(_device, path);
        _modules[name] = m;
        return m;
    }

    private static ulong HashSpec(ReadOnlySpan<SpecConstant> spec)
    {
        // FNV-1a 64
        ulong h = 0xcbf29ce484222325UL;
        foreach (SpecConstant s in spec)
        {
            h = (h ^ s.ConstantId) * 0x100000001b3UL;
            h = (h ^ (ulong)s.AsUInt32) * 0x100000001b3UL;
        }
        return h;
    }

    public void Dispose()
    {
        foreach ((_, VulkanKernel k) in _kernels) VulkanApi.vkDestroyPipeline(_device, k.Pipeline, 0);
        _kernels.Clear();
        foreach ((_, ulong mod) in _modules) VulkanApi.vkDestroyShaderModule(_device, mod, 0);
        _modules.Clear();
        if (_mainEntry != 0) { Marshal.FreeHGlobal(_mainEntry); _mainEntry = 0; }
        GC.SuppressFinalize(this);
    }
}

/// <summary>One specialization constant entry. Use the static factories: <see cref="UInt"/>, <see cref="Int"/>, <see cref="Float"/>, <see cref="Bool"/>.</summary>
public readonly struct SpecConstant
{
    public uint ConstantId { get; }
    public int SizeBytes { get; }
    public uint AsUInt32 { get; }

    private SpecConstant(uint id, uint value, int size) { ConstantId = id; AsUInt32 = value; SizeBytes = size; }

    public static SpecConstant UInt(uint id, uint v) => new(id, v, 4);
    public static SpecConstant Int(uint id, int v) => new(id, unchecked((uint)v), 4);
    public static SpecConstant Bool(uint id, bool v) => new(id, v ? 1u : 0u, 4);
    public static SpecConstant Float(uint id, float v)
    {
        unsafe { return new SpecConstant(id, *(uint*)&v, 4); }
    }

    public void WriteTo(Span<byte> dst)
    {
        dst[0] = (byte)(AsUInt32 & 0xFF);
        dst[1] = (byte)((AsUInt32 >> 8) & 0xFF);
        dst[2] = (byte)((AsUInt32 >> 16) & 0xFF);
        dst[3] = (byte)((AsUInt32 >> 24) & 0xFF);
    }
}
