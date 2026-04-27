# Vulkan Compute API — Research Notes

SharpInference accesses non-NVIDIA GPUs (and NVIDIA as an alternative) entirely through the **Vulkan Compute API** via P/Invoke — no native shared libraries beyond the system-installed Vulkan loader (`libvulkan.so.1` on Linux, `vulkan-1.dll` on Windows, `libMoltenVK.dylib` on macOS via MoltenVK). Compute shaders are authored in GLSL, compiled to SPIR-V at build time via `glslangValidator`, loaded at runtime via `vkCreateShaderModule`, and dispatched via `vkCmdDispatch`.

Vulkan compute is the **cross-vendor** counterpart to CUDA in this engine: the same model code that targets `IBackend` runs on CUDA on NVIDIA and on Vulkan on AMD / Intel / NVIDIA / ARM Mali / Qualcomm Adreno / Apple Silicon (via MoltenVK). The P/Invoke surface needed for inference is small (~55 functions) compared to a full graphics renderer (~250+), because we only need queues, buffers, descriptors, compute pipelines, command buffers, and synchronization — no swapchain, no render passes, no graphics pipeline state.

The trade-offs versus CUDA: (a) **no cuBLAS equivalent** — we hand-write a tiled GEMM compute shader; (b) **explicit synchronization** — fences, semaphores, and pipeline barriers are mandatory at every queue-submit boundary; (c) **descriptor / command-buffer bookkeeping** — verbose but predictable; (d) **subgroup size varies per vendor** (32 on NVIDIA / Intel Arc, 32 or 64 on AMD RDNA/RDNA2/RDNA3, 64 on AMD GCN), so reductions cannot assume a fixed warp size.

Sources: [Vulkan 1.3 Specification](https://docs.vulkan.org/spec/latest/index.html), [Vulkan API Registry](https://registry.khronos.org/vulkan/), [Khronos Vulkan Guide — Compute](https://docs.vulkan.org/guide/latest/computeshader.html), [VK_EXT_subgroup_size_control proposal](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_subgroup_size_control.html), [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers), [Vulkan-Samples (compute_nbody, compute_op)](https://github.com/KhronosGroup/Vulkan-Samples), [Sascha Willems Vulkan Samples](https://github.com/SaschaWillems/Vulkan), [VkFFT (real Vulkan compute lib in C)](https://github.com/DTolm/VkFFT).

---

## Why Vulkan over OpenCL / ROCm / SYCL

| Property | Vulkan compute | OpenCL | ROCm/HIP | SYCL/Level Zero |
|---|---|---|---|---|
| Pure-C# P/Invoke story | Excellent — single loader DLL | OK — `libOpenCL.so.1` ICD | Poor — many .so files | Poor — DPC++ centric |
| Cross-vendor | NVIDIA, AMD, Intel, Apple (MoltenVK), Mali, Adreno | NVIDIA, AMD, Intel | AMD only | Intel mainly |
| Linux-first | Yes — Mesa RADV (AMD), ANV (Intel), NVIDIA proprietary | Yes but stagnant on NVIDIA (deprecated past 3.0) | Yes — official | Yes |
| SPIR-V kernels | Yes — native IL | Yes (CL 2.1+, optional) | No (LLVM IR via comgr) | Yes |
| Subgroup primitives | Yes — `VK_KHR_shader_subgroup` (core in 1.1) | Yes (CL 2.0 sub-groups) | Yes (HIP wave intrinsics) | Yes (SYCL 2020) |
| FP16 storage + arithmetic | Yes — `shaderFloat16`, `storageBuffer16BitAccess` | Vendor-specific | Yes | Yes |
| Tensor-core / matrix accel | Yes via `VK_KHR_cooperative_matrix` (2024+) or `VK_NV_cooperative_matrix` | Limited | Yes (rocWMMA / MFMA) | Yes |
| Required runtime install | `libvulkan.so.1` (ships with Mesa / driver) | OpenCL ICD loader | Full ROCm stack (~1.5 GB) | oneAPI runtime |

**Decision:** Vulkan is the only single backend that satisfies *cross-vendor on Linux* + *pure C# P/Invoke* + *modern subgroup ops* + *near-native driver presence on every supported distro*. ROCm/OpenCL are listed in the roadmap as optional follow-ons.

---

## Loader Architecture

Vulkan uses a **loader → ICD** model. The application links only against the loader (`libvulkan.so.1`). The loader inspects `/etc/vulkan/icd.d/` JSON manifests and dlopens the correct vendor ICD (`radeon_icd.x86_64.json` → `libvulkan_radeon.so` on Mesa, `nvidia_icd.json` → `libGLX_nvidia.so.0`, `intel_icd.x86_64.json` → `libvulkan_intel.so`). Layers (e.g. validation) come from `/etc/vulkan/explicit_layer.d/` and `~/.local/share/vulkan/explicit_layer.d/`.

### Cross-platform loader resolution (NativeLibrary.SetDllImportResolver)

```csharp
internal static class VulkanLibraryResolver
{
    private const string LibraryName = "vulkan-1";   // logical name in [LibraryImport]

    public static void Register()
    {
        NativeLibrary.SetDllImportResolver(typeof(VulkanLibraryResolver).Assembly, Resolve);
    }

    private static nint Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName != LibraryName) return 0;

        if (OperatingSystem.IsLinux())
        {
            if (NativeLibrary.TryLoad("libvulkan.so.1", out var h)) return h;
            if (NativeLibrary.TryLoad("libvulkan.so",   out h))     return h;
        }
        else if (OperatingSystem.IsWindows())
        {
            if (NativeLibrary.TryLoad("vulkan-1.dll",   out var h)) return h;
        }
        else if (OperatingSystem.IsMacOS())
        {
            if (NativeLibrary.TryLoad("libvulkan.1.dylib", out var h))  return h;
            if (NativeLibrary.TryLoad("libMoltenVK.dylib", out h))      return h;   // MoltenVK direct
        }
        return 0;
    }
}
```

Mirror the existing `CudaLibraryResolver.cs` exactly. The validator `glslangValidator` is **build-time only**; the inference runtime never calls into it.

### Validation layers

Validation is opt-in and provided by `VK_LAYER_KHRONOS_validation`. Enabled at instance creation by passing the layer name. Required at development time, **must be off in production** (5–20× slowdown). On Linux, set `VK_LOADER_DEBUG=warn` to debug ICD resolution.

```csharp
// Enable when SHARPINFERENCE_VK_VALIDATION env var is set
bool enableValidation = Environment.GetEnvironmentVariable("SHARPINFERENCE_VK_VALIDATION") == "1";
```

---

## Vulkan Type & Handle System

All Vulkan handles fall into two categories:

| Category | Examples | C# representation |
|---|---|---|
| **Dispatchable** (pointer-sized) | `VkInstance`, `VkPhysicalDevice`, `VkDevice`, `VkQueue`, `VkCommandBuffer` | `nint` |
| **Non-dispatchable** (always 64-bit, even on 32-bit OS) | `VkBuffer`, `VkDeviceMemory`, `VkPipeline`, `VkShaderModule`, `VkDescriptorSet`, `VkSemaphore`, `VkFence` | `ulong` |

This rule comes from `VK_DEFINE_HANDLE` vs `VK_DEFINE_NON_DISPATCHABLE_HANDLE` in `vk_platform.h`. **Do not** treat non-dispatchable handles as `nint` — the marshalling will be wrong on platforms where pointer ≠ 64 bits.

```csharp
// Dispatchable handles
using VkInstance        = nint;
using VkPhysicalDevice  = nint;
using VkDevice          = nint;
using VkQueue           = nint;
using VkCommandBuffer   = nint;

// Non-dispatchable handles (always 64-bit)
using VkBuffer          = ulong;
using VkDeviceMemory    = ulong;
using VkPipeline        = ulong;
using VkPipelineLayout  = ulong;
using VkPipelineCache   = ulong;
using VkShaderModule    = ulong;
using VkDescriptorSetLayout = ulong;
using VkDescriptorPool      = ulong;
using VkDescriptorSet       = ulong;
using VkCommandPool         = ulong;
using VkSemaphore           = ulong;
using VkFence               = ulong;
using VkEvent               = ulong;
using VkQueryPool           = ulong;
```

> **Note:** `using` aliases are file-scoped in C# and don't change marshalling behavior; the JIT sees `nint` and `ulong` directly. The aliases above are documentation. In code, use the underlying primitive types (`nint`, `ulong`) on P/Invoke signatures.

### `VkResult` enum (key values)

| Value | Name | Meaning |
|---|---|---|
| 0 | `VK_SUCCESS` | OK |
| 1 | `VK_NOT_READY` | Fence/event not signaled |
| 2 | `VK_TIMEOUT` | `vkWaitForFences` timed out |
| 3 | `VK_EVENT_SET` | |
| 4 | `VK_EVENT_RESET` | |
| 5 | `VK_INCOMPLETE` | Returned array partial |
| -1 | `VK_ERROR_OUT_OF_HOST_MEMORY` | malloc failed |
| -2 | `VK_ERROR_OUT_OF_DEVICE_MEMORY` | VRAM exhausted |
| -3 | `VK_ERROR_INITIALIZATION_FAILED` | Driver init error |
| -4 | `VK_ERROR_DEVICE_LOST` | GPU hang / TDR |
| -5 | `VK_ERROR_MEMORY_MAP_FAILED` | `vkMapMemory` failed |
| -6 | `VK_ERROR_LAYER_NOT_PRESENT` | Validation layer missing |
| -7 | `VK_ERROR_EXTENSION_NOT_PRESENT` | Required ext missing |
| -8 | `VK_ERROR_FEATURE_NOT_PRESENT` | Required feature unsupported |
| -9 | `VK_ERROR_INCOMPATIBLE_DRIVER` | Loader can't find ICD |
| -1000257000 | `VK_ERROR_OUT_OF_POOL_MEMORY` | Descriptor pool exhausted |
| -1000069000 | `VK_ERROR_FRAGMENTATION_EXT` | Memory fragmentation |

Negative values are errors; non-negative are success or partial. Helper:

```csharp
public static class VkResultExtensions
{
    public static void ThrowOnError(this VkResult r, string op = "")
    {
        if ((int)r >= 0) return;
        throw new VulkanException($"Vulkan {op} failed: {r} ({(int)r})");
    }
}
```

### `sType` (`VkStructureType`)

Every Vulkan struct begins with `VkStructureType sType` and `void* pNext`. The `sType` value identifies the struct so the driver and validation layers can verify `pNext` chains. Get the enumerator values from the Vulkan registry; the most-used ones for compute are listed below.

| Value | Name |
|---|---|
| 0 | `VK_STRUCTURE_TYPE_APPLICATION_INFO` |
| 1 | `VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO` |
| 2 | `VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO` |
| 3 | `VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO` |
| 4 | `VK_STRUCTURE_TYPE_SUBMIT_INFO` |
| 5 | `VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO` |
| 6 | `VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE` |
| 8 | `VK_STRUCTURE_TYPE_FENCE_CREATE_INFO` |
| 9 | `VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO` |
| 12 | `VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO` |
| 15 | `VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO` |
| 17 | `VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO` |
| 18 | `VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO` |
| 28 | `VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO` |
| 30 | `VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO` |
| 32 | `VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO` |
| 33 | `VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO` |
| 34 | `VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO` |
| 35 | `VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET` |
| 39 | `VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO` |
| 40 | `VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO` |
| 42 | `VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO` |
| 44 | `VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER` |
| 46 | `VK_STRUCTURE_TYPE_MEMORY_BARRIER` |
| 1000094000 | `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES` |
| 1000059000 | `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2` |
| 1000059001 | `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2` |
| 1000225000 | `VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO` |
| 1000295000 | `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR` |
| 1000241000 | `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES` |

Always set `sType` and zero `pNext` unless chaining a `*FeatureFlags2` struct.

---

## Instance Management

Source: [Khronos Vulkan Spec §3 Instance](https://docs.vulkan.org/spec/latest/chapters/initialization.html).

C signatures:

```c
VkResult vkCreateInstance(
    const VkInstanceCreateInfo*  pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkInstance*                  pInstance);

void vkDestroyInstance(
    VkInstance                    instance,
    const VkAllocationCallbacks*  pAllocator);

PFN_vkVoidFunction vkGetInstanceProcAddr(VkInstance instance, const char* pName);
```

C# P/Invoke:

```csharp
[LibraryImport("vulkan-1", EntryPoint = "vkCreateInstance")]
public static partial VkResult vkCreateInstance(
    in VkInstanceCreateInfo pCreateInfo,
    nint pAllocator,
    out nint pInstance);

[LibraryImport("vulkan-1", EntryPoint = "vkDestroyInstance")]
public static partial void vkDestroyInstance(nint instance, nint pAllocator);
```

### `VkInstanceCreateInfo`

```csharp
[StructLayout(LayoutKind.Sequential)]
public struct VkApplicationInfo
{
    public VkStructureType  sType;            // VK_STRUCTURE_TYPE_APPLICATION_INFO
    public nint             pNext;
    public nint             pApplicationName; // const char*
    public uint             applicationVersion;
    public nint             pEngineName;
    public uint             engineVersion;
    public uint             apiVersion;       // VK_API_VERSION_1_3
}

[StructLayout(LayoutKind.Sequential)]
public struct VkInstanceCreateInfo
{
    public VkStructureType  sType;            // VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
    public nint             pNext;
    public uint             flags;            // 0
    public nint             pApplicationInfo; // const VkApplicationInfo*
    public uint             enabledLayerCount;
    public nint             ppEnabledLayerNames;     // const char* const*
    public uint             enabledExtensionCount;
    public nint             ppEnabledExtensionNames; // const char* const*
}
```

API-version selection: target **`VK_API_VERSION_1_3`** (`(1<<22) | (3<<12) | 0` = `0x00403000`). 1.3 promotes `VK_EXT_subgroup_size_control`, `VK_KHR_shader_non_semantic_info`, `VK_KHR_shader_integer_dot_product`, `VK_KHR_synchronization2`, and `VK_KHR_dynamic_rendering` to core. Linux drivers from 2022+ universally support 1.3. Fall back to 1.2 only if needed; 1.0 is too restrictive for FP16/subgroup work.

### Required instance extensions (none — keep minimal)

For a headless compute-only engine we need **zero instance extensions**. Optional:
- `VK_KHR_get_physical_device_properties2` — was an instance extension pre-1.1, now core; do not request explicitly.
- `VK_EXT_debug_utils` — only when validation is enabled.

### Layers

Only one layer: `VK_LAYER_KHRONOS_validation`, opt-in via env var.

---

## Physical Device Enumeration & Selection

```c
VkResult vkEnumeratePhysicalDevices(
    VkInstance         instance,
    uint32_t*          pPhysicalDeviceCount,
    VkPhysicalDevice*  pPhysicalDevices);

void vkGetPhysicalDeviceProperties(
    VkPhysicalDevice            physicalDevice,
    VkPhysicalDeviceProperties* pProperties);

void vkGetPhysicalDeviceProperties2(
    VkPhysicalDevice              physicalDevice,
    VkPhysicalDeviceProperties2*  pProperties);

void vkGetPhysicalDeviceFeatures2(
    VkPhysicalDevice            physicalDevice,
    VkPhysicalDeviceFeatures2*  pFeatures);

void vkGetPhysicalDeviceMemoryProperties(
    VkPhysicalDevice                  physicalDevice,
    VkPhysicalDeviceMemoryProperties* pMemoryProperties);

void vkGetPhysicalDeviceQueueFamilyProperties(
    VkPhysicalDevice          physicalDevice,
    uint32_t*                 pQueueFamilyPropertyCount,
    VkQueueFamilyProperties*  pQueueFamilyProperties);
```

C# P/Invoke (the two-call pattern is universal: first call with null array to get count, then allocate, then call again):

```csharp
[LibraryImport("vulkan-1", EntryPoint = "vkEnumeratePhysicalDevices")]
public static partial VkResult vkEnumeratePhysicalDevices(
    nint instance, ref uint pCount, [Out] nint[]? pPhysicalDevices);

[LibraryImport("vulkan-1", EntryPoint = "vkGetPhysicalDeviceProperties2")]
public static partial void vkGetPhysicalDeviceProperties2(
    nint physicalDevice, ref VkPhysicalDeviceProperties2 props);

[LibraryImport("vulkan-1", EntryPoint = "vkGetPhysicalDeviceFeatures2")]
public static partial void vkGetPhysicalDeviceFeatures2(
    nint physicalDevice, ref VkPhysicalDeviceFeatures2 features);

[LibraryImport("vulkan-1", EntryPoint = "vkGetPhysicalDeviceMemoryProperties")]
public static partial void vkGetPhysicalDeviceMemoryProperties(
    nint physicalDevice, out VkPhysicalDeviceMemoryProperties memProps);

[LibraryImport("vulkan-1", EntryPoint = "vkGetPhysicalDeviceQueueFamilyProperties")]
public static partial void vkGetPhysicalDeviceQueueFamilyProperties(
    nint physicalDevice, ref uint pCount, [Out] VkQueueFamilyProperties[]? pProps);
```

### `VkPhysicalDeviceProperties` (~824 bytes)

The full struct contains: `apiVersion`, `driverVersion`, `vendorID`, `deviceID`, `deviceType`, `deviceName[256]`, `pipelineCacheUUID[16]`, `limits` (a `VkPhysicalDeviceLimits` struct ~488 bytes), `sparseProperties`. **Use `vkGetPhysicalDeviceProperties2`** so you can chain `VkPhysicalDeviceSubgroupProperties` via `pNext`.

Critical fields to capture into `BackendCapabilities`:

| Field | Use |
|---|---|
| `vendorID` | `0x10DE` NVIDIA, `0x1002` AMD, `0x8086` Intel, `0x13B5` ARM (Mali), `0x5143` Qualcomm, `0x106B` Apple |
| `deviceType` | `0` other, `1` integrated, `2` discrete, `3` virtual, `4` CPU. Prefer **discrete (2)**. |
| `deviceName[256]` | UTF-8, NUL-terminated. Use `Encoding.UTF8.GetString` until first 0. |
| `limits.maxComputeWorkGroupCount[3]` | ≥ (65535, 65535, 65535) on every desktop GPU. |
| `limits.maxComputeWorkGroupInvocations` | min spec 128; modern desktop ≥ 1024. |
| `limits.maxComputeWorkGroupSize[3]` | min spec (128, 128, 64); modern desktop (1024, 1024, 64). |
| `limits.maxComputeSharedMemorySize` | min spec 16 KB; AMD/NV desktop 32–48 KB; Apple 32 KB. |
| `limits.maxStorageBufferRange` | min spec 128 MB; modern GPUs ≥ 4 GB. |
| `limits.maxPushConstantsSize` | min spec **128 bytes** — enforce ≤ 128 in code. |
| `limits.maxPerStageDescriptorStorageBuffers` | min spec **4**, modern desktop ≥ 16. Treat as **scarce**. |
| `limits.minStorageBufferOffsetAlignment` | typically 16, 64, or 256 — used when sub-allocating. |
| `limits.nonCoherentAtomSize` | typically 64 or 256 — flush range alignment. |
| `limits.minMemoryMapAlignment` | typically 64. |
| `limits.optimalBufferCopyOffsetAlignment` | typically 1 or 4. |
| `limits.timestampPeriod` | nanoseconds-per-timestamp-tick (for `vkCmdWriteTimestamp`-based profiling). |

Persist these into a `VulkanCapabilities` struct created at `VulkanBackend` construction; refer to it before kernel dispatch (e.g., to pick a workgroup size that fits).

### `VkPhysicalDeviceSubgroupProperties`

```csharp
[StructLayout(LayoutKind.Sequential)]
public struct VkPhysicalDeviceSubgroupProperties
{
    public VkStructureType  sType;          // 1000094000
    public nint             pNext;
    public uint             subgroupSize;   // 32 NV, 32/64 AMD RDNA, 64 GCN, 8/16/32 Intel
    public uint             supportedStages;          // VkShaderStageFlags
    public uint             supportedOperations;      // VkSubgroupFeatureFlags
    public uint             quadOperationsInAllStages;// VkBool32
}
```

`supportedOperations` is a bitmask:

| Bit | Name | Required min |
|---|---|---|
| 0x00000001 | `VK_SUBGROUP_FEATURE_BASIC_BIT` | guaranteed when GFX or Compute queue exists |
| 0x00000002 | `VK_SUBGROUP_FEATURE_VOTE_BIT` | optional |
| 0x00000004 | `VK_SUBGROUP_FEATURE_ARITHMETIC_BIT` | needed for `subgroupAdd` / `subgroupMul` / `subgroupMin` / `subgroupMax` |
| 0x00000008 | `VK_SUBGROUP_FEATURE_BALLOT_BIT` | optional |
| 0x00000010 | `VK_SUBGROUP_FEATURE_SHUFFLE_BIT` | needed for `subgroupShuffle` (warp shuffle) |
| 0x00000020 | `VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT` | needed for `subgroupShuffleXor` (butterfly) |
| 0x00000040 | `VK_SUBGROUP_FEATURE_CLUSTERED_BIT` | optional |
| 0x00000080 | `VK_SUBGROUP_FEATURE_QUAD_BIT` | required for `quadSwap*` |

For our reductions in GroupNorm / SDPA / softmax we need **ARITHMETIC + SHUFFLE + SHUFFLE_RELATIVE**. All three are supported on every desktop GPU since 2018 (NVIDIA Pascal+, AMD GCN 5+, Intel Gen9+). Validate at startup; fail with `UnsupportedDeviceException` if missing.

### `VkPhysicalDeviceVulkan13Features` (FP16, subgroups, sync2)

```csharp
[StructLayout(LayoutKind.Sequential)]
public struct VkPhysicalDeviceVulkan11Features
{
    public VkStructureType  sType;       // 1000175000
    public nint             pNext;
    public uint             storageBuffer16BitAccess;       // VkBool32 — FP16 in SSBOs
    public uint             uniformAndStorageBuffer16BitAccess;
    public uint             storagePushConstant16;
    public uint             storageInputOutput16;
    public uint             multiview;
    public uint             multiviewGeometryShader;
    public uint             multiviewTessellationShader;
    public uint             variablePointersStorageBuffer;
    public uint             variablePointers;
    public uint             protectedMemory;
    public uint             samplerYcbcrConversion;
    public uint             shaderDrawParameters;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkPhysicalDeviceVulkan12Features
{
    public VkStructureType  sType;       // 1000196000
    public nint             pNext;
    public uint             samplerMirrorClampToEdge;
    public uint             drawIndirectCount;
    public uint             storageBuffer8BitAccess;
    public uint             uniformAndStorageBuffer8BitAccess;
    public uint             storagePushConstant8;
    public uint             shaderBufferInt64Atomics;
    public uint             shaderSharedInt64Atomics;
    public uint             shaderFloat16;                  // ← FP16 arithmetic in shaders
    public uint             shaderInt8;
    // ... (50+ fields total) ...
}

[StructLayout(LayoutKind.Sequential)]
public struct VkPhysicalDeviceVulkan13Features
{
    public VkStructureType  sType;       // 1000241000
    public nint             pNext;
    public uint             robustImageAccess;
    public uint             inlineUniformBlock;
    public uint             descriptorBindingInlineUniformBlockUpdateAfterBind;
    public uint             pipelineCreationCacheControl;
    public uint             privateData;
    public uint             shaderDemoteToHelperInvocation;
    public uint             shaderTerminateInvocation;
    public uint             subgroupSizeControl;            // ← required to pin wave size
    public uint             computeFullSubgroups;
    public uint             synchronization2;
    public uint             textureCompressionASTC_HDR;
    public uint             shaderZeroInitializeWorkgroupMemory;
    public uint             dynamicRendering;
    public uint             shaderIntegerDotProduct;
    public uint             maintenance4;
}
```

**Required at device creation:** `shaderFloat16 = 1`, `storageBuffer16BitAccess = 1`, `subgroupSizeControl = 1`, `computeFullSubgroups = 1`, `synchronization2 = 1`. If any are unsupported, fall back: FP16 → FP32 path; subgroupSizeControl → query `subgroupSize` and bake it into shader constants at compile time.

### `VkPhysicalDeviceMemoryProperties`

```csharp
public const int VK_MAX_MEMORY_TYPES = 32;
public const int VK_MAX_MEMORY_HEAPS = 16;

[StructLayout(LayoutKind.Sequential)]
public struct VkMemoryType
{
    public uint propertyFlags;   // VkMemoryPropertyFlags
    public uint heapIndex;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkMemoryHeap
{
    public ulong size;
    public uint  flags;          // VkMemoryHeapFlags
}

[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkPhysicalDeviceMemoryProperties
{
    public uint memoryTypeCount;
    public fixed byte memoryTypes_raw[ /* 32 * sizeof(VkMemoryType) = 32 * 8 */ 256];
    public uint memoryHeapCount;
    public fixed byte memoryHeaps_raw[ /* 16 * sizeof(VkMemoryHeap)  = 16 * 16 */ 256];
}
```

`VkMemoryPropertyFlags`:

| Bit | Name | Use |
|---|---|---|
| 0x01 | `DEVICE_LOCAL` | VRAM — fast, GPU-only |
| 0x02 | `HOST_VISIBLE` | Mappable from CPU |
| 0x04 | `HOST_COHERENT` | No need to flush/invalidate |
| 0x08 | `HOST_CACHED` | CPU-cached; flush before GPU read |
| 0x10 | `LAZILY_ALLOCATED` | (transient images only) |
| 0x40 | `DEVICE_COHERENT_AMD` | AMD-only fine-grained coherence |

Discovery rules used by `VulkanMemoryAllocator`:

- **Device-local (weights & activations):** prefer `DEVICE_LOCAL` only.
- **Staging upload (CPU→GPU):** prefer `HOST_VISIBLE | HOST_COHERENT`. Avoid `HOST_CACHED` for write-only stage buffers.
- **Readback (GPU→CPU):** prefer `HOST_VISIBLE | HOST_CACHED | HOST_COHERENT`. If `HOST_COHERENT` not set, must `vkInvalidateMappedMemoryRanges` before reading.
- **AMD ReBAR / Smart Access Memory:** modern AMD/Intel/NV devices expose a special memory type with both `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` (sometimes 256 MB on older HW, full VRAM on ReBAR). When present, use it for hot weight/activation buffers — it eliminates the staging copy.

Full algorithm in [VULKAN_MEMORY_MANAGEMENT.md](VULKAN_MEMORY_MANAGEMENT.md).

### `VkQueueFamilyProperties`

```csharp
[StructLayout(LayoutKind.Sequential)]
public struct VkQueueFamilyProperties
{
    public uint queueFlags;        // VkQueueFlags
    public uint queueCount;
    public uint timestampValidBits;
    public VkExtent3D minImageTransferGranularity;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkExtent3D { public uint width, height, depth; }
```

`VkQueueFlags` bits: `0x1 GRAPHICS`, `0x2 COMPUTE`, `0x4 TRANSFER`, `0x8 SPARSE_BINDING`. **Selection rule for compute-only:**

1. Prefer a family that has `COMPUTE` set but **not** `GRAPHICS` — this is typically a dedicated async-compute queue (AMD ACE, NVIDIA AsyncCompute) and avoids contention with display work.
2. Fall back to any family with `COMPUTE`.
3. `TRANSFER` is implied by `COMPUTE | GRAPHICS` per spec, but if a transfer-only family exists, optionally use it for staging copies in parallel with compute.

For inference we use **one logical queue from one family** in the v1 backend; multi-queue is a later optimization (Phase 7+).

### Device selection ranking

```
score = 0
+ 4096   if deviceType == DISCRETE_GPU
+ 1024   if deviceType == INTEGRATED_GPU
+ 0      if deviceType == CPU
+ 256    if FP16 (shaderFloat16) supported
+ 256    if SubgroupSizeControl supported
+ 128    if Cooperative-Matrix extension supported
+ 64     if VRAM heap > 8 GB
+ deviceVRAM_GB
- 9999   if missing required compute queue
- 9999   if missing arithmetic + shuffle + shuffle_relative subgroup features
```

Tie-break by `deviceID` then ordinal. Persist the chosen device's `deviceLUID` (Vulkan 1.1 core, `VkPhysicalDeviceIDProperties`) so users can pin to the same GPU across launches.

---

## Logical Device & Queue Creation

```c
VkResult vkCreateDevice(
    VkPhysicalDevice           physicalDevice,
    const VkDeviceCreateInfo*  pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDevice*                  pDevice);

void vkDestroyDevice(VkDevice device, const VkAllocationCallbacks* pAllocator);

void vkGetDeviceQueue(VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex, VkQueue* pQueue);

VkResult vkDeviceWaitIdle(VkDevice device);
```

```csharp
[StructLayout(LayoutKind.Sequential)]
public struct VkDeviceQueueCreateInfo
{
    public VkStructureType  sType;       // 2
    public nint             pNext;
    public uint             flags;       // 0
    public uint             queueFamilyIndex;
    public uint             queueCount;
    public nint             pQueuePriorities; // const float* — len queueCount, values in [0,1]
}

[StructLayout(LayoutKind.Sequential)]
public struct VkDeviceCreateInfo
{
    public VkStructureType  sType;       // 3
    public nint             pNext;       // chain VkPhysicalDeviceFeatures2 here, NOT pEnabledFeatures
    public uint             flags;       // 0
    public uint             queueCreateInfoCount;
    public nint             pQueueCreateInfos;        // const VkDeviceQueueCreateInfo*
    public uint             enabledLayerCount;        // deprecated — set 0
    public nint             ppEnabledLayerNames;      // deprecated — set null
    public uint             enabledExtensionCount;
    public nint             ppEnabledExtensionNames;  // const char* const*
    public nint             pEnabledFeatures;         // null — use pNext chain instead
}
```

### Required device extensions (Vulkan 1.3)

| Extension | Why | Promoted in |
|---|---|---|
| `VK_KHR_synchronization2` | Modern barrier API | 1.3 core (still must enable explicitly on 1.2 path) |
| `VK_KHR_shader_float16_int8` | FP16 + Int8 in shaders | 1.2 core |
| `VK_EXT_subgroup_size_control` | Pin wave size at pipeline create | 1.3 core |
| `VK_KHR_push_descriptor` | Avoid descriptor pool churn (optional) | not core |
| `VK_KHR_16bit_storage` | FP16 in storage buffers | 1.1 core |
| `VK_EXT_memory_budget` | Real-time VRAM usage queries | not core |
| `VK_KHR_buffer_device_address` | Pointer-style buffer addressing | 1.2 core |
| `VK_KHR_cooperative_matrix` | (Optional, Phase 4) Tensor-core matrix ops | not core |

On Vulkan 1.3 most are core; we still must enable the **non-core** ones (`VK_EXT_memory_budget`, optional `VK_KHR_push_descriptor`, optional `VK_KHR_cooperative_matrix`).

### Feature-chain pattern

```csharp
var features13 = new VkPhysicalDeviceVulkan13Features
{
    sType = VkStructureType.PhysicalDeviceVulkan13Features,
    subgroupSizeControl = 1, computeFullSubgroups = 1,
    synchronization2 = 1, maintenance4 = 1
};
var features12 = new VkPhysicalDeviceVulkan12Features
{
    sType = VkStructureType.PhysicalDeviceVulkan12Features,
    shaderFloat16 = 1, storageBuffer8BitAccess = 0,
    bufferDeviceAddress = 1
};
var features11 = new VkPhysicalDeviceVulkan11Features
{
    sType = VkStructureType.PhysicalDeviceVulkan11Features,
    storageBuffer16BitAccess = 1
};
var features2 = new VkPhysicalDeviceFeatures2
{
    sType = VkStructureType.PhysicalDeviceFeatures2,
    features = default     // enable nothing legacy
};

// Build pNext chain manually with pinned pointers
fixed (VkPhysicalDeviceVulkan13Features* p13 = &features13)
fixed (VkPhysicalDeviceVulkan12Features* p12 = &features12)
fixed (VkPhysicalDeviceVulkan11Features* p11 = &features11)
{
    features12.pNext = (nint)p13;
    features11.pNext = (nint)p12;
    features2.pNext  = (nint)p11;

    var queuePri = stackalloc float[] { 1.0f };
    var queueCi = new VkDeviceQueueCreateInfo
    {
        sType = VkStructureType.DeviceQueueCreateInfo,
        queueFamilyIndex = computeFamilyIndex,
        queueCount = 1,
        pQueuePriorities = (nint)queuePri
    };

    var deviceCi = new VkDeviceCreateInfo
    {
        sType = VkStructureType.DeviceCreateInfo,
        pNext = (nint)(&features2),
        queueCreateInfoCount = 1,
        pQueueCreateInfos = (nint)(&queueCi),
        enabledExtensionCount = (uint)extNames.Length,
        ppEnabledExtensionNames = ConstStringArray.Pin(extNames)
    };
    VulkanApi.vkCreateDevice(phys, in deviceCi, 0, out _device).ThrowOnError("vkCreateDevice");
}

VulkanApi.vkGetDeviceQueue(_device, computeFamilyIndex, 0, out _queue);
```

`ConstStringArray.Pin` — utility that copies strings into a single pinned UTF-8 block and returns `const char**`. See `dotLLM`'s pattern for `ppEnabledExtensionNames` reuse.

---

## Buffer & Memory API

```c
VkResult vkCreateBuffer(VkDevice, const VkBufferCreateInfo*, const VkAllocationCallbacks*, VkBuffer*);
void     vkDestroyBuffer(VkDevice, VkBuffer, const VkAllocationCallbacks*);

void     vkGetBufferMemoryRequirements(VkDevice, VkBuffer, VkMemoryRequirements*);

VkResult vkAllocateMemory(VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*, VkDeviceMemory*);
void     vkFreeMemory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*);

VkResult vkBindBufferMemory(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize memoryOffset);

VkResult vkMapMemory(VkDevice, VkDeviceMemory, VkDeviceSize offset, VkDeviceSize size, uint32_t flags, void** ppData);
void     vkUnmapMemory(VkDevice, VkDeviceMemory);

VkResult vkFlushMappedMemoryRanges(VkDevice, uint32_t rangeCount, const VkMappedMemoryRange*);
VkResult vkInvalidateMappedMemoryRanges(VkDevice, uint32_t rangeCount, const VkMappedMemoryRange*);
```

```csharp
[StructLayout(LayoutKind.Sequential)]
public struct VkBufferCreateInfo
{
    public VkStructureType sType;        // 12
    public nint            pNext;
    public uint            flags;        // 0 (or SPARSE_*)
    public ulong           size;         // VkDeviceSize
    public uint            usage;        // VkBufferUsageFlags
    public uint            sharingMode;  // 0=EXCLUSIVE, 1=CONCURRENT
    public uint            queueFamilyIndexCount;
    public nint            pQueueFamilyIndices;
}
```

`VkBufferUsageFlags` (compute-relevant only):

| Bit | Name |
|---|---|
| 0x0001 | `TRANSFER_SRC_BIT` |
| 0x0002 | `TRANSFER_DST_BIT` |
| 0x0010 | `UNIFORM_TEXEL_BUFFER_BIT` |
| 0x0020 | `STORAGE_TEXEL_BUFFER_BIT` |
| 0x0040 | `UNIFORM_BUFFER_BIT` |
| 0x0080 | `STORAGE_BUFFER_BIT` ← all tensor data |
| 0x0100 | `INDEX_BUFFER_BIT` |
| 0x0200 | `VERTEX_BUFFER_BIT` |
| 0x0400 | `INDIRECT_BUFFER_BIT` |
| 0x00020000 | `SHADER_DEVICE_ADDRESS_BIT_KHR` (needs `bufferDeviceAddress`) |

For tensors: `STORAGE_BUFFER_BIT | TRANSFER_SRC_BIT | TRANSFER_DST_BIT`. Add `SHADER_DEVICE_ADDRESS_BIT_KHR` if we adopt `VK_KHR_buffer_device_address` for pointer-style addressing (lets shaders dereference 64-bit GPU pointers and avoid descriptor binding for some scratch buffers).

```csharp
[StructLayout(LayoutKind.Sequential)]
public struct VkMemoryRequirements
{
    public ulong size;             // bytes to allocate (rounded up by driver)
    public ulong alignment;        // typically 16, 64, or 256
    public uint  memoryTypeBits;   // bitmask of compatible memoryTypes indices
}

[StructLayout(LayoutKind.Sequential)]
public struct VkMemoryAllocateInfo
{
    public VkStructureType sType;            // 5
    public nint            pNext;
    public ulong           allocationSize;
    public uint            memoryTypeIndex;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkMappedMemoryRange
{
    public VkStructureType sType;   // 6
    public nint            pNext;
    public ulong           memory;  // VkDeviceMemory
    public ulong           offset;
    public ulong           size;    // VK_WHOLE_SIZE = 0xFFFFFFFFFFFFFFFFul
}
```

**Allocation count limits:** the spec guarantees only 4096 simultaneous `VkDeviceMemory` allocations. Modern desktop drivers raise this (Mesa RADV: 4096, NVIDIA Linux: 4096, Mesa ANV: 4096). Mobile is much lower (Mali ~256). **You must sub-allocate** — see [VULKAN_MEMORY_MANAGEMENT.md](VULKAN_MEMORY_MANAGEMENT.md).

`vkMapMemory` flags must be 0. Mapping is persistent (no need to unmap before each access). Coherence:

- If memory type has `HOST_COHERENT` → no flush/invalidate needed.
- Else: write → `vkFlushMappedMemoryRanges`; read → `vkInvalidateMappedMemoryRanges`. Range offset/size must be aligned to `nonCoherentAtomSize`.

---

## Shader Modules & Compute Pipelines

```c
VkResult vkCreateShaderModule(VkDevice, const VkShaderModuleCreateInfo*, const VkAllocationCallbacks*, VkShaderModule*);
void     vkDestroyShaderModule(VkDevice, VkShaderModule, const VkAllocationCallbacks*);

VkResult vkCreatePipelineLayout(VkDevice, const VkPipelineLayoutCreateInfo*, const VkAllocationCallbacks*, VkPipelineLayout*);
VkResult vkCreatePipelineCache(VkDevice, const VkPipelineCacheCreateInfo*, const VkAllocationCallbacks*, VkPipelineCache*);
VkResult vkGetPipelineCacheData(VkDevice, VkPipelineCache, size_t* pDataSize, void* pData);

VkResult vkCreateComputePipelines(
    VkDevice, VkPipelineCache,
    uint32_t createInfoCount, const VkComputePipelineCreateInfo*,
    const VkAllocationCallbacks*, VkPipeline*);

void     vkDestroyPipeline(VkDevice, VkPipeline, const VkAllocationCallbacks*);
```

```csharp
[StructLayout(LayoutKind.Sequential)]
public struct VkShaderModuleCreateInfo
{
    public VkStructureType sType;    // 15
    public nint            pNext;
    public uint            flags;
    public nuint           codeSize; // bytes; must be multiple of 4
    public nint            pCode;    // const uint32_t* — SPIR-V words
}

[StructLayout(LayoutKind.Sequential)]
public struct VkSpecializationMapEntry
{
    public uint  constantID;
    public uint  offset;       // bytes into pData
    public nuint size;         // bytes
}

[StructLayout(LayoutKind.Sequential)]
public struct VkSpecializationInfo
{
    public uint  mapEntryCount;
    public nint  pMapEntries;        // const VkSpecializationMapEntry*
    public nuint dataSize;
    public nint  pData;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkPipelineShaderStageCreateInfo
{
    public VkStructureType sType;       // 18
    public nint            pNext;       // chain VkPipelineShaderStageRequiredSubgroupSizeCreateInfo here
    public uint            flags;       // VkPipelineShaderStageCreateFlags
    public uint            stage;       // VkShaderStageFlagBits — COMPUTE = 0x20
    public ulong           module;      // VkShaderModule
    public nint            pName;       // const char* — entry point, usually "main"
    public nint            pSpecializationInfo;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkComputePipelineCreateInfo
{
    public VkStructureType sType;            // 28
    public nint            pNext;
    public uint            flags;            // VkPipelineCreateFlags
    public VkPipelineShaderStageCreateInfo stage;
    public ulong           layout;           // VkPipelineLayout
    public ulong           basePipelineHandle;
    public int             basePipelineIndex;
}
```

### Specialization constants

These are **the** mechanism for compile-time-constant kernel parameters in SPIR-V (workgroup size, tile size, channel count, etc.). Set in GLSL via `layout(constant_id = N) const uint TILE = 32;`. The driver re-runs the SPIR-V optimizer after substituting your values, producing a kernel as efficient as if those constants were hard-coded. We use this heavily for:

- Workgroup size (`local_size_x/y/z`)
- Tile dimensions (`TILE_M`, `TILE_N`, `TILE_K`)
- Whether FP16 path is enabled
- Activation type (SiLU=0, GELU=1, GELU_TANH=2, …)
- Subgroup size (set to detected `subgroupSize` so the GLSL can hard-code reduction stride)

### `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`

Promoted to 1.3 core. Lets us pin the subgroup size at pipeline creation time so a single SPIR-V file produces optimal code on AMD (force wave32) and Intel (force 32) without extra branches.

```csharp
[StructLayout(LayoutKind.Sequential)]
public struct VkPipelineShaderStageRequiredSubgroupSizeCreateInfo
{
    public VkStructureType sType;          // 1000225000
    public nint            pNext;
    public uint            requiredSubgroupSize;  // power-of-two within
                                                   // [minSubgroupSize, maxSubgroupSize]
}
```

Pair with the pipeline-stage flag `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT` (`0x00000008`) to require that the workgroup size be a multiple of the subgroup size — needed for full subgroup-arithmetic reductions to behave correctly with no idle lanes.

Query the legal range from `VkPhysicalDeviceSubgroupSizeControlProperties` (`sType` 1000225001):

```csharp
[StructLayout(LayoutKind.Sequential)]
public struct VkPhysicalDeviceSubgroupSizeControlProperties
{
    public VkStructureType sType;
    public nint            pNext;
    public uint            minSubgroupSize;            // typical: 8 (Intel), 32 (NV/AMD-RDNA), 64 (AMD-GCN)
    public uint            maxSubgroupSize;            // typical: 32 (NV), 64 (AMD)
    public uint            maxComputeWorkgroupSubgroups; // workgroup-invocations / minSubgroupSize
    public uint            requiredSubgroupSizeStages; // bitmask
}
```

### Pipeline cache

`VkPipelineCache` saves driver-compiled binaries across runs. Persist its blob to `~/.cache/sharpinference/vulkan/<deviceUUID>.pipeline_cache` to skip ~50–500 ms re-compile on every launch.

```csharp
// On startup:
byte[]? cacheData = TryReadCacheFile();
var cacheCi = new VkPipelineCacheCreateInfo {
    sType = VkStructureType.PipelineCacheCreateInfo,
    initialDataSize = (nuint)(cacheData?.Length ?? 0),
    pInitialData    = cacheData != null ? PinBytes(cacheData) : 0
};
vkCreatePipelineCache(device, in cacheCi, 0, out _pipelineCache);

// On shutdown:
nuint sz = 0;
vkGetPipelineCacheData(device, _pipelineCache, ref sz, 0);
byte[] data = new byte[sz];
fixed (byte* p = data) vkGetPipelineCacheData(device, _pipelineCache, ref sz, (nint)p);
File.WriteAllBytes(cachePath, data);
```

The cache contents include a UUID matching the physical device — invalid on a different GPU/driver and the driver will gracefully ignore it.

### Pipeline creation flow

```
1. Read .spv from disk (Assembly-relative path: "Spirv/groupnorm_silu.spv")
2. vkCreateShaderModule
3. Build VkSpecializationInfo with concrete values (TILE_M=128, USE_FP16=1, etc.)
4. Build VkPipelineShaderStageCreateInfo, chain RequiredSubgroupSize via pNext
5. Reuse pre-built VkPipelineLayout (one per descriptor-set-layout shape)
6. vkCreateComputePipelines(cache, ...) — JIT-compile to SASS / GCN / Intel ISA
7. vkDestroyShaderModule (module no longer needed once pipeline is built)
8. Cache pipeline handle in Dictionary<KernelKey, ulong> for reuse
```

`KernelKey` = `(KernelName, SpecializationHash)` so two specializations of the same shader (e.g. F32 vs F16) cache independently.

---

## Descriptor Sets

Vulkan binds shader resources via descriptor sets, not direct pointers. There is **a lot of state**, but for inference we use a single pattern repeated per kernel.

```c
VkResult vkCreateDescriptorSetLayout(VkDevice, const VkDescriptorSetLayoutCreateInfo*, const VkAllocationCallbacks*, VkDescriptorSetLayout*);
VkResult vkCreateDescriptorPool(VkDevice, const VkDescriptorPoolCreateInfo*, const VkAllocationCallbacks*, VkDescriptorPool*);
VkResult vkAllocateDescriptorSets(VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet*);
VkResult vkResetDescriptorPool(VkDevice, VkDescriptorPool, VkDescriptorPoolResetFlags);
void     vkUpdateDescriptorSets(VkDevice, uint32_t writeCount, const VkWriteDescriptorSet*, uint32_t copyCount, const VkCopyDescriptorSet*);
```

```csharp
[StructLayout(LayoutKind.Sequential)]
public struct VkDescriptorSetLayoutBinding
{
    public uint  binding;             // GLSL `layout(binding = N)`
    public uint  descriptorType;      // VkDescriptorType
    public uint  descriptorCount;     // 1 unless arrayed
    public uint  stageFlags;          // VK_SHADER_STAGE_COMPUTE_BIT = 0x20
    public nint  pImmutableSamplers;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkDescriptorPoolSize
{
    public uint type;                 // VkDescriptorType
    public uint descriptorCount;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkDescriptorBufferInfo
{
    public ulong buffer;              // VkBuffer
    public ulong offset;
    public ulong range;               // VK_WHOLE_SIZE valid
}

[StructLayout(LayoutKind.Sequential)]
public struct VkWriteDescriptorSet
{
    public VkStructureType sType;     // 35
    public nint            pNext;
    public ulong           dstSet;
    public uint            dstBinding;
    public uint            dstArrayElement;
    public uint            descriptorCount;
    public uint            descriptorType;
    public nint            pImageInfo;
    public nint            pBufferInfo;       // const VkDescriptorBufferInfo*
    public nint            pTexelBufferView;
}
```

`VkDescriptorType` values used: `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7` (everything tensor-shaped), `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER = 6` (small constants). Push constants (next section) replace most uniform-buffer use.

### Descriptor manager strategy

Naive Vulkan tutorials allocate one descriptor pool per object, which exhausts pool memory quickly under inference load (thousands of dispatches per pipeline run). We use:

1. **Layout dedup** — one `VkDescriptorSetLayout` per *shape* of bindings (e.g. `(SSBO, SSBO, SSBO)` for `add(a,b,c)`, `(SSBO, SSBO, SSBO, SSBO)` for `linear(x,w,b,y)`). Maximum ~12 distinct layouts in the engine.
2. **Pre-allocated pool ring** — one `VkDescriptorPool` sized for ~4096 sets, reset every frame (or every pipeline phase).
3. **Push descriptors (`VK_KHR_push_descriptor`)** — when supported, write descriptors directly into the command buffer with `vkCmdPushDescriptorSetKHR`, avoiding sets entirely. Preferred path; falls back to traditional pool when extension absent.

Implementation lives in `VulkanDescriptorManager`. See [VULKAN_MEMORY_MANAGEMENT.md](VULKAN_MEMORY_MANAGEMENT.md) for sub-allocator details.

---

## Push Constants

```c
// In pipeline layout:
VkResult vkCreatePipelineLayout(VkDevice, const VkPipelineLayoutCreateInfo*, ...);

// At dispatch:
void vkCmdPushConstants(
    VkCommandBuffer commandBuffer,
    VkPipelineLayout layout,
    VkShaderStageFlags stageFlags,
    uint32_t offset,
    uint32_t size,
    const void* pValues);
```

```csharp
[StructLayout(LayoutKind.Sequential)]
public struct VkPushConstantRange
{
    public uint stageFlags;   // VK_SHADER_STAGE_COMPUTE_BIT = 0x20
    public uint offset;
    public uint size;         // ≤ maxPushConstantsSize (≥128 bytes guaranteed)
}

[StructLayout(LayoutKind.Sequential)]
public struct VkPipelineLayoutCreateInfo
{
    public VkStructureType sType;             // 30
    public nint            pNext;
    public uint            flags;
    public uint            setLayoutCount;
    public nint            pSetLayouts;       // const VkDescriptorSetLayout*
    public uint            pushConstantRangeCount;
    public nint            pPushConstantRanges;
}
```

**Use push constants for all small per-dispatch parameters** — tensor shapes (N, C, H, W), strides, scalars (eps, scale, alpha), flags. The 128-byte guarantee gives ~32 32-bit values. For a Conv2D dispatch a typical packing is:

```
struct Conv2DPushConstants {
    uint  N, C_in, H, W;
    uint  C_out, kH, kW;
    uint  strideH, strideW, padH, padW;
    uint  out_H, out_W;
    uint  flags;            // bit0 = has_bias, bit1 = activation_type, ...
};   // 14 * 4 = 56 bytes — well under 128
```

Pinned `stackalloc` followed by `vkCmdPushConstants` — zero allocation per dispatch.

---

## Command Pool & Command Buffer Recording

```c
VkResult vkCreateCommandPool(VkDevice, const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*, VkCommandPool*);
void     vkDestroyCommandPool(VkDevice, VkCommandPool, const VkAllocationCallbacks*);
VkResult vkResetCommandPool(VkDevice, VkCommandPool, VkCommandPoolResetFlags);

VkResult vkAllocateCommandBuffers(VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer*);
void     vkFreeCommandBuffers(VkDevice, VkCommandPool, uint32_t, const VkCommandBuffer*);

VkResult vkBeginCommandBuffer(VkCommandBuffer, const VkCommandBufferBeginInfo*);
VkResult vkEndCommandBuffer(VkCommandBuffer);
VkResult vkResetCommandBuffer(VkCommandBuffer, VkCommandBufferResetFlags);

void vkCmdBindPipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline);
void vkCmdBindDescriptorSets(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout,
                             uint32_t firstSet, uint32_t setCount, const VkDescriptorSet*,
                             uint32_t dynamicOffsetCount, const uint32_t* pDynamicOffsets);
void vkCmdDispatch(VkCommandBuffer, uint32_t groupX, uint32_t groupY, uint32_t groupZ);
void vkCmdDispatchIndirect(VkCommandBuffer, VkBuffer, VkDeviceSize offset);

void vkCmdCopyBuffer(VkCommandBuffer, VkBuffer src, VkBuffer dst, uint32_t regionCount, const VkBufferCopy*);
void vkCmdFillBuffer(VkCommandBuffer, VkBuffer, VkDeviceSize offset, VkDeviceSize size, uint32_t data);
void vkCmdUpdateBuffer(VkCommandBuffer, VkBuffer, VkDeviceSize offset, VkDeviceSize size, const void* pData);

void vkCmdPipelineBarrier2(VkCommandBuffer, const VkDependencyInfo*);   // sync2
```

```csharp
[StructLayout(LayoutKind.Sequential)]
public struct VkCommandPoolCreateInfo
{
    public VkStructureType sType;            // 39
    public nint            pNext;
    public uint            flags;            // RESET_COMMAND_BUFFER=0x2, TRANSIENT=0x1
    public uint            queueFamilyIndex;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkCommandBufferAllocateInfo
{
    public VkStructureType sType;           // 40
    public nint            pNext;
    public ulong           commandPool;     // VkCommandPool
    public uint            level;           // 0=PRIMARY, 1=SECONDARY
    public uint            commandBufferCount;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkCommandBufferBeginInfo
{
    public VkStructureType sType;           // 42
    public nint            pNext;
    public uint            flags;           // ONE_TIME_SUBMIT=0x1
    public nint            pInheritanceInfo;// null for primary
}
```

### Per-dispatch recording skeleton

```csharp
public unsafe void Dispatch(
    VulkanKernel kernel,
    ReadOnlySpan<VkBuffer> buffers,
    ReadOnlySpan<byte>     pushConstants,
    uint groupX, uint groupY, uint groupZ)
{
    var cb = _currentCommandBuffer;

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.Pipeline);

    // Push descriptors path (avoids descriptor pool entirely)
    Span<VkWriteDescriptorSet> writes = stackalloc VkWriteDescriptorSet[buffers.Length];
    Span<VkDescriptorBufferInfo> bufInfos = stackalloc VkDescriptorBufferInfo[buffers.Length];
    for (int i = 0; i < buffers.Length; i++)
    {
        bufInfos[i] = new VkDescriptorBufferInfo { buffer = buffers[i], offset = 0, range = VK_WHOLE_SIZE };
        writes[i]   = new VkWriteDescriptorSet
        {
            sType = VkStructureType.WriteDescriptorSet,
            dstBinding = (uint)i,
            descriptorCount = 1,
            descriptorType = (uint)VkDescriptorType.StorageBuffer,
            pBufferInfo = (nint)Unsafe.AsPointer(ref bufInfos[i])
        };
    }
    fixed (VkWriteDescriptorSet* pWrites = writes)
        vkCmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  kernel.Layout, 0, (uint)writes.Length, (nint)pWrites);

    // Push constants
    fixed (byte* p = pushConstants)
        vkCmdPushConstants(cb, kernel.Layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, (uint)pushConstants.Length, (nint)p);

    vkCmdDispatch(cb, groupX, groupY, groupZ);

    // Implicit barrier for next op
    EmitMemoryBarrierShaderToShader(cb);
}
```

---

## Pipeline Barriers (Synchronization-2)

Synchronization is the hardest part of Vulkan. CUDA hides it on a single stream — Vulkan does not. Use `VK_KHR_synchronization2` (Vulkan 1.3 core) — the v1 sync API is harder to reason about.

```c
typedef struct VkMemoryBarrier2 {
    VkStructureType         sType;
    const void*             pNext;
    VkPipelineStageFlags2   srcStageMask;
    VkAccessFlags2          srcAccessMask;
    VkPipelineStageFlags2   dstStageMask;
    VkAccessFlags2          dstAccessMask;
} VkMemoryBarrier2;

typedef struct VkBufferMemoryBarrier2 {
    VkStructureType         sType;
    const void*             pNext;
    VkPipelineStageFlags2   srcStageMask;
    VkAccessFlags2          srcAccessMask;
    VkPipelineStageFlags2   dstStageMask;
    VkAccessFlags2          dstAccessMask;
    uint32_t                srcQueueFamilyIndex;
    uint32_t                dstQueueFamilyIndex;
    VkBuffer                buffer;
    VkDeviceSize            offset;
    VkDeviceSize            size;
} VkBufferMemoryBarrier2;

typedef struct VkDependencyInfo {
    VkStructureType                 sType;
    const void*                     pNext;
    VkDependencyFlags               dependencyFlags;
    uint32_t                        memoryBarrierCount;
    const VkMemoryBarrier2*         pMemoryBarriers;
    uint32_t                        bufferMemoryBarrierCount;
    const VkBufferMemoryBarrier2*   pBufferMemoryBarriers;
    uint32_t                        imageMemoryBarrierCount;
    const VkImageMemoryBarrier2*    pImageMemoryBarriers;
} VkDependencyInfo;

void vkCmdPipelineBarrier2(VkCommandBuffer commandBuffer, const VkDependencyInfo* pDependencyInfo);
```

`VkPipelineStageFlags2` (`uint64_t`) — relevant bits:

| Bit | Name |
|---|---|
| `0x00000000ull` | `NONE` |
| `0x00000800ull` | `COMPUTE_SHADER` |
| `0x00001000ull` | `ALL_COMMANDS` (heavy) |
| `0x00010000ull` | `COPY` |
| `0x00100000ull` | `HOST` |

`VkAccessFlags2` (`uint64_t`):

| Bit | Name |
|---|---|
| `0x00000040ull` | `SHADER_READ` |
| `0x00000080ull` | `SHADER_WRITE` |
| `0x00000800ull` | `TRANSFER_READ` |
| `0x00001000ull` | `TRANSFER_WRITE` |
| `0x00004000ull` | `HOST_READ` |
| `0x00008000ull` | `HOST_WRITE` |
| `0x40000000_00000000ull` | `SHADER_STORAGE_READ` (separated in sync2) |
| `0x80000000_00000000ull` | `SHADER_STORAGE_WRITE` |

### Barrier patterns we need

```
A) Compute → Compute (same queue, RAW on a tensor) — the most common
   srcStage = COMPUTE_SHADER, srcAccess = SHADER_STORAGE_WRITE
   dstStage = COMPUTE_SHADER, dstAccess = SHADER_STORAGE_READ

B) Transfer → Compute (after H2D upload)
   srcStage = COPY, srcAccess = TRANSFER_WRITE
   dstStage = COMPUTE_SHADER, dstAccess = SHADER_STORAGE_READ

C) Compute → Transfer (before D2H readback)
   srcStage = COMPUTE_SHADER, srcAccess = SHADER_STORAGE_WRITE
   dstStage = COPY, dstAccess = TRANSFER_READ

D) Compute → Host (when CPU will read mapped memory after compute)
   srcStage = COMPUTE_SHADER, srcAccess = SHADER_STORAGE_WRITE
   dstStage = HOST, dstAccess = HOST_READ
```

**Engine convention:** emit a single `BUFFER_MEMORY_BARRIER2` after every `vkCmdDispatch` covering only the *output* buffer of that op (range = (offset, size) of the tensor's allocation). This is conservative but cheap; the alternative — tracking dependency DAGs across hundreds of ops — is fragile. Avoid `ALL_COMMANDS` / global memory barriers; they kill async-compute concurrency.

The CUDA backend's "lazy-sync activation cache" pattern (see [CUDA_PERFORMANCE.md](CUDA_PERFORMANCE.md)) maps directly: every cached activation has an associated `(srcStage, srcAccess)` set by the producing dispatch; the consumer emits the barrier on first read.

---

## Queue Submission & Fences

```c
VkResult vkCreateFence(VkDevice, const VkFenceCreateInfo*, const VkAllocationCallbacks*, VkFence*);
VkResult vkResetFences(VkDevice, uint32_t fenceCount, const VkFence*);
VkResult vkWaitForFences(VkDevice, uint32_t fenceCount, const VkFence*, VkBool32 waitAll, uint64_t timeout);
VkResult vkGetFenceStatus(VkDevice, VkFence);

VkResult vkQueueSubmit2(VkQueue queue, uint32_t submitCount, const VkSubmitInfo2*, VkFence fence);
VkResult vkQueueWaitIdle(VkQueue queue);
```

```csharp
[StructLayout(LayoutKind.Sequential)]
public struct VkCommandBufferSubmitInfo
{
    public VkStructureType sType;          // 1000314001
    public nint            pNext;
    public nint            commandBuffer;  // VkCommandBuffer
    public uint            deviceMask;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkSemaphoreSubmitInfo
{
    public VkStructureType sType;          // 1000314000
    public nint            pNext;
    public ulong           semaphore;
    public ulong           value;          // for timeline semaphores
    public ulong           stageMask;      // VkPipelineStageFlags2
    public uint            deviceIndex;
}

[StructLayout(LayoutKind.Sequential)]
public struct VkSubmitInfo2
{
    public VkStructureType sType;                              // 1000314002
    public nint            pNext;
    public uint            flags;
    public uint            waitSemaphoreInfoCount;
    public nint            pWaitSemaphoreInfos;
    public uint            commandBufferInfoCount;
    public nint            pCommandBufferInfos;
    public uint            signalSemaphoreInfoCount;
    public nint            pSignalSemaphoreInfos;
}
```

**Use timeline semaphores** (`VK_KHR_timeline_semaphore`, core in 1.2) instead of binary semaphores + fences. A timeline semaphore is a monotonic 64-bit counter; the host or device can wait until the value reaches N. This eliminates the manual fence pool that plagues v1-style Vulkan code and matches CUDA's "stream-event" mental model.

```csharp
VkResult vkSignalSemaphore(VkDevice, const VkSemaphoreSignalInfo*);
VkResult vkWaitSemaphores(VkDevice, const VkSemaphoreWaitInfo*, uint64_t timeout);
VkResult vkGetSemaphoreCounterValue(VkDevice, VkSemaphore, uint64_t*);
```

### Single-stream submission model

For SD1.5 / SDXL we run everything on **one queue, one timeline semaphore**:

```
counter = 0
For each op i in [0..N):
    record cmd_i into command buffer
    barriers between cmd_{i-1} and cmd_i
    submit { wait: counter==i, signal: counter:=i+1 }
After step:
    wait for counter == N (host-side)
```

Multi-queue / async-compute is a Phase 7 optimization. The single-stream path matches CUDA's lazy-sync activation cache and lets us reuse all of the existing scheduler / cache logic.

---

## Buffer Copy & Staging

```c
void vkCmdCopyBuffer(VkCommandBuffer, VkBuffer src, VkBuffer dst, uint32_t regionCount, const VkBufferCopy*);
```

```csharp
[StructLayout(LayoutKind.Sequential)]
public struct VkBufferCopy
{
    public ulong srcOffset;
    public ulong dstOffset;
    public ulong size;
}
```

`vkCmdCopyBuffer` runs on the queue's transfer or compute path. Patterns:

- **Weight upload (one-shot, large):** allocate `HOST_VISIBLE | HOST_COHERENT` staging buffer (or chunked sequence of them up to free RAM), `memcpy` mmapped safetensors slice into it, `vkCmdCopyBuffer` to `DEVICE_LOCAL`, fence-wait, free staging. We do this in `PreloadWeights()` exactly like CUDA's `GpuTransferHelper.PreloadWeight`.
- **Activation H2D / D2H:** rare in SharpInference because activations stay GPU-resident. Only at pipeline boundaries (CLIP→UNet input, VAE→PNG output).
- **ReBAR fast path:** if a `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` memory type exists and the upload fits, write directly — no staging buffer, no copy command.

`vkCmdUpdateBuffer` is convenient for *very* small (< 64 KB, 4-byte aligned) updates — driver pulls data through the command stream — but is bandwidth-limited; do **not** use for tensors.

---

## Function Pointers (Extension Loading)

Core 1.3 functions are exported by the loader; extension functions (`vkCmdPushDescriptorSetKHR`, etc.) must be queried via `vkGetDeviceProcAddr`:

```c
PFN_vkVoidFunction vkGetDeviceProcAddr(VkDevice device, const char* pName);
```

```csharp
[LibraryImport("vulkan-1", EntryPoint = "vkGetDeviceProcAddr")]
public static partial nint vkGetDeviceProcAddr(nint device, [MarshalAs(UnmanagedType.LPUTF8Str)] string pName);

// Example: load and cache a delegate
private delegate* unmanaged[Cdecl]<nint, uint, ulong, uint, uint, nint, void> _pfnPushDescriptorSet;

internal void LoadExtensions(nint device)
{
    nint p = vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR");
    if (p != 0) _pfnPushDescriptorSet =
        (delegate* unmanaged[Cdecl]<nint, uint, ulong, uint, uint, nint, void>)p;
}
```

Use unmanaged function-pointer types (`delegate* unmanaged[Cdecl]<...>`) — zero P/Invoke marshalling overhead, mirrors the `nint`-as-function-handle pattern used in `CudaModule`.

---

## Comparison: Vulkan vs CUDA Driver API

| Concept | CUDA | Vulkan |
|---|---|---|
| Context | `CUcontext` (per thread / device) | `VkInstance` + `VkDevice` |
| Stream | `CUstream` | `VkQueue` + `VkCommandBuffer` recorded → `vkQueueSubmit` |
| Module | `CUmodule` (loaded PTX) | `VkShaderModule` (SPIR-V) |
| Kernel function | `CUfunction` (handle) | `VkPipeline` (pre-built compute pipeline) |
| Launch | `cuLaunchKernel(func, grid, block, sharedMem, stream, args, ...)` | `vkCmdBindPipeline + vkCmdPushConstants + vkCmdDispatch` |
| Workgroup vs thread | grid × block (runtime params) | workgroup count × `local_size_*` (compile-time in shader) |
| Constants per launch | kernel params (pointer array) | push constants (≤ 128 B) + spec constants (compile-time) |
| Memory alloc | `cuMemAlloc` / `cuMemAllocAsync` | `vkAllocateMemory` (4096 limit — sub-allocate) + `vkCreateBuffer` + `vkBindBufferMemory` |
| Memcpy | `cuMemcpyHtoD` / `DtoH` (sync) | `vkCmdCopyBuffer` (recorded) + barrier + queue submit |
| Sync | `cuStreamSynchronize` | Fence / timeline semaphore wait |
| Cross-op sync | implicit on single stream | explicit pipeline barrier per op |
| FP16 GEMM library | cuBLAS HGEMM | None — hand-written tiled GEMM compute shader |
| Tensor Cores / matrix accel | `wmma` PTX or cuBLASLt | `VK_KHR_cooperative_matrix` (where supported) |
| JIT compile | `cuModuleLoadData` ~50–500 ms | `vkCreateComputePipelines` ~5–500 ms (cache via `VkPipelineCache`) |
| Dynamic shared memory | `cuFuncSetAttribute(MAX_DYNAMIC_SHARED)` | GLSL `shared float scratch[N]` (compile-time) — for variable, two-pass spec-constant rebuild |
| Error handling | `CUresult` return | `VkResult` return (negative = error) |
| Async free | `cuMemFreeAsync` | None — must wait for fence then `vkFreeMemory` (or recycle from sub-allocator) |
| Pinned host memory | `cuMemHostAlloc` | `HOST_VISIBLE` memory type |
| GPU pointer arithmetic | trivial — pointers are 64-bit GPU virtual addr | requires `VK_KHR_buffer_device_address` for similar UX |
| Profile | `cuEventElapsedTime` | `vkCmdWriteTimestamp` + `vkGetQueryPoolResults` |

---

## P/Invoke Function List (Phase 3.5 minimum surface)

These ~55 functions cover the entire Vulkan compute backend. Mirror the layout of [CudaDriverApi.cs](../../src/SharpInference.Cuda/CudaDriverApi.cs) — flat static class, `[LibraryImport]`, no marshalling.

**Instance / device:**
`vkCreateInstance`, `vkDestroyInstance`,
`vkEnumeratePhysicalDevices`, `vkGetPhysicalDeviceProperties`, `vkGetPhysicalDeviceProperties2`,
`vkGetPhysicalDeviceFeatures`, `vkGetPhysicalDeviceFeatures2`,
`vkGetPhysicalDeviceMemoryProperties`, `vkGetPhysicalDeviceMemoryProperties2`,
`vkGetPhysicalDeviceQueueFamilyProperties`,
`vkCreateDevice`, `vkDestroyDevice`, `vkDeviceWaitIdle`,
`vkGetDeviceQueue`,
`vkGetInstanceProcAddr`, `vkGetDeviceProcAddr`.

**Buffers / memory:**
`vkCreateBuffer`, `vkDestroyBuffer`,
`vkGetBufferMemoryRequirements`, `vkGetBufferMemoryRequirements2`,
`vkAllocateMemory`, `vkFreeMemory`,
`vkBindBufferMemory`,
`vkMapMemory`, `vkUnmapMemory`,
`vkFlushMappedMemoryRanges`, `vkInvalidateMappedMemoryRanges`.

**Shader / pipeline:**
`vkCreateShaderModule`, `vkDestroyShaderModule`,
`vkCreatePipelineLayout`, `vkDestroyPipelineLayout`,
`vkCreatePipelineCache`, `vkDestroyPipelineCache`, `vkGetPipelineCacheData`, `vkMergePipelineCaches`,
`vkCreateComputePipelines`, `vkDestroyPipeline`.

**Descriptors:**
`vkCreateDescriptorSetLayout`, `vkDestroyDescriptorSetLayout`,
`vkCreateDescriptorPool`, `vkDestroyDescriptorPool`, `vkResetDescriptorPool`,
`vkAllocateDescriptorSets`, `vkUpdateDescriptorSets`.

**Commands:**
`vkCreateCommandPool`, `vkDestroyCommandPool`, `vkResetCommandPool`,
`vkAllocateCommandBuffers`, `vkFreeCommandBuffers`,
`vkBeginCommandBuffer`, `vkEndCommandBuffer`, `vkResetCommandBuffer`,
`vkCmdBindPipeline`, `vkCmdBindDescriptorSets`,
`vkCmdPushConstants`, `vkCmdDispatch`,
`vkCmdCopyBuffer`, `vkCmdFillBuffer`,
`vkCmdPipelineBarrier2`, `vkCmdWriteTimestamp2`.

**Sync:**
`vkCreateFence`, `vkDestroyFence`, `vkResetFences`, `vkWaitForFences`, `vkGetFenceStatus`,
`vkCreateSemaphore`, `vkDestroySemaphore`,
`vkSignalSemaphore`, `vkWaitSemaphores`, `vkGetSemaphoreCounterValue`,
`vkQueueSubmit2`, `vkQueueWaitIdle`.

**Optional / extension (pulled via `vkGetDeviceProcAddr`):**
`vkCmdPushDescriptorSetKHR`, `vkGetMemoryFdKHR` (interop), `vkGetPhysicalDeviceMemoryProperties2`.

---

## Vendor & Driver Compatibility Matrix

Tested on Linux unless noted.

| Vendor | Driver | Vulkan | FP16 (`shaderFloat16`) | Subgroup size | Cooperative Matrix | Notes |
|---|---|---|---|---|---|---|
| NVIDIA RTX 3060 / 3090 / 4090 | 535+ | 1.3 | Yes | 32 | `VK_NV_cooperative_matrix` (Turing+) / `VK_KHR_cooperative_matrix` (driver 550+) | Same chip as our CUDA target — useful A/B reference |
| AMD RX 6700 / 7900 (RDNA2/3) | Mesa RADV 23+ | 1.3 | Yes | 32 (wave32) or 64 (wave64) | `VK_KHR_cooperative_matrix` (RDNA3 + Mesa 24+) | Most important AMD target; `RADV_PERFTEST=gpl` |
| AMD Vega / Polaris (GCN) | Mesa RADV 23+ | 1.3 | Yes | 64 | No | Older — focus on RDNA initially |
| Intel Arc A770 / A580 | ANV / Intel Iris | 1.3 | Yes | 8 / 16 / 32 (variable) | partial | Variable subgroup size — must use `subgroupSizeControl` |
| Intel UHD 630 / Xe (iGPU) | ANV | 1.3 | Yes | 8–32 | No | Mostly for fallback; small VRAM |
| ARM Mali-G715 | Proprietary | 1.3 | Limited | 16 | No | Mobile — not a Phase 3.5 target |
| Apple M-series | MoltenVK 1.2+ | 1.2 (some 1.3) | Yes | 32 | partial via Metal Performance Shaders | macOS only — out of scope until Phase 7+ |

**Phase 3.5 supported targets:** NVIDIA Pascal+, AMD GCN5+ / all RDNA, Intel Arc, Intel Xe (iGPU optional). Mali / Apple deferred.

---

## Profiling & Debugging

| Need | Tool | How |
|---|---|---|
| Validation | `VK_LAYER_KHRONOS_validation` | Set env `SHARPINFERENCE_VK_VALIDATION=1` at startup |
| GPU trace | RenderDoc (Linux/Windows) | `RENDERDOC_HOOK_EGL=0` env, attach via UI |
| Per-kernel timing | `VkQueryPool` of timestamps | `vkCmdWriteTimestamp2` before/after dispatch, multiply diff by `limits.timestampPeriod` ns |
| Memory leaks | Validation + `vkAllocateMemory` count | Fail at shutdown if any non-freed `VkDeviceMemory`/`VkBuffer` |
| SPIR-V disassembly | `spirv-dis input.spv` | Bundled with Vulkan SDK |
| Compile-time dumps | `glslangValidator -H` | Emits annotated SPIR-V text |
| RADV shader trace | `RADV_DEBUG=preoptir,nir`, `AMD_VULKAN_ICD=RADV` | Mesa AMD |
| NVIDIA shader trace | Nsight Compute / Nsight Graphics | `ncu --target-processes all` |
| Intel shader trace | `INTEL_DEBUG=spv,nir` | ANV |

---

## Open Questions

- [ ] Whether `VK_KHR_cooperative_matrix` (Tensor Cores / WMMA equivalent) is mature enough on Mesa RADV / NV proprietary to be worth a Phase 4+ optimization pass. Initial GEMM will use plain subgroup-tiled FMA.
- [ ] Whether to ship one SPIR-V binary per kernel (with spec constants) or pre-specialized variants (slightly faster JIT, larger NuGet). Current plan: one .spv + spec constants.
- [ ] Whether to require Vulkan 1.3 hard or fall back to 1.2 on older Intel iGPU drivers (gain is small; complexity is real).
- [ ] Multi-GPU strategy — Vulkan natively supports device groups; useful for future tensor-parallel work, not Phase 3.5.
- [ ] Whether to use `VK_KHR_buffer_device_address` to give shaders raw 64-bit pointers (closer to CUDA UX). Saves descriptor binding for scratch buffers; needs `bufferDeviceAddress` feature.

---

## Implementation Notes

1. **`[LibraryImport]` over `[DllImport]`** — source-gen marshalling, faster startup. Mirror dotLLM's pattern.
2. **Cross-platform DLL resolution** — `NativeLibrary.SetDllImportResolver` for `vulkan-1` → `libvulkan.so.1`/`libvulkan-1.dll`/`libvulkan.1.dylib` (or `libMoltenVK.dylib` on macOS).
3. **Strong-typed handles** — wrap dispatchable in `nint` and non-dispatchable in `ulong` consistently. Define a `VkBuffer { ulong Handle; }` struct only if the readability win is worth the boxing risk.
4. **Pin extension-name strings once** — build a `static readonly byte[] s_extNamesBlob` of UTF-8 + nulls and a `static readonly nint s_extNamePtrs[]`; reuse across multiple `vkCreateDevice` attempts.
5. **Fail fast on missing required features** — surface `UnsupportedDeviceException` listing exactly which feature failed (`shaderFloat16`, `subgroupSizeControl`, etc.). Users can downgrade or pick a different GPU.
6. **Persist pipeline cache** — `~/.cache/sharpinference/vulkan/<deviceUUID>.pipeline_cache`. Saves 0.5–2 s on cold start.
7. **One command pool per thread** — command pools are not thread-safe. We use one per backend instance; multithreaded recording is Phase 7+.
8. **Reset don't free** — `vkResetCommandPool` (cheap) instead of `vkFreeCommandBuffers` per frame.
9. **Timeline semaphores beat binary semaphores + fences** — single 64-bit counter per logical stream replaces a fence pool.
10. **Subgroup ops require `requiredSubgroupSize`** — without it, AMD might pick wave64 when our shader assumes 32. Always specialize `gl_SubgroupSize` via spec constant after pinning.
11. **Push constants ≤ 128 bytes** — enforce `Debug.Assert(pushSize <= 128)`. Anything bigger uses a uniform buffer at binding 0.
12. **Memory budget** — `VK_EXT_memory_budget` lets us read `VkPhysicalDeviceMemoryBudgetPropertiesEXT.heapBudget` for OOM-prediction; surface `OutOfVramException` like CUDA does.
13. **Descriptor pool sizing** — pre-allocate `4096 × max_bindings_per_set` storage-buffer descriptors per frame; reset between phases. Use push descriptors (`VK_KHR_push_descriptor`) when supported to skip the pool entirely.
14. **No graphics queue** — `VkInstance` does not need any window-system extensions. Headless throughout.

---

## References

- [Vulkan 1.3 Specification](https://docs.vulkan.org/spec/latest/index.html) — canonical
- [Khronos Vulkan Guide — Compute Shader](https://docs.vulkan.org/guide/latest/computeshader.html)
- [Khronos Vulkan Guide — Subgroups](https://docs.vulkan.org/guide/latest/subgroups.html)
- [Vulkan-Samples](https://github.com/KhronosGroup/Vulkan-Samples) — `samples/api/compute_op`
- [Sascha Willems Vulkan Samples](https://github.com/SaschaWillems/Vulkan) — `examples/computenbody`
- [VkFFT](https://github.com/DTolm/VkFFT) — production-grade Vulkan compute, written in C
- [llama.cpp Vulkan backend](https://github.com/ggerganov/llama.cpp/tree/master/ggml/src/ggml-vulkan) — most relevant reference, GEMM + GEMV in compute shaders, vendor-tested
- [Granite Vulkan Compute Examples](https://github.com/Themaister/Granite/tree/master/granite/compute) — modern Vulkan compute idioms
- [Mesa RADV source](https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/amd/vulkan) — official AMD open driver
- [Intel ANV source](https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/intel/vulkan)
- [VK_EXT_subgroup_size_control proposal](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_subgroup_size_control.html)
- [VK_KHR_cooperative_matrix proposal](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_cooperative_matrix.html)
- [VK_KHR_synchronization2 proposal](https://www.khronos.org/blog/vulkan-sdk-1.2.182-released-with-new-extensions-for-vulkan-synchronization-and-pipeline-management)
- [Vulkan-Headers (vulkan_core.h)](https://github.com/KhronosGroup/Vulkan-Headers/blob/main/include/vulkan/vulkan_core.h) — definitive struct layouts and enum values
- [Vulkan Memory Allocator (VMA)](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) — production C++ ref for sub-allocation strategy
