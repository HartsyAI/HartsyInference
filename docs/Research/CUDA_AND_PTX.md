# CUDA Driver API and PTX reference

Bindings and launch implementations live in [HartsyInference.Cuda](../../src/HartsyInference.Cuda/).
This reference keeps ABI/algorithm traps; copied P/Invoke declarations, enum catalogs, speculative speedups,
and stale toolchain tables were removed. Query the actual device/driver and consult the upstream headers.

## Runtime and ownership

- CUDA Driver API uses per-thread current contexts. Establish the correct context before operations;
  backend isolation must include stream, caches, allocations, and disposal.
- CUdevice is an integer ordinal; handles are pointer-sized; CUdeviceptr is a 64-bit device address.
  Match the native ABI, not a historical wrapper's type aliases.
- Load shipped PTX from disk through CudaModule. Store resolved function handles; no embedded PTX or
  per-launch name lookup. Argument pointers must point to stable local variables on the stack.
- Stream-ordered allocation/free is asynchronous. Synchronize only where host access/lifetime demands it;
  an OOM retry must account for pending frees. See [engine rules](../Agents/AGENTS.md).
- A toolchain that emits newer PTX than the driver supports fails at JIT time. Inspect emitted .version
  and .target rather than equating SDK availability with runtime compatibility.
- cuBLAS is column-major: validate transpose/leading-dimension conventions against the row-major tensor
  layout. GEMM storage dtype and accumulation mode are separate decisions.

## PTX design constraints

- PTX is a virtual ISA; register allocation and final instructions are determined by the driver/ptxas.
  Inspect compiled resource use before declaring occupancy or throughput.
- Accumulate normalization, softmax, and sensitive reductions in F32 even with F16 storage.
  F16x2 packs two halves in one 32-bit register; explicitly convert to F32 for reductions.
- Warp shuffles require participating-lane masks and correct handling of partial warps.
  Cross-warp reductions need shared scratch plus synchronization, not another warp-local shuffle alone.
- Shared-memory bank mapping for 32-bit words is (byteAddress / 4) % 32. Padding a transpose tile can
  avoid bank conflicts; validate the actual layout, dtype, and access pattern.
- cp.async enables global-to-shared overlap on eligible targets; its waits/barriers are load-bearing.
  Tile sizes, pipeline stages, register blocking, and occupancy require measurement per shape/device.
- Use 64-bit address/index arithmetic where products can overflow. A 1024² spatial workload can exceed
  32-bit im2col indexing even when individual dimensions fit.
- Dynamic shared memory above the default limit needs a supported per-function opt-in; query limits.
  Do not copy hardware tables or assumed register budgets into dispatch code.
- For convolution, tile activation patches plus halos or use implicit GEMM; do not assume all models
  benefit from stride-1 Winograd. See [CONV2D_CUDA.md](CONV2D_CUDA.md).
- For attention, tile Q/K/V and maintain online softmax maxima, normalization sums, and output rescaling.
  Masking, GQA, softcap, and overflow behavior are part of correctness, not optional optimizations.
  See [FLASH_ATTENTION.md](FLASH_ATTENTION.md).

## Validation and profiling

Compare identical saved inputs against scalar/upstream references with the operation's documented tolerance.
Use CUDA events for GPU elapsed time and end-to-end wall time for user latency. Profile device transfers,
allocation, launch overhead, and kernel work separately. No generic handwritten-PTX speedup is guaranteed.
[Kernel instructions](../Agents/KERNEL.md) govern artifact rebuilds;
[PROFILING_METHODOLOGY.md](PROFILING_METHODOLOGY.md) governs measurements.

## References

- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)
- [cuBLAS](https://docs.nvidia.com/cuda/cublas/)
- [PTX ISA 9.2](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [managedCuda](https://github.com/kunzmi/managedCuda)
- [swigged.cuda](https://github.com/kaby76/swigged.cuda)
- [Device Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html)
- [Initialization](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html)
- [managedCuda DriverAPI.cs](https://github.com/kunzmi/managedCuda/blob/master/ManagedCUDA/DriverAPI.cs)
- [Context Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
- [Module Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)
- [Execution Control](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html)
- [Memory Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)
- [Stream Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)
- [Error Handling](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__ERROR.html)
- [cublas_api.h](https://gitlab.com/nvidia/headers/cuda-individual/cublas/-/blob/main/cublas_api.h)
- [swigged.cuda CUresult.cs](https://github.com/kaby76/swigged.cuda/blob/master/swigged.cuda/CUresult.cs)
- [CUDA Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [MS docs](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/tutorial-custom-marshaller)
- [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
- [NVIDIA Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)
- [NVIDIA Blog: Advanced CUDA Kernel Optimization (Handwritten PTX)](https://developer.nvidia.com/blog/advanced-nvidia-cuda-kernel-optimization-techniques-handwritten-ptx/)
- [NVIDIA Blog: Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
- [NVIDIA Blog: Understanding PTX](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/)
- [CUDA Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [Philip Fabianek: A Gentle Introduction to CUDA PTX](https://philipfabianek.com/posts/cuda-ptx-introduction/)
- [eunomia: CNN Convolution with Shared Memory](https://eunomia.dev/others/cuda-tutorial/06-cnn-convolution/)
- [Lei Mao: CUDA Shared Memory Bank Conflict-Free Access](https://leimao.github.io/blog/CUDA-Shared-Memory-Bank-Conflict-Free-Vectorized-Access/)
