using System.Reflection;
using BenchmarkDotNet.Attributes;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;

namespace HartsyInference.GpuBenchmarks;

/// <summary>Which backend <see cref="BenchmarkFixture"/> constructed, selected via the
/// <c>HARTSYINFERENCE_BENCH_BACKEND</c> environment variable ("cuda" default, or "vulkan").</summary>
public enum BenchmarkBackendKind { Cuda, Vulkan }

/// <summary>Shared fixture for all GPU benchmarks. Resolves the PTX/SPIR-V directory relative to the
/// benchmark binary (the csproj copies both under <c>Ptx/</c> and <c>Spirv/</c>) and exposes a single
/// <see cref="IBackend"/> instance for the benchmark class, so the same benchmark classes run against
/// either backend unchanged — set <c>HARTSYINFERENCE_BENCH_BACKEND=vulkan</c> to target Vulkan (default
/// is CUDA, preserving existing behavior). Throws a clear <see cref="InvalidOperationException"/> when
/// the selected backend's runtime is unavailable so that BenchmarkDotNet's validation step skips the
/// whole class with an actionable message.</summary>
public sealed class BenchmarkFixture : IDisposable
{
    /// <summary>Backend instance shared across iterations of one benchmark class. Constructed on demand.</summary>
    public IBackend Backend { get; }

    /// <summary>Which backend this fixture constructed.</summary>
    public BenchmarkBackendKind Kind { get; }

    /// <summary>Directory containing compiled PTX kernel files (CUDA only; null under Vulkan).</summary>
    public string? PtxDir { get; }

    /// <summary><see cref="Backend"/> cast to <see cref="Cuda.CudaBackend"/>, for benchmarks that
    /// exercise CUDA-exclusive functionality with no Vulkan equivalent (the memory-pool API in
    /// <c>MemoryAllocFreeBenchmarks</c>, the <c>QuantizedMatMul</c> dequant-GEMM path in
    /// <c>QuantMatMulGpuBenchmarks</c>). Throws with an actionable message when this fixture was
    /// constructed against a different backend.</summary>
    public CudaBackend CudaBackend => Backend as CudaBackend
        ?? throw new InvalidOperationException(
            $"This benchmark requires the CUDA backend; current backend is {Kind} (set via HARTSYINFERENCE_BENCH_BACKEND). " +
            "Unset the variable, or set it to 'cuda', to run it.");

    private int _disposed;

    /// <summary>Creates the backend selected by <c>HARTSYINFERENCE_BENCH_BACKEND</c> ("cuda" default, or
    /// "vulkan") on device ordinal 0. Caller is responsible for running this on a real GPU machine — the
    /// benchmarks have no fallback path.</summary>
    public BenchmarkFixture()
    {
        string kindEnv = Environment.GetEnvironmentVariable("HARTSYINFERENCE_BENCH_BACKEND") ?? "cuda";
        Kind = kindEnv.Equals("vulkan", StringComparison.OrdinalIgnoreCase) ? BenchmarkBackendKind.Vulkan : BenchmarkBackendKind.Cuda;
        string assemblyDir = Path.GetDirectoryName(typeof(BenchmarkFixture).Assembly.Location)!;

        if (Kind == BenchmarkBackendKind.Cuda)
        {
            if (!CudaContext.IsAvailable())
            {
                throw new InvalidOperationException(
                    "CUDA is not available. HartsyInference.GpuBenchmarks requires an NVIDIA GPU with the driver installed. " +
                    "On a CPU-only machine, run HartsyInference.Benchmarks instead.");
            }
            string ptxDir = Path.Combine(assemblyDir, "Ptx");
            if (!Directory.Exists(ptxDir))
            {
                throw new DirectoryNotFoundException($"PTX directory not found at {ptxDir}. The csproj should copy `src/HartsyInference.Cuda/Ptx/*.ptx` here on build.");
            }
            PtxDir = ptxDir;
            Backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
        }
        else
        {
            string spvDir = Path.Combine(assemblyDir, "Spirv");
            if (!Directory.Exists(spvDir))
            {
                throw new DirectoryNotFoundException($"SPIR-V directory not found at {spvDir}. The csproj should copy `src/HartsyInference.Vulkan/Spirv/*.spv` here on build.");
            }
            Backend = new VulkanBackend(deviceOrdinal: 0, spvDir: spvDir);
        }
    }

    /// <summary>Allocates a tensor of the given shape and fills with deterministic-pseudo-random F32
    /// data so benchmark inputs aren't all-zeros (which can hit fast paths in some kernels).</summary>
    public static unsafe Tensor AllocateF32(TensorShape shape, int seed = 12345)
    {
        Tensor t = new(shape, DType.F32);
        long count = shape.ElementCount;
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < count; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return t;
    }

    /// <summary>Allocates an F16 tensor. Stored as <c>ushort</c> half-precision via
    /// <see cref="System.Runtime.InteropServices.MemoryMarshal"/> on the F32 source.</summary>
    public static unsafe Tensor AllocateF16(TensorShape shape, int seed = 12345)
    {
        Tensor f32 = AllocateF32(shape, seed);
        try
        {
            return f32.CastTo(DType.F16);
        }
        finally
        {
            f32.Dispose();
        }
    }

    /// <summary>Allocates an FP8 E4M3 tensor (engine's <c>FloatToFp8E4M3</c> rounding via
    /// <see cref="Tensor.CastTo"/>). Source values are in [-1, 1], comfortably inside E4M3 range.</summary>
    public static unsafe Tensor AllocateF8E4M3(TensorShape shape, int seed = 12345)
    {
        Tensor f32 = AllocateF32(shape, seed);
        try { return f32.CastTo(DType.F8E4M3); }
        finally { f32.Dispose(); }
    }

    /// <summary>Synchronizes the GPU stream — required between iterations to make BenchmarkDotNet's
    /// timing reflect the kernel time and not just the launch time.</summary>
    public void Sync() => Backend.Sync();

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            Backend.Dispose();
        }
    }
}
