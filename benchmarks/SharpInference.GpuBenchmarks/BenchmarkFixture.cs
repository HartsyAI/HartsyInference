using System.Reflection;
using BenchmarkDotNet.Attributes;
using SharpInference.Core.Tensors;
using SharpInference.Cuda;

namespace SharpInference.GpuBenchmarks;

/// <summary>Shared fixture for all GPU benchmarks. Resolves the PTX directory relative to the
/// benchmark binary (the csproj copies it under <c>Ptx/</c>) and exposes a single
/// <see cref="CudaBackend"/> instance for the benchmark class. Throws a clear
/// <see cref="InvalidOperationException"/> when CUDA is unavailable so that BenchmarkDotNet's
/// validation step skips the whole class with an actionable message.</summary>
public sealed class BenchmarkFixture : IDisposable
{
    /// <summary>Backend instance shared across iterations of one benchmark class. Constructed on demand.</summary>
    public CudaBackend Backend { get; }

    /// <summary>Directory containing compiled PTX kernel files.</summary>
    public string PtxDir { get; }

    private int _disposed;

    /// <summary>Creates the backend on the user's primary CUDA device. Caller is responsible for
    /// running this on a real GPU machine — the benchmarks have no fallback path.</summary>
    public BenchmarkFixture()
    {
        if (!CudaContext.IsAvailable())
        {
            throw new InvalidOperationException(
                "CUDA is not available. SharpInference.GpuBenchmarks requires an NVIDIA GPU with the driver installed. " +
                "On a CPU-only machine, run SharpInference.Benchmarks instead.");
        }
        string assemblyDir = Path.GetDirectoryName(typeof(BenchmarkFixture).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            throw new DirectoryNotFoundException($"PTX directory not found at {ptxDir}. The csproj should copy `src/SharpInference.Cuda/Ptx/*.ptx` here on build.");
        }
        PtxDir = ptxDir;
        Backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
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
