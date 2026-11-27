using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Layers;

/// <summary>The many-segment join on CUDA against the CPU, for a head-major token axis (the scatter path) and a leading token axis (the concat path).
/// Reported as skipped, never passed, when no CUDA device opens.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class KvSegmentStoreGpuTests
{
    private const int Heads = 2, Dim = 4;
    private static readonly int[] Lengths = [3, 1, 5, 2];

    private readonly ITestOutputHelper _output;

    public KvSegmentStoreGpuTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [CudaFact]
    public void AJoinOfFourHeadMajorSegmentsOnCudaMatchesTheCpu()
    {
        using CpuBackend cpu = new();
        using CudaBackend cuda = new(0, PtxDirectory());
        (float[] cpuKeys, float[] cpuValues) = Join(cpu, Lengths, headMajor: true);
        (float[] gpuKeys, float[] gpuValues) = Join(cuda, Lengths, headMajor: true);

        Assert.Equal(cpuKeys, gpuKeys);
        Assert.Equal(cpuValues, gpuValues);
    }

    [CudaFact]
    public void AJoinOfFourLeadingAxisSegmentsOnCudaMatchesTheCpu()
    {
        using CpuBackend cpu = new();
        using CudaBackend cuda = new(0, PtxDirectory());
        (float[] cpuKeys, float[] cpuValues) = Join(cpu, Lengths, headMajor: false);
        (float[] gpuKeys, float[] gpuValues) = Join(cuda, Lengths, headMajor: false);

        Assert.Equal(cpuKeys, gpuKeys);
        Assert.Equal(cpuValues, gpuValues);
    }

    [CudaFact]
    public void TimeOneBlockJoinOnCuda()
    {
        using CudaBackend cuda = new(0, PtxDirectory());
        using KvSegmentStore store = Store(cuda, Lengths, headMajor: true);
        Consume(store, Lengths.Length);   // warm-up: kernels load and the first copies allocate

        const int repeats = 20;
        Stopwatch watch = Stopwatch.StartNew();
        for (int i = 0; i < repeats; i++)
        {
            Consume(store, Lengths.Length);
        }
        watch.Stop();
        _output.WriteLine($"Join of {Lengths.Length} head-major segments, keys and values, on CUDA: {watch.Elapsed.TotalMilliseconds / repeats:F3} ms per join (mean of {repeats}).");
    }

    /// <summary>Builds the segments on <paramref name="backend"/>, joins them all, and reads the keys and values back.</summary>
    private static (float[] Keys, float[] Values) Join(IBackend backend, int[] lengths, bool headMajor)
    {
        using KvSegmentStore store = Store(backend, lengths, headMajor);
        (Tensor keys, Tensor values) = store.Join(0, 0, lengths.Length);
        try
        {
            return (keys.AsSpan<float>().ToArray(), values.AsSpan<float>().ToArray());
        }
        finally
        {
            keys.Dispose();
            values.Dispose();
        }
    }

    /// <summary>A store of one layer holding one segment per length, each filled with distinct values.</summary>
    private static KvSegmentStore Store(IBackend backend, int[] lengths, bool headMajor)
    {
        KvSegmentStore store = new(backend, layers: 1, tokenAxis: headMajor ? 2 : 0);
        for (int s = 0; s < lengths.Length; s++)
        {
            long[] shape = headMajor ? new long[] { 1, Heads, lengths[s], Dim } : new long[] { lengths[s], Heads * Dim };
            store.Append(0, Filled(s + 1, shape), Filled(-(s + 1), shape));
        }
        return store;
    }

    /// <summary>Joins every segment and reads the result back, so the timed region includes the device work.</summary>
    private static void Consume(KvSegmentStore store, int count)
    {
        (Tensor keys, Tensor values) = store.Join(0, 0, count);
        try
        {
            _ = keys.AsSpan<float>()[0];
            _ = values.AsSpan<float>()[0];
        }
        finally
        {
            keys.Dispose();
            values.Dispose();
        }
    }

    private static Tensor Filled(float value, long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++)
        {
            s[i] = value + i * 0.001f;
        }
        return t;
    }

    private static string PtxDirectory()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate))
            {
                return candidate;
            }
        }
        throw new DirectoryNotFoundException("Could not find src/HartsyInference.Cuda/Ptx above the test binary.");
    }

    /// <summary>A fact reported as skipped, never passed, when no CUDA device opens.</summary>
    private sealed class CudaFactAttribute : FactAttribute
    {
        public CudaFactAttribute()
        {
            if (!CudaContext.IsAvailable()) Skip = "SKIPPED: CUDA unavailable.";
        }
    }
}
