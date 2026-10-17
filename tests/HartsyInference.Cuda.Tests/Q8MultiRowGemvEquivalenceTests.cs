using HartsyInference.Core.Configuration;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>The multi-row Q8 GEMV (<c>mul_mat_vec_q8_0_q8_1_mb</c> and <c>_ksplit_mb</c>) against one single-row launch per
/// activation row, bit for bit. Both paths are covered: K below the K-split floor takes the plain kernel, and K above it with
/// four warps per row takes the K-split kernel. M runs from 1 to 5, so M=3 and M=5 leave a partial last group of Q8_MB=4 rows.</summary>
[Collection("CudaSerial")]
public sealed unsafe class Q8MultiRowGemvEquivalenceTests
{
    private const int Q8Block = 32;
    private const int Q8BlockBytes = 34;

    private readonly ITestOutputHelper _output;

    public Q8MultiRowGemvEquivalenceTests(ITestOutputHelper output) => _output = output;

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(2048, 3072, 0)]   // plain path: K below the K-split floor, knob 0 forces one warp per row
    [InlineData(4096, 8192, 4)]   // K-split path: four warps per row, K above the floor
    public void MultiRow_MatchesPerRowLaunches_BitForBit(int n, int k, int ksplit)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        KnobStore.Set(EngineKnobs.GemvKsplit, ksplit);
        try
        {
            using CudaBackend backend = new(0, ptxDir);
            CudaKernels kernels = backend.Kernels ?? throw new InvalidOperationException("CUDA kernels not loaded");
            for (int m = 1; m <= 5; m++)
            {
                foreach (bool withBias in new[] { false, true })
                {
                    int mismatches = CompareOne(kernels, n, k, m, withBias, out int first);
                    _output.WriteLine($"N={n} K={k} M={m} bias={withBias} ksplit={ksplit}: mismatches={mismatches}");
                    Assert.True(mismatches == 0,
                        $"multi-row GEMV differs from per-row launches at N={n} K={k} M={m} bias={withBias} ksplit={ksplit}; first mismatch at element {first}.");
                }
            }
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.GemvKsplit);
        }
    }

    /// <summary>Runs the multi-row launch and the per-row reference over the same random inputs and counts elements whose bits differ.</summary>
    private static int CompareOne(CudaKernels kernels, int n, int k, int m, bool withBias, out int first)
    {
        int nblk = k / Q8Block;
        Random rng = new(unchecked(n * 31 + k * 7 + m * 3 + (withBias ? 1 : 0)));
        List<ulong> allocations = new();
        try
        {
            byte[] weight = new byte[(long)n * nblk * Q8BlockBytes];
            for (long block = 0; block < (long)n * nblk; block++)
            {
                long at = block * Q8BlockBytes;
                ushort scaleBits = BitConverter.HalfToUInt16Bits((Half)(0.01f + (float)rng.NextDouble() * 0.09f));
                weight[at] = (byte)scaleBits;
                weight[at + 1] = (byte)(scaleBits >> 8);
                for (int i = 0; i < Q8Block; i++)
                {
                    weight[at + 2 + i] = unchecked((byte)(sbyte)rng.Next(-127, 128));
                }
            }
            byte[] activations = new byte[(long)m * k];
            for (int i = 0; i < activations.Length; i++)
            {
                activations[i] = unchecked((byte)(sbyte)rng.Next(-127, 128));
            }
            float[] activationScales = new float[m * nblk];
            for (int i = 0; i < activationScales.Length; i++)
            {
                activationScales[i] = 0.001f + (float)rng.NextDouble() * 0.05f;
            }
            float[] bias = new float[n];
            for (int i = 0; i < bias.Length; i++)
            {
                bias[i] = (float)(rng.NextDouble() - 0.5);
            }

            ulong dWeight = Upload(weight, allocations);
            ulong dActivations = Upload(activations, allocations);
            ulong dScales = Upload(activationScales, allocations);
            ulong dBias = withBias ? Upload(bias, allocations) : 0UL;
            ulong dMulti = Allocate((long)m * n * sizeof(float), allocations);
            ulong dPerRow = Allocate((long)m * n * sizeof(float), allocations);

            kernels.LaunchMulMatVecQ8_0Q8_1(dMulti, dActivations, dScales, dWeight, dBias, n, k, m, 0);
            kernels.LaunchMulMatVecQ8_0Q8_1PerRow(dPerRow, dActivations, dScales, dWeight, dBias, n, k, m, 0);

            float[] multi = Download(dMulti, (long)m * n);
            float[] perRow = Download(dPerRow, (long)m * n);
            int mismatches = 0;
            first = -1;
            for (int i = 0; i < multi.Length; i++)
            {
                if (BitConverter.SingleToInt32Bits(multi[i]) == BitConverter.SingleToInt32Bits(perRow[i]))
                {
                    continue;
                }
                if (first < 0)
                {
                    first = i;
                }
                mismatches++;
            }
            return mismatches;
        }
        finally
        {
            foreach (ulong pointer in allocations)
            {
                CudaDriverApi.cuMemFree(pointer);
            }
        }
    }

    private static ulong Allocate(long bytes, List<ulong> allocations)
    {
        CudaDriverApi.cuMemAlloc(out ulong pointer, (nuint)bytes).ThrowOnError();
        allocations.Add(pointer);
        return pointer;
    }

    private static ulong Upload<T>(T[] host, List<ulong> allocations) where T : unmanaged
    {
        long bytes = (long)host.Length * sizeof(T);
        ulong pointer = Allocate(bytes, allocations);
        fixed (T* source = host)
        {
            CudaDriverApi.cuMemcpyHtoD(pointer, (nint)source, (nuint)bytes).ThrowOnError();
        }
        return pointer;
    }

    private static float[] Download(ulong pointer, long count)
    {
        float[] host = new float[count];
        fixed (float* destination = host)
        {
            CudaDriverApi.cuMemcpyDtoH((nint)destination, pointer, (nuint)(count * sizeof(float))).ThrowOnError();
        }
        return host;
    }
}
