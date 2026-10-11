using System.Diagnostics;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Effective DRAM bandwidth of the dp4a Q8_1 GEMV kernels that dominate LLM decode: the dense
/// <c>mul_mat_vec_*_q8_1</c> entries and the expert-indexed <c>moe_gateup_id_*</c> / <c>moe_down_id_*</c> ones, at real
/// model shapes (Mixtral, Qwen3-30B-A3B, OLMoE, Granite-3.0-3B-A800M, dense 7-8B). Raw device buffers filled with random
/// bytes (timing only; correctness lives in <see cref="CudaMoeIndexedTests"/> and <see cref="Dp4aGemvGroundTruthTests"/>),
/// and the routed expert set alternates between two disjoint id sets so no weight is L2-resident across iterations.
/// Bytes counted are the weight rows read once. Opt-in: <c>dotnet test --filter Category=GemvBench</c>; set
/// <c>GEMV_BENCH_PEAK_GBS</c> to the card's DRAM peak (default 360, the RTX 3060).</summary>
[Collection("CudaSerial")]
[Trait("Category", "GemvBench")]
public sealed unsafe class GemvBandwidthMicroBench
{
    private readonly ITestOutputHelper _output;
    public GemvBandwidthMicroBench(ITestOutputHelper output) => _output = output;

    private enum Kind { Dense, GateUp, Down }

    private sealed record Case(string Tag, Kind Kind, DType Format, int N, int K, int Experts, int TopK, int Tokens = 1);

    private static readonly Case[] Cases =
    {
        new("mixtral gate/up q4k", Kind.GateUp, DType.Q4_K, 14336, 4096, 4, 2),
        new("mixtral down q6k", Kind.Down, DType.Q6_K, 4096, 14336, 4, 2),
        new("mixtral down q4k", Kind.Down, DType.Q4_K, 4096, 14336, 4, 2),
        new("qwen3-30b gate/up q4k", Kind.GateUp, DType.Q4_K, 768, 2048, 128, 8),
        new("qwen3-30b down q6k", Kind.Down, DType.Q6_K, 2048, 768, 128, 8),
        new("qwen3-30b down q4k", Kind.Down, DType.Q4_K, 2048, 768, 128, 8),
        new("olmoe gate/up q4k", Kind.GateUp, DType.Q4_K, 1024, 2048, 64, 8),
        new("olmoe down q6k", Kind.Down, DType.Q6_K, 2048, 1024, 64, 8),
        new("granite-moe gate/up q4k", Kind.GateUp, DType.Q4_K, 512, 1536, 40, 8),
        new("granite-moe down q6k", Kind.Down, DType.Q6_K, 1536, 512, 40, 8),
        new("qwen3-30b gate/up q8_0", Kind.GateUp, DType.Q8_0, 768, 2048, 128, 8),
        new("qwen3-30b down q8_0", Kind.Down, DType.Q8_0, 2048, 768, 128, 8),
        new("dense q4k 4096x4096", Kind.Dense, DType.Q4_K, 4096, 4096, 2, 1),
        new("dense q4k 14336x4096", Kind.Dense, DType.Q4_K, 14336, 4096, 2, 1),
        new("dense q6k 4096x14336", Kind.Dense, DType.Q6_K, 4096, 14336, 2, 1),
        new("dense q6k 4096x4096", Kind.Dense, DType.Q6_K, 4096, 4096, 2, 1),
        new("dense q8_0 4096x4096", Kind.Dense, DType.Q8_0, 4096, 4096, 2, 1),
        new("dense q8_0 2048x8192", Kind.Dense, DType.Q8_0, 2048, 8192, 2, 1),
    };

    private static long RowBytes(DType f, int k) =>
        f == DType.Q4_K ? k / 256 * 144L : f == DType.Q6_K ? k / 256 * 210L : k / 32 * 34L;

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void GemvBandwidth()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        double peak = double.TryParse(Environment.GetEnvironmentVariable("GEMV_BENCH_PEAK_GBS"), out double pk) ? pk : 360.0;
        string? only = Environment.GetEnvironmentVariable("GEMV_BENCH_ONLY");

        if (int.TryParse(Environment.GetEnvironmentVariable("GEMV_BENCH_KSPLIT"), out int ks)) KnobStore.Set(EngineKnobs.GemvKsplit, ks);
        using CudaBackend cuda = new(0, ptxDir);
        CudaKernels k = cuda.Kernels ?? throw new InvalidOperationException("no kernels");
        _output.WriteLine($"{"shape",-26}{"us",9}{"MB",8}{"GB/s",8}{"%peak",7}");
        foreach (Case c in Cases)
        {
            if (only is not null && !c.Tag.Contains(only, StringComparison.Ordinal)) continue;
            (double us, double mb) = Run(k, c);
            double gbs = mb * 1e6 / (us * 1e-6) / 1e9;
            _output.WriteLine($"{c.Tag,-26}{us,9:F1}{mb,8:F1}{gbs,8:F0}{100 * gbs / peak,6:F0}%");
        }
        KnobStore.Clear(EngineKnobs.GemvKsplit);
    }

    private static (double us, double mb) Run(CudaKernels k, Case c)
    {
        long rowBytes = RowBytes(c.Format, c.K);
        long expertStride = rowBytes * c.N;
        int mats = c.Kind == Kind.GateUp ? 2 : 1;
        int rows = c.TopK * c.Tokens;
        int xRows = c.Kind == Kind.Down ? rows : c.Tokens;
        List<ulong> allocs = new();
        ulong Alloc(long bytes, bool random)
        {
            CudaDriverApi.cuMemAlloc(out ulong p, (nuint)bytes).ThrowOnError();
            allocs.Add(p);
            if (random)
            {
                byte[] host = new byte[Math.Min(bytes, 1 << 24)];
                new Random(17).NextBytes(host);
                for (int i = 1; i < host.Length; i += 2) host[i] &= 0x3B;   // keep fp16 fields finite and small-ish
                fixed (byte* h = host)
                    for (long off = 0; off < bytes; off += host.Length)
                        CudaDriverApi.cuMemcpyHtoD(p + (ulong)off, (nint)h, (nuint)Math.Min(host.Length, bytes - off)).ThrowOnError();
            }
            else CudaDriverApi.cuMemsetD8(p, 0, (nuint)bytes).ThrowOnError();
            return p;
        }
        try
        {
            ulong w0 = Alloc(expertStride * c.Experts, true);
            ulong w1 = mats == 2 ? Alloc(expertStride * c.Experts, true) : 0;
            ulong xq = Alloc((long)xRows * c.K, true);
            ulong xd = Alloc((long)xRows * (c.K / 32) * 4, false);
            ulong xs = Alloc((long)xRows * (c.K / 32) * 4, false);
            ulong outp = Alloc((long)rows * c.N * 4, false);
            // Two disjoint id sets (first and second half of the experts), alternated per launch.
            int half = c.Experts / 2;
            int[] idsA = new int[rows], idsB = new int[rows];
            for (int i = 0; i < rows; i++) { idsA[i] = i % half; idsB[i] = half + i % half; }
            ulong idA = Alloc(rows * 4, false), idB = Alloc(rows * 4, false);
            fixed (int* a = idsA) CudaDriverApi.cuMemcpyHtoD(idA, (nint)a, (nuint)(rows * 4)).ThrowOnError();
            fixed (int* b = idsB) CudaDriverApi.cuMemcpyHtoD(idB, (nint)b, (nuint)(rows * 4)).ThrowOnError();

            void Launch(int it)
            {
                bool odd = (it & 1) != 0;
                if (c.Kind == Kind.Dense)
                {
                    ulong w = w0 + (odd ? (ulong)expertStride : 0);
                    if (c.Format == DType.Q4_K) k.LaunchMulMatVecQ4KQ8_1(outp, xq, xd, xs, w, 0, c.N, c.K, 1, 0);
                    else if (c.Format == DType.Q6_K) k.LaunchMulMatVecQ6KQ8_1(outp, xq, xd, w, 0, c.N, c.K, 1, 0);
                    else k.LaunchMulMatVecQ8_0Q8_1(outp, xq, xd, w, 0, c.N, c.K, 1, 0);
                }
                else if (c.Kind == Kind.GateUp)
                    k.LaunchMoeGateUpId(c.Format, outp, xq, xd, xs, w0, w1, odd ? idB : idA, expertStride, c.N, c.K, c.TopK, rows, c.Experts, false, 0);
                else
                    k.LaunchMoeDownId(c.Format, outp, xq, xd, xs, w0, odd ? idB : idA, expertStride, c.N, c.K, rows, c.Experts, 0);
            }

            const int warmup = 20;
            int distinct = c.Kind == Kind.Dense ? 1 : Math.Min(rows, half);
            double mb = (double)distinct * mats * expertStride / 1e6;
            int iters = (int)Math.Clamp(4000.0 / mb, 50, 4000);
            for (int i = 0; i < warmup; i++) Launch(i);
            CudaDriverApi.cuStreamSynchronize(0).ThrowOnError();
            double best = double.MaxValue;
            for (int rep = 0; rep < 3; rep++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                for (int i = 0; i < iters; i++) Launch(i);
                CudaDriverApi.cuStreamSynchronize(0).ThrowOnError();
                best = Math.Min(best, sw.Elapsed.TotalMicroseconds / iters);
            }
            return (best, mb);
        }
        finally
        {
            foreach (ulong p in allocs) CudaDriverApi.cuMemFree(p);
        }
    }
}
