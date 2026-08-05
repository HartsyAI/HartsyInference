using System.Diagnostics;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Measures how close the fp8 Linear path gets to the GPU's tensor-core ceiling on MiniMax-H3's four
/// real GEMM shapes, and splits the per-Linear cost into GEMM vs activation-quantization.
///
/// <para>Not a pass/fail correctness test — it prints a table and asserts only that the path ran. It exists
/// because the whole H3 performance plan hangs on one number: if cuBLASLt is already at the hardware peak,
/// no kernel rewrite can help and the remaining win has to come from removing work, not doing it faster.</para>
///
/// <para>Feeding an already-fp8 activation takes the <c>input.DType.IsFp8</c> branch, which skips the absmax +
/// quantize launches entirely — so (F32-activation − fp8-activation) is the quantization tax, measured rather
/// than estimated.</para>
///
/// <para>Run: <c>dotnet test tests/HartsyInference.Cuda.Tests -f net10.0 --filter
/// FullyQualifiedName~MiniMaxH3GemmFloor</c> (needs a free GPU — see benchmarks/minimax_h3/h3_bench.sh for
/// why a second CUDA tenant invalidates the numbers).</para></summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed class MiniMaxH3GemmFloorTests
{
    private readonly ITestOutputHelper _output;
    public MiniMaxH3GemmFloorTests(ITestOutputHelper output) => _output = output;

    private const int Blocks = 50;
    private const int SeqLen = 6550;

    /// <summary>(label, outDim, inDim) for the four fp8 Linears in every H3 block.</summary>
    private static readonly (string Name, int N, int K)[] Shapes =
    [
        ("qkv_proj", 21504, 5376),
        ("out_proj", 5376, 7168),
        ("mlp.fc1", 28672, 5376),
        ("mlp.fc2", 5376, 14336),
    ];

    [Fact]
    public unsafe void Fp8LinearFloor_OnH3Shapes()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        using CudaBackend backend = new CudaBackend(0, ptxDir);
        if (!backend.Fp8Executor.IsSupported)
        {
            _output.WriteLine("SKIPPED: native fp8 GEMM needs SM 8.9+.");
            return;
        }
        backend.EnableNativeFp8Gemm = true;

        _output.WriteLine($"M={SeqLen}, x{Blocks} blocks, fp8 e4m3 weights, F32 output");
        _output.WriteLine($"{"shape",-10} {"N",7} {"K",7} {"fp8act_ms",10} {"TFLOPS",8} {"f32act_ms",10} {"quant_ms",9}");

        double totalGemmMs = 0.0, totalFullMs = 0.0;
        foreach ((string name, int n, int k) in Shapes)
        {
            double gemmMs = TimeLinear(backend, n, k, fp8Activation: true, _output);
            double fullMs = TimeLinear(backend, n, k, fp8Activation: false, _output);
            double tflops = 2.0 * SeqLen * n * k / (gemmMs * 1e-3) / 1e12;
            totalGemmMs += gemmMs;
            totalFullMs += fullMs;
            _output.WriteLine($"{name,-10} {n,7} {k,7} {gemmMs,10:F3} {tflops,8:F1} {fullMs,10:F3} {fullMs - gemmMs,9:F3}");
        }

        _output.WriteLine("");
        _output.WriteLine($"x{Blocks} blocks: GEMM-only {totalGemmMs * Blocks / 1000.0:F3} s, "
            + $"with F32 activation quant {totalFullMs * Blocks / 1000.0:F3} s "
            + $"(quant tax {(totalFullMs - totalGemmMs) * Blocks / 1000.0:F3} s)");
        Assert.True(totalGemmMs > 0.0, "no timing recorded");
    }

    /// <summary>Median of several timed <see cref="CudaBackend.Linear"/> calls, in ms. Median rather than mean:
    /// a single clock-boost dip on an otherwise clean run skews a mean and there is no reason to let it.</summary>
    /// <remarks>Logs free VRAM around each measurement. This is also the regression net for the CacheActivation
    /// rebind leak it originally exposed: every call allocates a fresh device output buffer against the same
    /// output tensor, so "net leak" must stay at 0 — it read 5942 MB before that fix.</remarks>
    private static unsafe double TimeLinear(CudaBackend backend, int n, int k, bool fp8Activation,
        ITestOutputHelper log)
    {
        const int Warmup = 3;
        const int Iters = 9;

        long freeAtEntry = backend.FreeMemoryBytes();
        Tensor input = new Tensor(new TensorShape(SeqLen, k), fp8Activation ? DType.F8E4M3 : DType.F32);
        Tensor weight = new Tensor(new TensorShape(n, k), DType.F8E4M3);
        Tensor output = new Tensor(new TensorShape(SeqLen, n), DType.F32);
        try
        {
            Random rng = new Random(5);
            if (fp8Activation)
            {
                byte* ip = (byte*)input.DataPointer;
                for (long i = 0; i < (long)SeqLen * k; i++) ip[i] = (byte)(rng.Next(1, 0x40));
            }
            else
            {
                float* ip = (float*)input.DataPointer;
                for (long i = 0; i < (long)SeqLen * k; i++) ip[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
            byte* wp = (byte*)weight.DataPointer;
            for (long i = 0; i < (long)n * k; i++) wp[i] = (byte)(rng.Next(1, 0x40));

            // Warmup also uploads the weight, so the timed calls hit the resident cache like a real step does.
            for (int i = 0; i < Warmup; i++) backend.Linear(output, input, weight, bias: null);
            backend.Sync();

            double[] samples = new double[Iters];
            for (int i = 0; i < Iters; i++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                backend.Linear(output, input, weight, bias: null);
                backend.Sync();
                sw.Stop();
                samples[i] = sw.Elapsed.TotalMilliseconds;
            }
            Array.Sort(samples);
            double median = samples[Iters / 2];

            long freeBeforeEvict = backend.FreeMemoryBytes();
            input.Dispose(); weight.Dispose(); output.Dispose();
            backend.FreeAllDeviceMemory();
            long freeAfterEvict = backend.FreeMemoryBytes();
            log.WriteLine($"    [vram] N={n} K={k} fp8act={fp8Activation}: entry {freeAtEntry >> 20} MB"
                + $" -> before-evict {freeBeforeEvict >> 20} MB -> after-evict {freeAfterEvict >> 20} MB"
                + $" (reclaimed {(freeAfterEvict - freeBeforeEvict) >> 20} MB, net leak {(freeAtEntry - freeAfterEvict) >> 20} MB)");
            return median;
        }
        catch
        {
            input.Dispose(); weight.Dispose(); output.Dispose();
            throw;
        }
    }
}
