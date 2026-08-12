using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>
/// Explicit-only microbenchmark for the Qwen3 image-text-encoder attention choice. It compares the
/// production causal path (two GQA expansion kernels plus masked materialized SDPA), native-GQA
/// FlashAttention, and the same expanded path with the opt-in F16-I/O cuDNN engine.
/// </summary>
/// <remarks>
/// Run with <c>HARTSY_LLAMA_ATTN_BENCH=1 dotnet test --filter
/// FullyQualifiedName~LlamaGqaAttentionMicroBench</c>. Each route is warmed before nine batched trials;
/// medians therefore exclude PTX loading, allocation-pool growth, and cuDNN graph compilation. Production
/// dispatch is intentionally not changed by this benchmark.
/// </remarks>
[Collection("CudaSerial")]
[Trait("Category", "LlamaAttentionBench")]
public sealed unsafe class LlamaGqaAttentionMicroBench
{
    private const int Batch = 1;
    private const int QueryHeads = 32;
    private const int KvHeads = 8;
    private const int HeadDim = 128;
    private const int KvGroup = QueryHeads / KvHeads;
    private const int Trials = 9;

    private readonly ITestOutputHelper _output;

    public LlamaGqaAttentionMicroBench(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    [Theory]
    [InlineData(16)]
    [InlineData(64)]
    [InlineData(128)]
    [InlineData(256)]
    [InlineData(512)]
    public void Qwen3CausalGqa_Crossover(int sequence)
    {
        if (Environment.GetEnvironmentVariable("HARTSY_LLAMA_ATTN_BENCH") != "1")
        {
            _output.WriteLine("SKIPPED: set HARTSY_LLAMA_ATTN_BENCH=1 to run this explicit microbenchmark");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        string? previousCudnn = Environment.GetEnvironmentVariable("HARTSY_SDPA_CUDNN");
        string? previousForceF16 = Environment.GetEnvironmentVariable("HARTSY_SDPA_F16");
        string? previousNoF16 = Environment.GetEnvironmentVariable("HARTSY_SDPA_NO_F16");
        string? previousSage = Environment.GetEnvironmentVariable("HARTSY_SAGE_ATTN");
        string? previousForceFlash = Environment.GetEnvironmentVariable("HARTSY_SDPA_FORCE_FLASH");
        string? previousForceTiled = Environment.GetEnvironmentVariable("HARTSY_SDPA_FORCE_TILED");
        string? previousV2 = Environment.GetEnvironmentVariable("HARTSY_SDPA_V2");
        try
        {
            Environment.SetEnvironmentVariable("HARTSY_SDPA_CUDNN", "1");
            Environment.SetEnvironmentVariable("HARTSY_SDPA_F16", null);
            Environment.SetEnvironmentVariable("HARTSY_SDPA_NO_F16", null);
            Environment.SetEnvironmentVariable("HARTSY_SAGE_ATTN", "0");
            Environment.SetEnvironmentVariable("HARTSY_SDPA_FORCE_FLASH", null);
            Environment.SetEnvironmentVariable("HARTSY_SDPA_FORCE_TILED", null);
            Environment.SetEnvironmentVariable("HARTSY_SDPA_V2", null);

            RunShape(sequence);
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_SDPA_CUDNN", previousCudnn);
            Environment.SetEnvironmentVariable("HARTSY_SDPA_F16", previousForceF16);
            Environment.SetEnvironmentVariable("HARTSY_SDPA_NO_F16", previousNoF16);
            Environment.SetEnvironmentVariable("HARTSY_SAGE_ATTN", previousSage);
            Environment.SetEnvironmentVariable("HARTSY_SDPA_FORCE_FLASH", previousForceFlash);
            Environment.SetEnvironmentVariable("HARTSY_SDPA_FORCE_TILED", previousForceTiled);
            Environment.SetEnvironmentVariable("HARTSY_SDPA_V2", previousV2);
        }
    }

    private void RunShape(int sequence)
    {
        TensorShape qShape = new(Batch, QueryHeads, sequence, HeadDim);
        TensorShape kvShape = new(Batch, KvHeads, sequence, HeadDim);
        float scale = 1f / MathF.Sqrt(HeadDim);

        using Tensor query = RandomF32(qShape, seed: 1000 + sequence);
        using Tensor key = RandomF32(kvShape, seed: 2000 + sequence);
        using Tensor value = RandomF32(kvShape, seed: 3000 + sequence);
        using Tensor maskHost = CausalMask(sequence);
        using Tensor maskDevice = new(maskHost.Shape, DType.F32);
        using Tensor keyRepeated = new(qShape, DType.F32);
        using Tensor valueRepeated = new(qShape, DType.F32);
        using Tensor outputCurrent = new(qShape, DType.F32);
        using Tensor outputFlash = new(qShape, DType.F32);
        using Tensor outputCudnn = new(qShape, DType.F32);
        using CudaBackend backend = new(0, PtxDir());
        IBackend ops = backend;

        // Make the mask resident exactly as LlamaStyleEncoder does. Q/K/V become resident in the first warmup.
        ops.SliceRows(maskDevice, maskHost, 0);
        backend.Sync();

        void Current()
        {
            ops.RepeatKvHeads(keyRepeated, key, KvHeads, KvGroup);
            ops.RepeatKvHeads(valueRepeated, value, KvHeads, KvGroup);
            ops.ScaledDotProductAttention(
                outputCurrent, query, keyRepeated, valueRepeated, maskDevice, scale, allowF16: false);
        }

        void NativeFlash() => ops.FlashAttention(
            outputFlash, query, key, value, sequence, KvGroup, causal: true, qOffset: 0, scale);

        void Cudnn()
        {
            ops.RepeatKvHeads(keyRepeated, key, KvHeads, KvGroup);
            ops.RepeatKvHeads(valueRepeated, value, KvHeads, KvGroup);
            ops.ScaledDotProductAttention(
                outputCudnn, query, keyRepeated, valueRepeated, maskDevice, scale, allowF16: true);
        }

        // Warm in production-first order. The cuDNN warmup includes graph construction but timings do not.
        for (int i = 0; i < 5; i++) Current();
        backend.Sync();
        for (int i = 0; i < 5; i++) NativeFlash();
        backend.Sync();
        long cudnnBefore = backend.CudnnSdpaExecutionCount;
        for (int i = 0; i < 5; i++) Cudnn();
        backend.Sync();
        bool cudnnEngaged = backend.CudnnSdpaExecutionCount - cudnnBefore == 5;

        int currentIterations = CalibratedIterations(Current, backend);
        int flashIterations = CalibratedIterations(NativeFlash, backend);
        int cudnnIterations = cudnnEngaged ? CalibratedIterations(Cudnn, backend) : 1;

        double[] currentMs = Measure(Current, backend, currentIterations);
        double[] flashMs = Measure(NativeFlash, backend, flashIterations);
        double[] cudnnMs = cudnnEngaged
            ? Measure(Cudnn, backend, cudnnIterations)
            : Enumerable.Repeat(double.NaN, Trials).ToArray();

        // Refresh each output once after timing, then compare the exact buffers whose route was measured.
        Current();
        NativeFlash();
        if (cudnnEngaged) Cudnn();
        backend.Sync();
        ErrorMetrics flashError = Difference(outputCurrent, outputFlash);
        ErrorMetrics cudnnError = cudnnEngaged
            ? Difference(outputCurrent, outputCudnn)
            : new ErrorMetrics(double.NaN, double.NaN, double.NaN);

        double currentMedian = Median(currentMs);
        double flashMedian = Median(flashMs);
        double cudnnMedian = Median(cudnnMs);
        double repeatedMiB = 2d * QueryHeads * sequence * HeadDim * sizeof(float) / (1024 * 1024);
        double scoreMiB = (double)Batch * QueryHeads * sequence * sequence * sizeof(float) / (1024 * 1024);
        double maskMiB = (double)sequence * sequence * sizeof(float) / (1024 * 1024);

        _output.WriteLine(
            $"S={sequence}: current Repeat+masked-SDPA {currentMedian:F4} ms " +
            $"(n={currentIterations}, rounds=[{Format(currentMs)}])");
        _output.WriteLine(
            $"S={sequence}: native-GQA Flash       {flashMedian:F4} ms " +
            $"(n={flashIterations}, rounds=[{Format(flashMs)}]), current/flash={currentMedian / flashMedian:F3}x");
        _output.WriteLine(cudnnEngaged
            ? $"S={sequence}: Repeat+masked-cuDNN   {cudnnMedian:F4} ms " +
              $"(n={cudnnIterations}, rounds=[{Format(cudnnMs)}]), current/cuDNN={currentMedian / cudnnMedian:F3}x"
            : $"S={sequence}: Repeat+masked-cuDNN   unavailable ({CudnnRuntime.Reason})");
        _output.WriteLine(
            $"S={sequence}: Flash-current max={flashError.MaxAbs:E4}, RMSE={flashError.Rmse:E4}, meanAbs={flashError.MeanAbs:E4}; " +
            $"cuDNN-current max={cudnnError.MaxAbs:E4}, RMSE={cudnnError.Rmse:E4}, meanAbs={cudnnError.MeanAbs:E4}");
        _output.WriteLine(
            $"S={sequence}: avoidable global intermediates: repeated K+V={repeatedMiB:F3} MiB, " +
            $"materialized scores={scoreMiB:F3} MiB, resident causal mask={maskMiB:F3} MiB");
    }

    private static int CalibratedIterations(Action route, CudaBackend backend)
    {
        const int probes = 5;
        Stopwatch stopwatch = Stopwatch.StartNew();
        for (int i = 0; i < probes; i++) route();
        backend.Sync();
        stopwatch.Stop();
        double millisecondsPerCall = Math.Max(stopwatch.Elapsed.TotalMilliseconds / probes, 0.001);
        return Math.Clamp((int)Math.Ceiling(40.0 / millisecondsPerCall), 20, 4000);
    }

    private static double[] Measure(Action route, CudaBackend backend, int iterations)
    {
        double[] result = new double[Trials];
        Stopwatch stopwatch = new();
        for (int trial = 0; trial < Trials; trial++)
        {
            stopwatch.Restart();
            for (int i = 0; i < iterations; i++) route();
            backend.Sync();
            stopwatch.Stop();
            result[trial] = stopwatch.Elapsed.TotalMilliseconds / iterations;
        }
        return result;
    }

    private static Tensor CausalMask(int sequence)
    {
        Tensor mask = new(new TensorShape(1, 1, sequence, sequence), DType.F32);
        float* values = (float*)mask.DataPointer;
        for (int row = 0; row < sequence; row++)
        for (int column = 0; column < sequence; column++)
            values[row * sequence + column] = column <= row ? 0f : -1e30f;
        return mask;
    }

    private static Tensor RandomF32(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F32);
        float* values = (float*)tensor.DataPointer;
        Random random = new(seed);
        for (long i = 0; i < tensor.ElementCount; i++)
            values[i] = (float)(random.NextDouble() - 0.5);
        return tensor;
    }

    private static ErrorMetrics Difference(Tensor expected, Tensor actual)
    {
        float* expectedValues = (float*)expected.DataPointer;
        float* actualValues = (float*)actual.DataPointer;
        double maxAbs = 0;
        double sumAbs = 0;
        double sumSquared = 0;
        for (long i = 0; i < expected.ElementCount; i++)
        {
            double difference = Math.Abs((double)expectedValues[i] - actualValues[i]);
            maxAbs = Math.Max(maxAbs, difference);
            sumAbs += difference;
            sumSquared += difference * difference;
        }
        return new ErrorMetrics(
            maxAbs,
            Math.Sqrt(sumSquared / expected.ElementCount),
            sumAbs / expected.ElementCount);
    }

    private static double Median(double[] values)
    {
        double[] sorted = values.Order().ToArray();
        return sorted[sorted.Length / 2];
    }

    private static string Format(double[] values) =>
        string.Join(", ", values.Select(value => double.IsNaN(value) ? "n/a" : value.ToString("F4")));

    private readonly record struct ErrorMetrics(double MaxAbs, double Rmse, double MeanAbs);
}
