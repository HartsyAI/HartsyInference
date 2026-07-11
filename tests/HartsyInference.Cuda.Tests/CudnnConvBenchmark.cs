using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Wall-clock benchmark of the cuDNN conv-forward fast path vs the im2col+cuBLAS fallback on
/// SDXL-UNet-shaped F16 convolutions. NOT a correctness test (CudnnConvTests covers that) — it measures
/// the real speedup the fast path buys once a version-matched cuDNN is provisioned. Opt-in:
/// <c>dotnet test --filter Category=ConvBench</c>. Skips cleanly if CUDA or cuDNN is unavailable.</summary>
[Collection("CudaSerial")]
[Trait("Category", "ConvBench")]
public sealed unsafe class CudnnConvBenchmark
{
    private readonly ITestOutputHelper _output;
    public CudnnConvBenchmark(ITestOutputHelper output) => _output = output;

    private static uint _rng = 0x1234567u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return (_rng & 0xFFFF) / 65535f - 0.5f; }

    private static Tensor RndF16(TensorShape shape)
    {
        Tensor t = new(shape, DType.F16);
        ushort* p = (ushort*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = BitConverter.HalfToUInt16Bits((Half)Rand());
        return t;
    }

    // (batch, inCh, H, W, outCh, k, stride, pad) — a spread of SDXL-UNet conv shapes across resolutions.
    private static readonly (int b, int cIn, int h, int w, int cOut, int k, int s, int p)[] Shapes =
    {
        (2, 320, 64, 64, 320, 3, 1, 1),   // high-res block
        (2, 640, 32, 32, 640, 3, 1, 1),   // mid-res block
        (2, 1280, 16, 16, 1280, 3, 1, 1), // low-res block
        (2, 320, 64, 64, 320, 1, 1, 0),   // 1x1 projection
    };

    [Fact]
    public void Benchmark_CudnnVsIm2Col()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        CudnnRuntime.EnsureProbed();
        _output.WriteLine($"cuDNN: {(CudnnRuntime.Available ? $"ACTIVE ({CudnnRuntime.Version})" : "NOT ACTIVE")} — {CudnnRuntime.Reason}");
        if (!CudnnRuntime.Available) { _output.WriteLine("SKIPPED: cuDNN not provisioned — nothing to compare against"); return; }

        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        const int warmup = 5, iters = 40;
        double totCudnn = 0, totIm2Col = 0;
        _output.WriteLine($"warmup={warmup} iters={iters}  (ms = per-call average)");
        _output.WriteLine($"{"shape",-28}{"im2col",10}{"cuDNN",10}{"speedup",10}  engaged");

        foreach ((int b, int cIn, int h, int w, int cOut, int k, int s, int p) in Shapes)
        {
            int outH = (h + 2 * p - k) / s + 1, outW = (w + 2 * p - k) / s + 1;
            using Tensor input = RndF16(new TensorShape(b, cIn, h, w));
            using Tensor weight = RndF16(new TensorShape(cOut, cIn, k, k));
            using Tensor bias = RndF16(new TensorShape(cOut));
            using Tensor output = new(new TensorShape(b, cOut, outH, outW), DType.F16);

            double cudnnMs = TimeConv(ptxDir, "1", input, weight, bias, output, s, p, warmup, iters, out bool engaged);
            double im2colMs = TimeConv(ptxDir, "0", input, weight, bias, output, s, p, warmup, iters, out _);
            totCudnn += cudnnMs; totIm2Col += im2colMs;
            string tag = $"{b}x{cIn}x{h}x{w} k{k}s{s}";
            _output.WriteLine($"{tag,-28}{im2colMs,9:F3}m{cudnnMs,9:F3}m{im2colMs / cudnnMs,9:F2}x  {engaged}");
        }
        _output.WriteLine(new string('-', 66));
        _output.WriteLine($"{"TOTAL",-28}{totIm2Col,9:F3}m{totCudnn,9:F3}m{totIm2Col / totCudnn,9:F2}x");
    }

    private static double TimeConv(string ptxDir, string flag, Tensor input, Tensor weight, Tensor bias,
        Tensor output, int stride, int pad, int warmup, int iters, out bool engaged)
    {
        string? prev = Environment.GetEnvironmentVariable("HARTSY_CONV_CUDNN");
        Environment.SetEnvironmentVariable("HARTSY_CONV_CUDNN", flag);
        try
        {
            using CudaBackend backend = new(0, ptxDir);
            IBackend b = backend;
            for (int i = 0; i < warmup; i++) b.Conv2D(output, input, weight, bias, stride, stride, pad, pad);
            backend.Sync();
            Stopwatch sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++) b.Conv2D(output, input, weight, bias, stride, stride, pad, pad);
            backend.Sync();
            sw.Stop();
            engaged = backend.CudnnConvEngaged;
            return sw.Elapsed.TotalMilliseconds / iters;
        }
        finally { Environment.SetEnvironmentVariable("HARTSY_CONV_CUDNN", prev); }
    }
}
