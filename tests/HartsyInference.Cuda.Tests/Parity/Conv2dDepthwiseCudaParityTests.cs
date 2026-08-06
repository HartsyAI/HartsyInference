using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>Parity coverage for CUDA Conv2dDepthwise against the CPU-verified hand-computed reference,
/// at both toy and realistic (NormalBAE MBConv, ConvNeXt-Small) scale. Written while bisecting a
/// CUDA-only NormalBAE/UperNet-Seg bug; this op turned out correct — the actual cause was
/// MaskRows/Transpose2D being called with a <c>.Reshape</c> view as the write target, which orphans
/// the CUDA activation cache entry from the object the caller reads back (see
/// NormalBaeModel.SqueezeExcite and UperNetSegModel for the fix). Kept as regression coverage.</summary>
[Collection("CudaSerial")]
public sealed unsafe class Conv2dDepthwiseCudaParityTests
{
    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    [Fact]
    public void Conv2dDepthwise_2Channels_Stride1_Pad1_3x3_MatchesHandComputed_CUDA()
    {
        using CudaBackend backend = new CudaBackend(0, PtxDir());
        using Tensor input = new(new TensorShape(1, 2, 3, 3), DType.F32);
        Span<float> inData = input.AsSpan<float>();
        inData[0] = 1; inData[1] = 2; inData[2] = 3;
        inData[3] = 4; inData[4] = 5; inData[5] = 6;
        inData[6] = 7; inData[7] = 8; inData[8] = 9;
        inData[9] = 9; inData[10] = 8; inData[11] = 7;
        inData[12] = 6; inData[13] = 5; inData[14] = 4;
        inData[15] = 3; inData[16] = 2; inData[17] = 1;

        using Tensor weight = new(new TensorShape(2, 1, 3, 3), DType.F32);
        Span<float> wData = weight.AsSpan<float>();
        for (int i = 0; i < 9; i++) wData[i] = 1f;
        for (int i = 9; i < 18; i++) wData[i] = 0f;
        wData[9 + 4] = 1f;

        using Tensor bias = new(new TensorShape(2), DType.F32);
        bias.AsSpan<float>()[0] = 10f;
        bias.AsSpan<float>()[1] = -10f;

        using Tensor output = new(new TensorShape(1, 2, 3, 3), DType.F32);
        backend.Conv2dDepthwise(output, input, weight, bias, 1, 1, 1, 1);
        backend.Sync();
        ReadOnlySpan<float> outData = new ReadOnlySpan<float>((float*)output.DataPointer, 18);

        float[] expectedC0 = [22, 31, 26, 37, 55, 43, 34, 49, 38];
        for (int i = 0; i < 9; i++)
            Assert.True(Math.Abs(expectedC0[i] - outData[i]) < 1e-3f, $"C0[{i}]: expected {expectedC0[i]}, got {outData[i]}");

        float[] inC1 = [9, 8, 7, 6, 5, 4, 3, 2, 1];
        for (int i = 0; i < 9; i++)
            Assert.True(Math.Abs((inC1[i] - 10) - outData[9 + i]) < 1e-3f, $"C1[{i}]: expected {inC1[i] - 10}, got {outData[9 + i]}");
    }

    [Fact]
    public void Conv2dDepthwise_Stride2_Pad0_MatchesCpu()
    {
        // The exact call pattern NormalBAE's stride-2 blocks use: pre-padded input, stride 2, pad 0.
        using Tensor input = new(new TensorShape(1, 1, 4, 4), DType.F32);
        Span<float> inData = input.AsSpan<float>();
        for (int i = 0; i < 16; i++) inData[i] = i + 1;

        using Tensor weight = new(new TensorShape(1, 1, 3, 3), DType.F32);
        Span<float> wData = weight.AsSpan<float>();
        for (int i = 0; i < 9; i++) wData[i] = 1f;

        using Tensor cpuOut = new(new TensorShape(1, 1, 1, 1), DType.F32);
        using (HartsyInference.Cpu.CpuBackend cpuImpl = new HartsyInference.Cpu.CpuBackend())
        {
            IBackend cpu = cpuImpl;
            cpu.Conv2dDepthwise(cpuOut, input, weight, null, 2, 2, 0, 0);
        }

        using CudaBackend backend = new CudaBackend(0, PtxDir());
        using Tensor cudaOut = new(new TensorShape(1, 1, 1, 1), DType.F32);
        backend.Conv2dDepthwise(cudaOut, input, weight, null, 2, 2, 0, 0);
        backend.Sync();

        float cpuVal = *(float*)cpuOut.DataPointer;
        float cudaVal = *(float*)cudaOut.DataPointer;
        Assert.True(Math.Abs(cpuVal - cudaVal) < 1e-3f, $"expected {cpuVal}, got {cudaVal}");
    }

    private static Tensor Random(TensorShape shape, int seed, float lo = -1f, float hi = 1f)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        long n = shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)(rng.NextDouble() * (hi - lo) + lo);
        return t;
    }

    [Theory]
    [InlineData(96, 128, 128, 7, 1, 3)]   // ConvNeXt-Small depthwise: 96ch, 7x7, stride1 pad3
    [InlineData(24, 64, 64, 3, 2, 0)]     // NormalBAE MBConv stride2 (pre-padded, pad0)
    public void Conv2dDepthwise_RealisticScale_MatchesCpu(int channels, int h, int w, int kernel, int stride, int pad)
    {
        using Tensor input = Random(new TensorShape(1, channels, h, w), seed: 1);
        using Tensor weight = Random(new TensorShape(channels, 1, kernel, kernel), seed: 2, lo: -0.3f, hi: 0.3f);
        using Tensor bias = Random(new TensorShape(channels), seed: 3, lo: -0.1f, hi: 0.1f);

        int outH = (h + 2 * pad - kernel) / stride + 1;
        int outW = (w + 2 * pad - kernel) / stride + 1;

        using Tensor cpuOut = new(new TensorShape(1, channels, outH, outW), DType.F32);
        using (HartsyInference.Cpu.CpuBackend cpuImpl = new HartsyInference.Cpu.CpuBackend())
        {
            IBackend cpu = cpuImpl;
            cpu.Conv2dDepthwise(cpuOut, input, weight, bias, stride, stride, pad, pad);
        }

        using CudaBackend backend = new CudaBackend(0, PtxDir());
        using Tensor cudaOut = new(new TensorShape(1, channels, outH, outW), DType.F32);
        backend.Conv2dDepthwise(cudaOut, input, weight, bias, stride, stride, pad, pad);
        backend.Sync();

        float* cpuP = (float*)cpuOut.DataPointer;
        float* cudaP = (float*)cudaOut.DataPointer;
        long n = cpuOut.ElementCount;
        double maxErr = 0, sumErr = 0;
        long firstBad = -1;
        for (long i = 0; i < n; i++)
        {
            double err = Math.Abs(cpuP[i] - cudaP[i]);
            if (err > maxErr) { maxErr = err; if (firstBad < 0 && err > 1e-2) firstBad = i; }
            sumErr += err;
        }
        double avgErr = sumErr / n;
        Assert.True(maxErr < 1e-2, $"maxErr={maxErr:E3} avgErr={avgErr:E3} n={n} firstBadIdx={firstBad} " +
            $"cpu[0]={cpuP[0]} cuda[0]={cudaP[0]} cpu[last]={cpuP[n - 1]} cuda[last]={cudaP[n - 1]}");
    }
}
