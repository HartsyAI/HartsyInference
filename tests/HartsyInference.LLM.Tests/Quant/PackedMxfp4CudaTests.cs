using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Quant;

/// <summary>The CUDA MXFP4-packed linear against the Core host reference, on random packed weights.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class PackedMxfp4CudaTests
{
    private readonly ITestOutputHelper _output;
    public PackedMxfp4CudaTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(3, 64, 6)]
    [InlineData(1, 224, 40)]
    public void Cuda_MatchesHostReference(int n, int cols, int rows)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        Random rng = new(rows * 7 + cols);
        byte[] nibbles = new byte[rows * cols / 2];
        rng.NextBytes(nibbles);
        byte[] scales = new byte[rows * (cols / 32)];
        for (int i = 0; i < scales.Length; i++) scales[i] = (byte)rng.Next(118, 130);
        float[] x = new float[n * cols];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        using Tensor weight = Bytes(nibbles, rows, cols / 2);
        using Tensor scale = Bytes(scales, rows, cols / 32);
        QuantRecipe recipe = new()
        {
            Encoding = QuantEncoding.Mxfp4E8M0, Geometry = new BlockGeometry(1, 32), ScaleDType = DType.U8,
            LogicalRows = rows, LogicalCols = cols, Scale = scale,
        };
        using Tensor input = Floats(x, n, cols);
        using Tensor expected = new(new TensorShape(n, rows), DType.F32);
        using Tensor actual = new(new TensorShape(n, rows), DType.F32);

        Mxfp4LinearReference.Apply(expected, input, weight, recipe);
        using CudaBackend backend = new(0, PtxDir());
        IBackend device = backend;
        device.Mxfp4LinearF32(actual, input, weight, recipe);
        // Without the module the backend answers from the host reference, so the device path must be checked as loaded.
        CudaKernels? kernels = (CudaKernels?)typeof(CudaBackend).GetField("_kernels", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)?.GetValue(backend);
        Assert.True(kernels?.HasMxfp4LinearKernel == true, "mxfp4_linear_f32.ptx did not load; the device path did not run.");
        backend.OffloadActivation(actual);

        float[] want = expected.AsReadOnlySpan<float>().ToArray();
        float[] got = actual.AsReadOnlySpan<float>().ToArray();
        for (int i = 0; i < want.Length; i++)
            Assert.True(MathF.Abs(want[i] - got[i]) <= 1e-3f * MathF.Max(1f, MathF.Abs(want[i])), $"y[{i}] host {want[i]} cuda {got[i]}");
    }

    private static Tensor Bytes(byte[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.U8);
        values.CopyTo(t.AsSpan<byte>());
        return t;
    }

    private static Tensor Floats(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static string PtxDir()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }
        return Path.Combine(AppContext.BaseDirectory, "Ptx");
    }
}
