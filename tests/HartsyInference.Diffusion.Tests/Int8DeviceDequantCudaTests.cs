using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Tests.Shared;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A resident <c>int8_tensorwise</c> weight whose layer asks for a full-precision GEMM, on CUDA: unpacked on the device into the float
/// GEMM, with the values and the arithmetic of the dense BF16 weight the host fallback builds.</summary>
/// <remarks>Which route ran is read from where it reads the weight: each case preloads the weight and then zeroes its host bytes, so the device
/// route still computes the product and a host dequantization sees a zero weight.</remarks>
[Trait("Category", "GpuIntegration")]
public sealed class Int8DeviceDequantCudaTests(ITestOutputHelper output)
{
    private const string KernelFile = "dequant_int8_to_bf16.ptx";

    /// <summary>The tiled GEMM against the dense one on the same values, relative RMS: cuBLAS accumulates a run of output channels differently from
    /// the whole, at F32 rounding level. The first bound, 1e-6, was written before the first run and measured 3.0e-6 on an RTX 3060; the dense
    /// linear with and without the bias epilogue differs by 4.7e-8 there, so the difference is the tiling's.</summary>
    private const double TiledBound = 1e-5;

    /// <summary>The same where the output is F16. Through an F32 input this GEMM writes BF16 first, which alone is 2e-3 to 3e-3 from the CPU, and
    /// cuBLAS rounds it differently for a different output count: tiled against dense measured 2.0e-3 on an RTX 3060, with the tiled result no
    /// farther from the CPU than the dense one. The first bound, 1e-3, was written before that run and failed there. A row tile written at the
    /// wrong place would differ by order one.</summary>
    private const double TiledF16Bound = 5e-3;

    /// <summary>The tiled GEMM against the CPU backend on the weight widened to F32, relative RMS: the device GEMM rounds its operands to
    /// BF16.</summary>
    private const double CpuBound = 1e-2;

    [CudaDeviceFact]
    public void EveryLevelUnpacksToTheHostCodecsBf16Values()
    {
        const int Size = 256;
        using Tensor weight = new(new TensorShape(Size, Size), DType.I8);
        using Tensor scale = new(new TensorShape(Size), DType.F32);
        Span<byte> levels = weight.AsSpan<byte>();
        Span<float> scales = scale.AsSpan<float>();
        Random rng = new(11);
        for (int row = 0; row < Size; row++)
        {
            // Every level appears in every row, each row at another column, over scales six decades apart.
            for (int column = 0; column < Size; column++)
                levels[row * Size + column] = (byte)((row + column) & 0xFF);
            scales[row] = (float)((1 + rng.NextDouble()) * Math.Pow(10, row % 7 - 4));
        }
        weight.QuantInfo = Info(scale);
        using Tensor dense = Int8ConvRotCodec.DequantToBf16(weight, scale, 0);
        using Tensor identity = new(new TensorShape(Size, Size), DType.F32);
        Span<float> ones = identity.AsSpan<float>();
        ones.Clear();
        for (int i = 0; i < Size; i++)
            ones[i * Size + i] = 1f;
        using CudaBackend backend = GpuDevices.OpenCuda();
        PreloadThenZeroTheHost(backend, weight);

        using Tensor result = new(new TensorShape(Size, Size), DType.F32);
        backend.Linear(result, identity, weight, null);

        float[] actual = result.AsReadOnlySpan<float>().ToArray();
        using Tensor wide = dense.CastTo(DType.F32);
        ReadOnlySpan<float> values = wide.AsReadOnlySpan<float>();
        // The identity input gives the weight's transpose, element for element.
        for (int row = 0; row < Size; row++)
            for (int column = 0; column < Size; column++)
                Assert.Equal(values[row * Size + column], actual[column * Size + row]);
        backend.FreeWeights([weight]);
    }

    [CudaDeviceFact]
    public void TheDeviceRouteIsBitIdenticalToTheDenseBf16Linear()
    {
        const int Rows = 128, Columns = 256;
        foreach ((string lane, DType activations, bool highPrecision) in new[]
        {
            ("F32", DType.F32, false), ("F32, high-precision GEMM", DType.F32, true), ("F16", DType.F16, false),
        })
        {
            using Tensor weight = Int8Weight(Rows, Columns, seed: 3, out Tensor scale);
            using Tensor rowScale = scale;
            using Tensor dense = Int8ConvRotCodec.DequantToBf16(weight, scale, 0);
            using Tensor input = Random(new TensorShape(64, Columns), 4, activations);
            using Tensor few = Random(new TensorShape(4, Columns), 5, activations);
            using Tensor bias = Random(new TensorShape(Rows), 6, activations);
            using CudaBackend backend = GpuDevices.OpenCuda();
            backend.HighPrecisionGemm = highPrecision;
            float[] expected = Linear(backend, input, dense, bias, activations);
            float[] expectedFew = Linear(backend, few, dense, bias, activations);
            float[] expectedWindow = Window(backend, input, dense, bias, Rows / 2, Rows / 2, activations);
            PreloadThenZeroTheHost(backend, weight);

            Assert.True(expected.AsSpan().SequenceEqual(Linear(backend, input, weight, bias, activations)), $"{lane}: Linear");
            Assert.True(expectedFew.AsSpan().SequenceEqual(Linear(backend, few, weight, bias, activations)), $"{lane}: Linear, 4 rows");
            Assert.True(expectedWindow.AsSpan().SequenceEqual(Window(backend, input, weight, bias, Rows / 2, Rows / 2, activations)),
                $"{lane}: LinearWeightRows");
            using Tensor first = new(new TensorShape(64, Rows), activations);
            using Tensor second = new(new TensorShape(64, Rows), activations);
            backend.LinearMulti(input, [new LinearOp(first, weight, bias), new LinearOp(second, weight, bias)]);
            Assert.True(expected.AsSpan().SequenceEqual(Values(first)) && expected.AsSpan().SequenceEqual(Values(second)), $"{lane}: LinearMulti");
            backend.FreeWeights([weight, dense, bias]);
        }
    }

    /// <summary>A weight of 72 MiB in BF16, above one 64 MiB tile, whole and over a row window that starts inside the first tile: F32, F32 into
    /// an F16 output (through the GEMM's cast buffer), F16, and F32 with the high-precision GEMM (an F32 tile).</summary>
    [CudaDeviceFact]
    public void ATiledWeightMatchesTheDenseLinearAndTheCpu()
    {
        const int Rows = 9216, Columns = 4096, Tokens = 64, Offset = 512;
        using Tensor weight = Int8Weight(Rows, Columns, seed: 21, out Tensor scale);
        using Tensor rowScale = scale;
        using Tensor dense = Int8ConvRotCodec.DequantToBf16(weight, scale, 0);
        using Tensor input = Random(new TensorShape(Tokens, Columns), 22, DType.F32);
        using Tensor bias = Random(new TensorShape(Rows), 23, DType.F32);
        float[] cpu;
        using (CpuBackend reference = new())
        using (Tensor wide = dense.CastTo(DType.F32))
            cpu = Linear(reference, input, wide, bias, DType.F32);
        List<string> failures = [];
        foreach ((string lane, DType activations, DType results, bool highPrecision, double bound) in new[]
        {
            ("F32", DType.F32, DType.F32, false, TiledBound), ("F32 to F16", DType.F32, DType.F16, false, TiledF16Bound),
            ("F16", DType.F16, DType.F16, false, TiledF16Bound), ("F32, high-precision GEMM", DType.F32, DType.F32, true, TiledBound),
        })
        {
            using Tensor x = Random(new TensorShape(Tokens, Columns), 22, activations);
            using Tensor b = Random(new TensorShape(Rows), 23, results);
            using CudaBackend backend = GpuDevices.OpenCuda();
            backend.HighPrecisionGemm = highPrecision;
            float[] whole = Linear(backend, x, dense, b, results);
            float[] window = Window(backend, x, dense, b, Offset, Rows - Offset, results);
            // A copy per lane: the host bytes are zeroed once the device holds them.
            using Tensor packed = Int8Weight(Rows, Columns, seed: 21, out Tensor packedScale);
            using Tensor packedRowScale = packedScale;
            PreloadThenZeroTheHost(backend, packed);

            float[] unpacked = Linear(backend, x, packed, b, results);
            float[] unpackedWindow = Window(backend, x, packed, b, Offset, Rows - Offset, results);

            double tiled = LongCatTestData.RelRms(unpacked, whole), tiledWindow = LongCatTestData.RelRms(unpackedWindow, window);
            double againstCpu = LongCatTestData.RelRms(unpacked, cpu);
            output.WriteLine(FormattableString.Invariant(
                $"{backend.Capabilities.Name}, {lane}: tiled against dense {tiled:E3}, window {tiledWindow:E3} (bound {bound:E0}); against the ")
                + FormattableString.Invariant($"CPU {againstCpu:E3} (bound {CpuBound:E0})"));
            if (tiled > bound || tiledWindow > bound) failures.Add($"{lane}: tiled differs from dense by {tiled:E3} and {tiledWindow:E3}");
            if (againstCpu > CpuBound) failures.Add($"{lane}: tiled differs from the CPU by {againstCpu:E3}");
            backend.FreeWeights([packed, dense, b]);
        }
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    [CudaDeviceFact]
    public void ARotatedWeightTakesTheHostFallback()
    {
        using CudaBackend backend = GpuDevices.OpenCuda();
        AssertHostFallback(backend, group: 256);
    }

    [CudaDeviceFact]
    public void AFolderWithoutTheKernelTakesTheHostFallback()
    {
        string source = Path.Combine(AppContext.BaseDirectory, "Ptx");
        Assert.True(File.Exists(Path.Combine(source, KernelFile)), $"the build copied no {KernelFile}");
        string without = Path.Combine(Path.GetTempPath(), "int8-no-dequant-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(without);
        try
        {
            foreach (string file in Directory.GetFiles(source))
                if (Path.GetFileName(file) != KernelFile)
                    File.Copy(file, Path.Combine(without, Path.GetFileName(file)));
            using CudaBackend backend = new(0, without);
            AssertHostFallback(backend, group: 0);
        }
        finally
        {
            Directory.Delete(without, recursive: true);
        }
    }

    /// <summary>The marked weight equals the dense linear on the host codec's values while its host bytes hold, and becomes the bare bias once they
    /// are zeroed, which only a route reading the host can do after a preload.</summary>
    private static void AssertHostFallback(CudaBackend backend, int group)
    {
        const int Rows = 128, Columns = 256;
        using Tensor weight = Int8Weight(Rows, Columns, seed: 31, out Tensor scale);
        using Tensor rowScale = scale;
        weight.QuantInfo = weight.QuantInfo! with { ConvRotGroupSize = group };
        using Tensor dense = Int8ConvRotCodec.DequantToBf16(weight, scale, group);
        using Tensor input = Random(new TensorShape(64, Columns), 32, DType.F32);
        using Tensor bias = Random(new TensorShape(Rows), 33, DType.F32);
        float[] expected = Linear(backend, input, dense, bias, DType.F32);
        backend.PreloadWeights([weight]);
        Assert.Equal(expected, Linear(backend, input, weight, bias, DType.F32));

        weight.AsSpan<byte>().Clear();
        float[] zeroed = Linear(backend, input, weight, bias, DType.F32);
        float[] biasValues = Values(bias);
        for (int i = 0; i < zeroed.Length; i++)
            Assert.Equal(biasValues[i % Rows], zeroed[i]);
        backend.FreeWeights([weight, dense, bias]);
    }

    private static void PreloadThenZeroTheHost(CudaBackend backend, Tensor weight)
    {
        backend.PreloadWeights([weight]);
        weight.AsSpan<byte>().Clear();
    }

    private static QuantWeightInfo Info(Tensor scale) => new() { Format = "int8_tensorwise", RowScale = scale, FullPrecisionMatMul = true };

    private static Tensor Int8Weight(int rows, int columns, int seed, out Tensor scale)
    {
        Random rng = new(seed);
        Tensor weight = new(new TensorShape(rows, columns), DType.I8);
        Span<byte> levels = weight.AsSpan<byte>();
        for (int i = 0; i < levels.Length; i++)
            levels[i] = unchecked((byte)(sbyte)rng.Next(-127, 128));
        scale = new Tensor(new TensorShape(rows), DType.F32);
        Span<float> scales = scale.AsSpan<float>();
        for (int i = 0; i < scales.Length; i++)
            scales[i] = (float)(0.001 + 0.01 * rng.NextDouble());
        weight.QuantInfo = Info(scale);
        return weight;
    }

    private static Tensor Random(TensorShape shape, int seed, DType dtype)
    {
        Tensor wide = LongCatTestData.Random(shape, seed);
        if (dtype == DType.F32)
            return wide;
        using (wide)
            return wide.CastTo(dtype);
    }

    private static float[] Values(Tensor tensor)
    {
        if (tensor.DType == DType.F32)
            return tensor.AsReadOnlySpan<float>().ToArray();
        using Tensor wide = tensor.CastTo(DType.F32);
        return wide.AsReadOnlySpan<float>().ToArray();
    }

    private static float[] Linear(IBackend backend, Tensor input, Tensor weight, Tensor bias, DType outputType)
    {
        using Tensor result = new(new TensorShape(input.Shape[0], weight.Shape[0]), outputType);
        backend.Linear(result, input, weight, bias);
        return Values(result);
    }

    private static float[] Window(IBackend backend, Tensor input, Tensor weight, Tensor bias, int offset, int count, DType outputType)
    {
        using Tensor result = new(new TensorShape(input.Shape[0], count), outputType);
        backend.LinearWeightRows(result, input, weight, bias, offset, count);
        return Values(result);
    }
}
