using HartsyInference.Core.Backends;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Tests.Shared;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The LongCat INT8 weights on the device against the host dequantization, bit for bit: mode 1 (a BF16-input layer, the reference's
/// BF16 nearest-even product) and mode 2 (a float32-input layer, ForceF32Gemm, the exact float32 product). The weight is applied to an identity
/// input, so the output is the weight itself in the GEMM's dtype: an exact pass shows the device unpacked the values the host would.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class LongCatInt8DeviceVsHostCudaTests
{
    private const int Size = 64;

    [CudaDeviceFact]
    public void ModeOneUnpacksTheBf16NearestEvenProductOnTheDevice()
    {
        using Tensor weight = Weight(out Tensor scale);
        weight.QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = scale, FullPrecisionMatMul = true, ReferenceRounding = true };
        using Tensor expected = Int8ConvRotCodec.DequantToBf16(weight, scale, 0, referenceRounding: true);
        using Tensor identity = BfIdentity();
        using CudaBackend backend = GpuDevices.OpenCuda();
        backend.PreloadWeights([weight]);
        weight.AsSpan<byte>().Clear();

        float[] device = Linear(backend, identity, weight);
        AssertWeightBits(expected, device, "mode 1 (BF16 nearest-even)");
    }

    [CudaDeviceFact]
    public void ModeTwoUnpacksTheExactFloat32ProductOnTheDeviceAndUsesAFloat32Gemm()
    {
        using Tensor weight = Weight(out Tensor scale);
        weight.QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = scale, FullPrecisionMatMul = true, ReferenceRounding = true };
        weight.ForceF32Gemm = true;
        using Tensor expected = Int8ConvRotCodec.DequantToF32Exact(weight, scale);
        using Tensor identity = F32Identity();
        using CudaBackend backend = GpuDevices.OpenCuda();
        backend.PreloadWeights([weight]);
        weight.AsSpan<byte>().Clear();

        float[] device = Linear(backend, identity, weight);
        AssertFloatWeightBits(expected, device, "mode 2 (exact float32)");
    }

    /// <summary>The host fallback under test. A row window takes the resident route out (its eligibility needs the whole weight), and
    /// FullPrecisionMatMul off takes the device unpack out, so the dense float32 weight is built on the host and the GEMM stays float32.</summary>
    [CudaDeviceFact]
    public void ModeTwoThroughTheHostFallbackKeepsTheFloat32Gemm()
    {
        using Tensor weight = Weight(out Tensor scale);
        weight.QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = scale, FullPrecisionMatMul = false, ReferenceRounding = true };
        weight.ForceF32Gemm = true;
        using Tensor expected = Int8ConvRotCodec.DequantToF32Exact(weight, scale);
        using Tensor identity = F32Identity();
        using CudaBackend backend = GpuDevices.OpenCuda();

        float[] fallback = LinearRows(backend, identity, weight);
        AssertFloatWeightBits(expected, fallback, "fallback mode 2 (exact float32)");
    }

    /// <summary>Mode 2 on the device through a row window: the same exact float32 product as the whole-weight device route.</summary>
    [CudaDeviceFact]
    public void ModeTwoRowWindowOnTheDeviceUnpacksTheSameFloat32Product()
    {
        using Tensor weight = Weight(out Tensor scale);
        weight.QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = scale, FullPrecisionMatMul = true, ReferenceRounding = true };
        weight.ForceF32Gemm = true;
        using Tensor expected = Int8ConvRotCodec.DequantToF32Exact(weight, scale);
        using Tensor identity = F32Identity();
        using CudaBackend backend = GpuDevices.OpenCuda();
        backend.PreloadWeights([weight]);
        weight.AsSpan<byte>().Clear();

        float[] device = LinearRows(backend, identity, weight);
        AssertFloatWeightBits(expected, device, "mode 2 row window (exact float32)");
    }

    /// <summary>Another family's forced-float32 weight (no ReferenceRounding) keeps its route: the host fallback's BF16 dequant under the BF16 GEMM,
    /// not the device unpack under a float32 GEMM. The activations are random, so the two GEMM dtypes give different bits and a route change shows.</summary>
    [CudaDeviceFact]
    public void AnotherFamilysForcedFloat32WeightKeepsItsHostFallbackRoute()
    {
        using Tensor weight = Weight(out Tensor scale);
        weight.QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = scale, FullPrecisionMatMul = true };
        weight.ForceF32Gemm = true;
        using Tensor dense = Int8ConvRotCodec.DequantToBf16(weight, scale, 0, referenceRounding: false);
        using Tensor input = RandomInput();
        using CudaBackend backend = GpuDevices.OpenCuda();

        float[] oldRoute = Linear(backend, input, dense);
        float[] got = Linear(backend, input, weight);
        AssertSameBits(oldRoute, got, "another family's forced-float32 weight (host fallback, BF16 GEMM)");
    }

    /// <summary>Levels of every value and scales over several decades, so the BF16 rounding of the product and of the scale both occur.</summary>
    private static Tensor Weight(out Tensor scale)
    {
        Tensor weight = new(new TensorShape(Size, Size), DType.I8);
        scale = new Tensor(new TensorShape(Size), DType.F32);
        Span<byte> levels = weight.AsSpan<byte>();
        Span<float> scales = scale.AsSpan<float>();
        Random rng = new(23);
        for (int row = 0; row < Size; row++)
        {
            for (int column = 0; column < Size; column++)
                levels[row * Size + column] = (byte)rng.Next(0, 256);
            scales[row] = (float)((1 + rng.NextDouble()) * Math.Pow(10, row % 5 - 3));
        }
        return weight;
    }

    private static Tensor BfIdentity()
    {
        Tensor identity = new(new TensorShape(Size, Size), DType.BF16);
        Span<ushort> bits = identity.AsSpan<ushort>();
        bits.Clear();
        for (int i = 0; i < Size; i++)
            bits[i * Size + i] = BFloat16Bits.FromFloat(1f);
        return identity;
    }

    private static Tensor F32Identity()
    {
        Tensor identity = new(new TensorShape(Size, Size), DType.F32);
        Span<float> values = identity.AsSpan<float>();
        values.Clear();
        for (int i = 0; i < Size; i++)
            values[i * Size + i] = 1f;
        return identity;
    }

    /// <summary>The identity applied to the weight: out[t][n] = W[n][t], so the output is the weight transposed in the GEMM's dtype.</summary>
    private static float[] Linear(IBackend backend, Tensor input, Tensor weight)
    {
        using Tensor output = new(new TensorShape(Size, Size), DType.F32);
        backend.Linear(output, input, weight, null);
        return output.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary>The identity applied through the row-window entry, all rows: the same product as <see cref="Linear"/>.</summary>
    private static float[] LinearRows(CudaBackend backend, Tensor input, Tensor weight)
    {
        using Tensor output = new(new TensorShape(Size, Size), DType.F32);
        backend.LinearWeightRows(output, input, weight, null, 0, Size);
        return output.AsReadOnlySpan<float>().ToArray();
    }

    private static Tensor RandomInput()
    {
        Tensor input = new(new TensorShape(Size, Size), DType.F32);
        Span<float> values = input.AsSpan<float>();
        Random rng = new(41);
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)(rng.NextDouble() * 2 - 1);
        return input;
    }

    private static void AssertSameBits(float[] expected, float[] got, string what)
    {
        Assert.Equal(expected.Length, got.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToUInt32Bits(expected[i]) == BitConverter.SingleToUInt32Bits(got[i]),
                $"{what} differs at element {i}: expected {expected[i]:R}, got {got[i]:R}");
    }

    /// <summary>The F32 variant of <see cref="AssertWeightBits"/>: the host values are F32, the device output is F32 transposed.</summary>
    private static void AssertFloatWeightBits(Tensor expected, float[] device, string mode)
    {
        ReadOnlySpan<float> values = expected.AsReadOnlySpan<float>();
        for (int n = 0; n < Size; n++)
            for (int t = 0; t < Size; t++)
            {
                float host = values[n * Size + t];
                float got = device[t * Size + n];
                Assert.True(BitConverter.SingleToUInt32Bits(host) == BitConverter.SingleToUInt32Bits(got),
                    $"{mode} differs at weight [{n}, {t}]: host {host:R}, device {got:R}");
            }
    }

    /// <summary>Compares the device output with the host weight, transposed (the identity gives out[t][n] = W[n][t]), bit for bit.</summary>
    private static void AssertWeightBits(Tensor expected, float[] device, string mode)
    {
        ReadOnlySpan<ushort> bits = expected.AsReadOnlySpan<ushort>();
        for (int n = 0; n < Size; n++)
            for (int t = 0; t < Size; t++)
            {
                float host = BFloat16Bits.ToFloat(bits[n * Size + t]);
                float got = device[t * Size + n];
                Assert.True(BitConverter.SingleToUInt32Bits(host) == BitConverter.SingleToUInt32Bits(got),
                    $"{mode} differs at weight [{n}, {t}]: host {host:R}, device {got:R}");
            }
    }
}

/// <summary>The per-tensor flags that choose the INT8 rounding and the GEMM dtype must survive a copy to another tensor and a LoRA adjunct view:
/// a copy that dropped <see cref="Tensor.ForceF32Gemm"/> would take the BF16 GEMM for a float32-input layer.</summary>
public sealed class LongCatInt8FlagCopyTests
{
    private const int Size = 64;

    [Fact]
    public void TheRoundingAndGemmFlagsSurviveACopyAndALowRankAdjunctView()
    {
        Tensor weight = new(new TensorShape(Size, Size), DType.I8);
        Tensor scale = new(new TensorShape(Size), DType.F32);
        scale.AsSpan<float>().Fill(1f);
        weight.QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = scale, FullPrecisionMatMul = true, ReferenceRounding = true };
        weight.ForceF32Gemm = true;

        using Tensor copy = weight.To(weight.Device);
        Assert.True(copy.ForceF32Gemm);
        Assert.True(copy.QuantInfo!.ReferenceRounding);

        LowRankAdjunctTerm term = new()
        {
            Down = LongCatTestData.Random(new TensorShape(4, Size), 1, 0.5f),
            Up = LongCatTestData.Random(new TensorShape(Size, 4), 2, 0.5f),
            Scale = 1f,
            Gate = null,
        };
        using Tensor view = weight.WithLowRankAdjunct(new LowRankAdjunct { Terms = [term] });
        Assert.True(view.ForceF32Gemm);
        Assert.True(view.QuantInfo!.ReferenceRounding);
        weight.Dispose();
    }
}
