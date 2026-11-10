using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Backends.Latent;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

public sealed class Fp8BlockLinearTests
{
    [Fact]
    public void Fp8BlockLinear_MatchesDequantizedDenseProduct_WithPartialBlocks()
    {
        // 130 outputs and 300 inputs leave partial 128-blocks in both directions.
        const int m = 3, n = 130, k = 300, block = 128;
        Random rng = new(21);
        byte[] codes = RandomCodes(rng, n * k);
        float[] scales = RandomFloats(rng, 2 * 3, 0.5f, 2f);
        float[] x = RandomFloats(rng, m * k, -1f, 1f);
        float[] bias = RandomFloats(rng, n, -1f, 1f);
        using Tensor input = Make(x, m, k);
        using Tensor weight = MakeBytes(codes, n, k);
        using Tensor scale = Make(scales, 2, 3);
        using Tensor b = Make(bias, n);
        using Tensor output = new(new TensorShape(m, n), DType.F32);

        new CpuBackend().Fp8BlockLinear(output, input, weight, scale, b, block, block);

        float[] y = output.AsReadOnlySpan<float>().ToArray();
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double sum = bias[j];
                for (int c = 0; c < k; c++)
                {
                    float s = scales[(j / block) * 3 + c / block];
                    float w = LatentCodec.DecodeE4M3(codes[j * k + c]) * s;
                    sum += (double)x[i * k + c] * w;
                }
                Assert.True(Math.Abs(sum - y[i * n + j]) <= 1e-5 * Math.Max(1, Math.Abs(sum)), $"[{i},{j}] {sum} vs {y[i * n + j]}");
            }
    }

    [Fact]
    public void Fp8BlockLinear_WithoutBias_IsTheBareProduct()
    {
        const int m = 2, n = 4, k = 128;
        Random rng = new(4);
        byte[] codes = RandomCodes(rng, n * k);
        float[] x = RandomFloats(rng, m * k, -1f, 1f);
        using Tensor input = Make(x, m, k);
        using Tensor weight = MakeBytes(codes, n, k);
        using Tensor scale = Make(new float[] { 1f }, 1, 1);
        using Tensor output = new(new TensorShape(m, n), DType.F32);

        new CpuBackend().Fp8BlockLinear(output, input, weight, scale, null);

        float[] y = output.AsReadOnlySpan<float>().ToArray();
        for (int j = 0; j < n; j++)
        {
            double sum = 0;
            for (int c = 0; c < k; c++) sum += (double)x[c] * LatentCodec.DecodeE4M3(codes[j * k + c]);
            Assert.True(Math.Abs(sum - y[j]) <= 1e-5 * Math.Max(1, Math.Abs(sum)), $"[{j}] {sum} vs {y[j]}");
        }
    }

    [Fact]
    public void Fp8BlockLinear_HandComputedSingleCode()
    {
        // e4m3 0x38 decodes to 1.0; with scale 2 every weight is 2, so each output is 2 * sum(x).
        const int k = 4;
        using Tensor input = Make(new float[] { 1f, 2f, 3f, 4f }, 1, k);
        using Tensor weight = MakeBytes(new byte[] { 0x38, 0x38, 0x38, 0x38 }, 1, k);
        using Tensor scale = Make(new float[] { 2f }, 1, 1);
        using Tensor output = new(new TensorShape(1, 1), DType.F32);

        new CpuBackend().Fp8BlockLinear(output, input, weight, scale, null);

        Assert.Equal(20f, output.AsReadOnlySpan<float>()[0]);
    }

    [Fact]
    public void Fp8BlockLinear_NanWeightCodePropagates()
    {
        // 0x7F is e4m3 NaN; any output that reads it must be NaN.
        using Tensor input = Make(new float[] { 1f, 1f }, 1, 2);
        using Tensor weight = MakeBytes(new byte[] { 0x7F, 0x38 }, 1, 2);
        using Tensor scale = Make(new float[] { 1f }, 1, 1);
        using Tensor output = new(new TensorShape(1, 1), DType.F32);

        new CpuBackend().Fp8BlockLinear(output, input, weight, scale, null);

        Assert.True(float.IsNaN(output.AsReadOnlySpan<float>()[0]));
    }

    [Fact]
    public void Fp8BlockLinear_RejectsZeroInputWidth()
    {
        using Tensor input = new(new TensorShape(1, 0), DType.F32);
        using Tensor weight = new(new TensorShape(4, 0), DType.F8E4M3);
        using Tensor scale = new(new TensorShape(1, 1), DType.F32);
        using Tensor output = new(new TensorShape(1, 4), DType.F32);
        Assert.Throws<ArgumentException>(() => Fp8BlockLinearReference.Validate(output, input, weight, scale, null, 128, 128));
    }

    [Fact]
    public void Fp8BlockLinear_RejectsScaleGridOfTheWrongShape()
    {
        using Tensor input = new(new TensorShape(1, 256), DType.F32);
        using Tensor weight = new(new TensorShape(128, 256), DType.F8E4M3);
        using Tensor scale = new(new TensorShape(1, 1), DType.F32);
        using Tensor output = new(new TensorShape(1, 128), DType.F32);
        Assert.Throws<ArgumentException>(() => Fp8BlockLinearReference.Validate(output, input, weight, scale, null, 128, 128));
    }

    [Fact]
    public void Fp8BlockLinear_RefusesAnF32Weight()
    {
        using Tensor input = new(new TensorShape(1, 128), DType.F32);
        using Tensor weight = new(new TensorShape(4, 128), DType.F32);
        using Tensor scale = new(new TensorShape(1, 1), DType.F32);
        using Tensor output = new(new TensorShape(1, 4), DType.F32);
        Assert.Throws<ArgumentException>(() => Fp8BlockLinearReference.Validate(output, input, weight, scale, null, 128, 128));
    }

    internal static byte[] RandomCodes(Random rng, int count)
    {
        byte[] codes = new byte[count];
        for (int i = 0; i < count; i++)
        {
            // Finite codes only: NaN (0x7F, 0xFF) would make every comparison NaN.
            byte code = LatentCodec.EncodeE4M3((float)(rng.NextDouble() * 2 - 1));
            codes[i] = (code & 0x7F) == 0x7F ? (byte)0 : code;
        }
        return codes;
    }

    internal static float[] RandomFloats(Random rng, int count, float lo, float hi)
    {
        float[] v = new float[count];
        for (int i = 0; i < count; i++) v[i] = lo + (float)rng.NextDouble() * (hi - lo);
        return v;
    }

    internal static Tensor Make(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    internal static Tensor MakeBytes(byte[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F8E4M3);
        values.CopyTo(t.AsSpan<byte>());
        return t;
    }
}
