using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>A tensor's <see cref="Tensor.Fp8ScaleFactor"/> scales its product on Vulkan whatever the operand's dtype, and
/// before the bias is added. The F16 block loops carry their residual damp there on 16-bit and F32 weights, with the
/// damp already in the bias values, so a GEMM that drops the factor runs the stream at the wrong scale against its own
/// bias. The reference is computed here from the values the device reads.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanGemmScaleParityTests
{
    private const float Damp = 1f / 32f;

    /// <summary>Rows 48 and 5 are the fragment-aligned and ragged shapes; an F32 product of at most 16 rows would
    /// otherwise take the fused GEMV, which has no alpha.</summary>
    [VulkanTheory]
    [InlineData(false, false, false, 48)]
    [InlineData(false, false, false, 5)]
    [InlineData(true, true, true, 48)]
    [InlineData(true, true, true, 5)]
    [InlineData(false, true, false, 48)]
    [InlineData(true, true, false, 48)]
    public void LinearScalesTheProductByTheWeightFactorBeforeTheBias(bool inputF16, bool weightF16, bool outputF16, int rows)
    {
        const int k = 64, n = 48;
        Random rng = new(11);
        using Tensor input = Make(Values(rng, rows * k), inputF16, rows, k);
        using Tensor weight = Make(Values(rng, n * k), weightF16, n, k);
        using Tensor bias = Make(Values(rng, n), f16: false, n);
        weight.Fp8ScaleFactor = Damp;
        using Tensor output = new(new TensorShape(rows, n), outputF16 ? DType.F16 : DType.F32);

        using VulkanBackend vulkan = new();
        vulkan.Linear(output, input, weight, bias);

        float[] x = Read(input), w = Read(weight), b = Read(bias), actual = Read(output);
        float worst = 0f;
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < n; c++)
            {
                double sum = 0;
                for (int i = 0; i < k; i++) sum += (double)x[r * k + i] * w[c * k + i];
                worst = MathF.Max(worst, MathF.Abs(actual[r * n + c] - (float)(Damp * sum + b[c])));
            }
        }
        // Bias values are of order 1 and the damped product of order 0.1: dropping the factor is off by several
        // units, and scaling the bias with the product by most of one.
        Assert.True(worst < Tolerance(inputF16 || weightF16 || outputF16), $"worst error {worst}");
    }

    [VulkanTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void MatMulScalesTheProductByBothOperandFactors(bool f16)
    {
        const int m = 32, k = 64, n = 48;
        Random rng = new(12);
        using Tensor a = Make(Values(rng, m * k), f16, m, k);
        using Tensor b = Make(Values(rng, k * n), f16, k, n);
        a.Fp8ScaleFactor = 0.5f;
        b.Fp8ScaleFactor = Damp;
        using Tensor output = new(new TensorShape(m, n), f16 ? DType.F16 : DType.F32);

        using VulkanBackend vulkan = new();
        vulkan.MatMul(output, a, b);

        float[] av = Read(a), bv = Read(b), actual = Read(output);
        float worst = 0f;
        for (int r = 0; r < m; r++)
        {
            for (int c = 0; c < n; c++)
            {
                double sum = 0;
                for (int i = 0; i < k; i++) sum += (double)av[r * k + i] * bv[i * n + c];
                worst = MathF.Max(worst, MathF.Abs(actual[r * n + c] - (float)(0.5 * Damp * sum)));
            }
        }
        Assert.True(worst < Tolerance(f16), $"worst error {worst}");
    }

    /// <summary>The per-batch path of <c>BatchedMatMul</c> (a weight per batch entry), one offset GEMM per entry.</summary>
    [VulkanTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void BatchedMatMulScalesEveryEntryByBothOperandFactors(bool f16)
    {
        const int batch = 2, m = 16, k = 64, n = 48;
        Random rng = new(13);
        using Tensor a = Make(Values(rng, batch * m * k), f16, batch, m, k);
        using Tensor b = Make(Values(rng, batch * k * n), f16, batch, k, n);
        a.Fp8ScaleFactor = 0.5f;
        b.Fp8ScaleFactor = Damp;
        using Tensor output = new(new TensorShape(batch, m, n), f16 ? DType.F16 : DType.F32);

        using VulkanBackend vulkan = new();
        vulkan.BatchedMatMul(output, a, b);

        float[] av = Read(a), bv = Read(b), actual = Read(output);
        float worst = 0f;
        for (int e = 0; e < batch; e++)
        {
            for (int r = 0; r < m; r++)
            {
                for (int c = 0; c < n; c++)
                {
                    double sum = 0;
                    for (int i = 0; i < k; i++) sum += (double)av[(e * m + r) * k + i] * bv[(e * k + i) * n + c];
                    worst = MathF.Max(worst, MathF.Abs(actual[(e * m + r) * n + c] - (float)(0.5 * Damp * sum)));
                }
            }
        }
        Assert.True(worst < Tolerance(f16), $"worst error {worst}");
    }

    /// <summary>A GGUF weight with a factor, at the row counts on both sides of <c>gemv_quant</c>'s limit: five rows would
    /// take the fused GEMV, which has no alpha and must decline, and forty-eight take the dequantize-then-GEMM path. The
    /// reference is the same call with a factor of 1, so the test needs no second dequantizer. <c>Linear</c> and
    /// <c>QuantizedMatMul</c> each pass the factor themselves, so both entry points run.</summary>
    [VulkanTheory]
    [InlineData(5, false)]
    [InlineData(48, false)]
    [InlineData(5, true)]
    [InlineData(48, true)]
    public void QuantizedLinearScalesTheProductByTheWeightFactorBeforeTheBias(int rows, bool viaQuantizedMatMul)
    {
        const int k = 64, n = 48;
        Random rng = new(14);
        using Tensor input = Make(Values(rng, rows * k), f16: false, rows, k);
        using Tensor weightF32 = Make(Values(rng, n * k), f16: false, n, k);
        using Tensor plainWeight = GgufQuantizer.Quantize(weightF32, DType.Q8_0);
        using Tensor scaledWeight = GgufQuantizer.Quantize(weightF32, DType.Q8_0);
        scaledWeight.Fp8ScaleFactor = Damp;
        using Tensor bias = Make(Values(rng, n), f16: false, n);
        using Tensor plain = new(new TensorShape(rows, n), DType.F32);
        using Tensor scaled = new(new TensorShape(rows, n), DType.F32);

        using VulkanBackend vulkan = new();
        if (viaQuantizedMatMul)
        {
            vulkan.QuantizedMatMul(plain, input, plainWeight, bias);
            vulkan.QuantizedMatMul(scaled, input, scaledWeight, bias);
        }
        else
        {
            vulkan.Linear(plain, input, plainWeight, bias);
            vulkan.Linear(scaled, input, scaledWeight, bias);
        }

        float[] b = Read(bias), expected = Read(plain), actual = Read(scaled);
        float worst = 0f;
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < n; c++)
                worst = MathF.Max(worst, MathF.Abs(actual[r * n + c] - (Damp * (expected[r * n + c] - b[c]) + b[c])));
        }
        Assert.True(worst < Tolerance(anyF16: true), $"worst error {worst}");
    }

    /// <summary>F16 results carry a 16-bit store of values up to about 2; F32 only the summation order.</summary>
    private static float Tolerance(bool anyF16) => anyF16 ? 5e-3f : 1e-4f;

    private static float[] Values(Random rng, int count)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return values;
    }

    private static Tensor Make(float[] values, bool f16, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), f16 ? DType.F16 : DType.F32);
        if (f16)
        {
            Span<Half> dst = t.AsSpan<Half>();
            for (int i = 0; i < values.Length; i++) dst[i] = (Half)values[i];
        }
        else
        {
            values.CopyTo(t.AsSpan<float>());
        }
        return t;
    }

    /// <summary>The tensor's values as the device holds them (an F16 tensor's are its rounded ones).</summary>
    private static float[] Read(Tensor t)
    {
        float[] values = new float[t.ElementCount];
        if (t.DType == DType.F16)
        {
            ReadOnlySpan<Half> src = t.AsReadOnlySpan<Half>();
            for (int i = 0; i < values.Length; i++) values[i] = (float)src[i];
        }
        else
        {
            t.AsReadOnlySpan<float>().CopyTo(values);
        }
        return values;
    }
}
