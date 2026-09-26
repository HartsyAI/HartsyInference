using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>The cooperative-matrix-2 flash attention against a double-precision host SDPA on F16-rounded inputs:
/// head-major and token-major, grouped-query, masked, and sequence lengths that are not tile multiples.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanFlashCm2Tests(ITestOutputHelper output)
{
    private const double Tolerance = 1e-3;

    private static bool VulkanAvailable()
    {
        try { using VulkanInstance i = new(); return i.EnumeratePhysicalDevices().Length > 0; }
        catch { return false; }
    }

    [Theory]
    [InlineData(1, 4, 4, 300, 300, 128, false)]
    [InlineData(1, 4, 2, 300, 257, 128, true)]
    [InlineData(2, 8, 8, 130, 70, 64, false)]
    [InlineData(1, 6, 3, 65, 129, 64, true)]
    public void HeadMajor_MatchesReference(int batch, int hq, int hkv, int sq, int skv, int d, bool masked)
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();
        if (!backend.SupportsTokenMajorGqaAttention) return;
        float[] q = Half(Random(batch * hq * sq * d, 1));
        float[] k = Half(Random(batch * hkv * skv * d, 2));
        float[] v = Half(Random(batch * hkv * skv * d, 3));
        float[]? mask = masked ? Random(sq * skv, 4, 2f) : null;
        float scale = 1f / MathF.Sqrt(d);
        float[] expected = Reference(q, k, v, mask, batch, hq, hkv, sq, skv, d, scale);

        using Tensor qt = FromHost(new TensorShape(batch, hq, sq, d), q);
        using Tensor kt = FromHost(new TensorShape(batch, hkv, skv, d), k);
        using Tensor vt = FromHost(new TensorShape(batch, hkv, skv, d), v);
        using Tensor? mt = mask is null ? null : FromHost(new TensorShape(sq, skv), mask);
        using Tensor ot = new(new TensorShape(batch, hq, sq, d), DType.F32);
        backend.ScaledDotProductAttention(ot, qt, kt, vt, mt, scale, allowF16: true);
        Assert.True(MaxError(expected, ot.AsReadOnlySpan<float>()) < Tolerance);
    }

    [Theory]
    [InlineData(48, 12, 300, 128, true)]
    [InlineData(24, 24, 200, 128, false)]
    [InlineData(10, 5, 97, 64, false)]
    [InlineData(8, 4, 90, 80, true)]
    [InlineData(4, 2, 70, 96, false)]
    public void TokenMajorGqa_MatchesReference(int hq, int hkv, int s, int d, bool masked)
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();
        if (!backend.SupportsTokenMajorGqaAttention) return;
        float[] q = Half(Random(hq * s * d, 5));
        float[] k = Half(Random(hkv * s * d, 6));
        float[] v = Half(Random(hkv * s * d, 7));
        float[]? mask = masked ? Random(s * s, 8, 2f) : null;
        float scale = 1f / MathF.Sqrt(d);
        float[] expected = ToTokenMajor(Reference(q, k, v, mask, 1, hq, hkv, s, s, d, scale), hq, s, d);

        using Tensor qt = FromHost(new TensorShape(s, hq * d), ToTokenMajor(q, hq, s, d));
        using Tensor kt = FromHost(new TensorShape(s, hkv * d), ToTokenMajor(k, hkv, s, d));
        using Tensor vt = FromHost(new TensorShape(s, hkv * d), ToTokenMajor(v, hkv, s, d));
        using Tensor? mt = mask is null ? null : FromHost(new TensorShape(s, s), mask);
        using Tensor ot = new(new TensorShape(s, hq * d), DType.F32);
        backend.ScaledDotProductAttentionTokenMajor(ot, qt, kt, vt, mt, hq, hkv, d, scale, allowF16: true);
        Assert.True(MaxError(expected, ot.AsReadOnlySpan<float>()) < Tolerance);
    }

    private double MaxError(float[] expected, ReadOnlySpan<float> actual)
    {
        double worst = 0.0;
        for (int i = 0; i < expected.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(expected[i] - actual[i]));
        }
        output.WriteLine($"max |Δ| = {worst:E3} over {expected.Length} values");
        return worst;
    }

    /// <summary>Head-major [B,H,S,D] softmax(QKᵀ·scale + mask)·V in double; query head h reads kv head h/(hq/hkv).</summary>
    private static float[] Reference(float[] q, float[] k, float[] v, float[]? mask, int batch, int hq, int hkv, int sq, int skv,
        int d, float scale)
    {
        float[] o = new float[batch * hq * sq * d];
        double[] scores = new double[skv];
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < hq; h++)
            {
                int kvh = h / (hq / hkv);
                for (int i = 0; i < sq; i++)
                {
                    double max = double.NegativeInfinity;
                    for (int j = 0; j < skv; j++)
                    {
                        double dot = 0.0;
                        for (int c = 0; c < d; c++)
                        {
                            dot += q[(((b * hq + h) * sq + i) * d) + c] * (double)k[(((b * hkv + kvh) * skv + j) * d) + c];
                        }
                        scores[j] = (dot * scale) + (mask is null ? 0.0 : mask[(i * skv) + j]);
                        max = Math.Max(max, scores[j]);
                    }
                    double sum = 0.0;
                    for (int j = 0; j < skv; j++)
                    {
                        scores[j] = Math.Exp(scores[j] - max);
                        sum += scores[j];
                    }
                    for (int c = 0; c < d; c++)
                    {
                        double acc = 0.0;
                        for (int j = 0; j < skv; j++)
                        {
                            acc += scores[j] * v[(((b * hkv + kvh) * skv + j) * d) + c];
                        }
                        o[(((b * hq + h) * sq + i) * d) + c] = (float)(acc / sum);
                    }
                }
            }
        }
        return o;
    }

    private static float[] ToTokenMajor(float[] headMajor, int heads, int s, int d)
    {
        float[] t = new float[headMajor.Length];
        for (int h = 0; h < heads; h++)
        {
            for (int i = 0; i < s; i++)
            {
                Array.Copy(headMajor, ((h * s) + i) * d, t, ((i * heads) + h) * d, d);
            }
        }
        return t;
    }

    private static float[] Random(int n, int seed, float amplitude = 1f)
    {
        Random rng = new(seed);
        float[] a = new float[n];
        for (int i = 0; i < n; i++)
        {
            a[i] = (float)((rng.NextDouble() * 2) - 1) * amplitude;
        }
        return a;
    }

    private static float[] Half(float[] a)
    {
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = (float)(System.Half)a[i];
        }
        return a;
    }

    private static Tensor FromHost(TensorShape shape, float[] data)
    {
        Tensor t = new(shape, DType.F32);
        data.CopyTo(t.AsSpan<float>());
        return t;
    }
}
