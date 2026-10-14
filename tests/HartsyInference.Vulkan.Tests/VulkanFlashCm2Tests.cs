using HartsyInference.Core.Backends;
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
    [InlineData(1, 6, 3, 65, 129, 64, true)]
    public void HeadMajor_MatchesReference(int batch, int hq, int hkv, int sq, int skv, int d, bool masked)
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        if (!backend.SupportsTokenMajorGqaAttention) { output.WriteLine("SKIPPED: no cooperative-matrix-2 device"); return; }
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
    [InlineData(10, 5, 97, 64, false)]
    [InlineData(4, 2, 70, 96, false)]
    public void TokenMajorGqa_MatchesReference(int hq, int hkv, int s, int d, bool masked)
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        if (!backend.SupportsTokenMajorGqaAttention) { output.WriteLine("SKIPPED: no cooperative-matrix-2 device"); return; }
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

    /// <summary>Causal prefill through <c>FlashAttention</c> (the path LLM prefill takes) on the cm2 kernel against the
    /// CPU reference. K/V live in a buffer whose sequence capacity exceeds kvLen, as a KV cache does, so the
    /// head stride comes from the capacity while the attended length comes from kvLen.</summary>
    [Theory]
    [InlineData(16, 8, 64, 77, 77, 0, 0, 0)]     // granite-3.0-1b: GQA 2, d 64, tile tail
    [InlineData(14, 2, 64, 100, 100, 0, 0, 0)]   // Qwen2.5-0.5B: GQA 7, d 64
    [InlineData(32, 8, 128, 130, 130, 0, 0, 0)]  // Qwen3-4B class: GQA 4, d 128
    [InlineData(16, 16, 128, 96, 96, 0, 0, 0)]   // OLMoE-1B-7B: MHA, d 128
    [InlineData(32, 8, 64, 70, 70, 0, 0, 17)]    // Llama-3.2-1B: GQA 4, d 64, capacity > kvLen
    [InlineData(32, 8, 64, 70, 200, 130, 0, 0)]  // chunked prefill: query rows at offset 130 over 200 keys
    [InlineData(16, 8, 64, 90, 90, 0, 32, 0)]    // sliding window 32
    [InlineData(16, 8, 64, 150, 150, 0, 48, 9)]  // window plus capacity padding plus a partial tile
    public void CausalPrefill_MatchesReference(int hq, int hkv, int d, int sq, int kvLen, int qOffset, int window, int pad) =>
        CausalPrefillMatchesReference(flashCm2: true, hq, hkv, d, sq, kvLen, qOffset, window, pad);

    /// <summary>The scalar causal path, which <c>numerics.vkFlashCm2Prefill=0</c> keeps for every prefill: the same
    /// reference, on the shapes that cover GQA 7, a chunk at a query offset, and a window over capacity padding.
    /// No cm2 dispatch may run.</summary>
    [Theory]
    [InlineData(14, 2, 64, 100, 100, 0, 0, 0)]   // Qwen2.5-0.5B: GQA 7, d 64
    [InlineData(32, 8, 64, 70, 200, 130, 0, 0)]  // chunked prefill: query rows at offset 130 over 200 keys
    [InlineData(16, 8, 64, 150, 150, 0, 48, 9)]  // window plus capacity padding plus a partial tile
    public void CausalPrefill_FlagOff_UsesScalarPath_MatchesReference(int hq, int hkv, int d, int sq, int kvLen, int qOffset, int window, int pad) =>
        CausalPrefillMatchesReference(flashCm2: false, hq, hkv, d, sq, kvLen, qOffset, window, pad);

    /// <summary>Shared body of the two causal theories: the CPU reference, FlashAttention on the backend that
    /// <c>HARTSY_TEST_VULKAN_DEVICE</c> names (or the default), and a check that the cm2 kernel ran exactly when the flag is on.</summary>
    private void CausalPrefillMatchesReference(bool flashCm2, int hq, int hkv, int d, int sq, int kvLen, int qOffset, int window, int pad)
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        if (flashCm2 && !backend.SupportsTokenMajorGqaAttention) { output.WriteLine("SKIPPED: no cooperative-matrix-2 device"); return; }
        backend.EnableFlashCm2Prefill = flashCm2;
        float scale = 1f / MathF.Sqrt(d);
        float[] q = Half(Random(hq * sq * d, 11));
        float[] k = Half(Random(hkv * kvLen * d, 12));
        float[] v = Half(Random(hkv * kvLen * d, 13));
        float[] expected = new float[hq * sq * d];
        using (Tensor rq = FromHost(new TensorShape(1, hq, sq, d), q))
        using (Tensor rk = FromHost(new TensorShape(1, hkv, kvLen, d), k))
        using (Tensor rv = FromHost(new TensorShape(1, hkv, kvLen, d), v))
        using (Tensor ro = new(new TensorShape(1, hq, sq, d), DType.F32))
        {
            AttentionReference.FlashAttention(ro, rq, rk, rv, kvLen, hq / hkv, causal: true, qOffset, scale,
                slidingWindow: window);
            ro.AsReadOnlySpan<float>().CopyTo(expected);
        }
        int cap = kvLen + pad;
        float[] kPadded = Pad(k, hkv, kvLen, cap, d);
        float[] vPadded = Pad(v, hkv, kvLen, cap, d);
        using Tensor qt = FromHost(new TensorShape(1, hq, sq, d), q);
        using Tensor kt = FromHost(new TensorShape(1, hkv, cap, d), kPadded);
        using Tensor vt = FromHost(new TensorShape(1, hkv, cap, d), vPadded);
        using Tensor ot = new(new TensorShape(1, hq, sq, d), DType.F32);
        long before = backend.FlashCm2PrefillDispatches;
        backend.FlashAttention(ot, qt, kt, vt, kvLen, hq / hkv, causal: true, qOffset, scale, slidingWindow: window);
        Assert.Equal(before + (flashCm2 ? 1 : 0), backend.FlashCm2PrefillDispatches);
        Assert.True(MaxError(expected, ot.AsReadOnlySpan<float>()) < Tolerance);
    }

    private static float[] Pad(float[] dense, int heads, int len, int cap, int d)
    {
        float[] p = new float[heads * cap * d];
        for (int h = 0; h < heads; h++)
        {
            Array.Copy(dense, h * len * d, p, h * cap * d, len * d);
        }
        return p;
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
