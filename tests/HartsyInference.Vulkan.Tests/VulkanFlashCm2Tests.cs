using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
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
    [InlineData(32, 8, 64, 100, 300, 200, 32, 0)]  // chunk at offset 200 over 300 keys, window 32 across key tiles
    [InlineData(16, 8, 64, 70, 200, 130, 48, 0)]   // chunk at offset 130, window 48 spanning a tile boundary
    [InlineData(14, 2, 64, 65, 65, 0, 0, 0)]       // one row past a 64-row tile, full causal
    [InlineData(8, 8, 128, 127, 127, 0, 32, 0)]    // head dim 128, window 32, odd length
    [InlineData(16, 8, 64, 64, 64, 0, 48, 0)]      // exact tile multiples with a window
    [InlineData(16, 8, 64, 200, 200, 0, 0, 0)]     // full causal over a prompt that is not a tile multiple
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

    /// <summary>A prefill above the shape bound takes the scalar kernel: no cm2 dispatch. The shape is 8193 query rows by 8192
    /// keys, one row past the 8K bound. Its first 64 rows are checked against the reference for a 64-row prefill over the
    /// same keys, which is exact for causal attention and keeps the CPU reference cheap.</summary>
    [Fact]
    public void CausalPrefill_AboveTheShapeBound_TakesTheScalarPath()
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        backend.EnableFlashCm2Prefill = true;
        const int hq = 1, hkv = 1, d = 64, sq = 8193, kvLen = 8192, check = 64;
        float scale = 1f / MathF.Sqrt(d);
        float[] q = Half(Random(hq * sq * d, 21));
        float[] k = Half(Random(hkv * kvLen * d, 22));
        float[] v = Half(Random(hkv * kvLen * d, 23));
        float[] expected = new float[hq * check * d];
        using (Tensor rq = FromHost(new TensorShape(1, hq, check, d), q.AsSpan(0, hq * check * d).ToArray()))
        using (Tensor rk = FromHost(new TensorShape(1, hkv, kvLen, d), k))
        using (Tensor rv = FromHost(new TensorShape(1, hkv, kvLen, d), v))
        using (Tensor ro = new(new TensorShape(1, hq, check, d), DType.F32))
        {
            AttentionReference.FlashAttention(ro, rq, rk, rv, kvLen, hq / hkv, causal: true, 0, scale, slidingWindow: 0);
            ro.AsReadOnlySpan<float>().CopyTo(expected);
        }
        using Tensor qt = FromHost(new TensorShape(1, hq, sq, d), q);
        using Tensor kt = FromHost(new TensorShape(1, hkv, kvLen, d), k);
        using Tensor vt = FromHost(new TensorShape(1, hkv, kvLen, d), v);
        using Tensor ot = new(new TensorShape(1, hq, sq, d), DType.F32);
        long before = backend.FlashCm2PrefillDispatches;
        backend.FlashAttention(ot, qt, kt, vt, kvLen, hq / hkv, causal: true, 0, scale, slidingWindow: 0);
        Assert.Equal(before, backend.FlashCm2PrefillDispatches);
        Assert.True(MaxError(expected, ot.AsReadOnlySpan<float>().Slice(0, hq * check * d)) < Tolerance);
    }

    /// <summary>The causal mask is computed inside the kernel, so every layer of a prefill takes the cm2 path on the same
    /// inputs and no host mask is built or kept. A decode step (one query row) still takes the scalar kernel.</summary>
    [Fact]
    public void CausalPrefill_TakesCm2OnEveryLayer_AndDecodeStillTakesTheScalarPath()
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        backend.EnableFlashCm2Prefill = true;
        const int hq = 2, hkv = 2, d = 64, sq = 128, kvLen = 128;
        float scale = 1f / MathF.Sqrt(d);
        using Tensor qt = FromHost(new TensorShape(1, hq, sq, d), Half(Random(hq * sq * d, 31)));
        using Tensor kt = FromHost(new TensorShape(1, hkv, kvLen, d), Half(Random(hkv * kvLen * d, 32)));
        using Tensor vt = FromHost(new TensorShape(1, hkv, kvLen, d), Half(Random(hkv * kvLen * d, 33)));
        using Tensor ot = new(new TensorShape(1, hq, sq, d), DType.F32);
        long before = backend.FlashCm2PrefillDispatches;
        backend.FlashAttention(ot, qt, kt, vt, kvLen, 1, causal: true, 0, scale, slidingWindow: 0);
        backend.FlashAttention(ot, qt, kt, vt, kvLen, 1, causal: true, 0, scale, slidingWindow: 0);
        Assert.Equal(before + 2, backend.FlashCm2PrefillDispatches);
        // A decode step (one query row) does not take the cm2 path.
        using Tensor q1 = FromHost(new TensorShape(1, hq, 1, d), Half(Random(hq * d, 34)));
        using Tensor o1 = new(new TensorShape(1, hq, 1, d), DType.F32);
        backend.FlashAttention(o1, q1, kt, vt, kvLen, 1, causal: true, kvLen - 1, scale, slidingWindow: 0);
        Assert.Equal(before + 2, backend.FlashCm2PrefillDispatches);
    }

    /// <summary>A causal prefill whose first query row lies past the keys (<c>qOffset &gt;= kvLen</c>) is refused by the cm2 gate,
    /// as the decode path refuses it, and takes the scalar kernel: no cm2 dispatch, and the output still matches the CPU reference.</summary>
    [Theory]
    [InlineData(16, 8, 64, 64, 64, 64, 0)]    // the first query row is the first row past the 64 keys
    [InlineData(16, 8, 64, 70, 64, 80, 32)]   // a window over a query offset past the keys
    public void CausalPrefill_QueryOffsetPastTheKeys_TakesTheScalarPath(int hq, int hkv, int d, int sq, int kvLen, int qOffset, int window)
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        backend.EnableFlashCm2Prefill = true;
        float scale = 1f / MathF.Sqrt(d);
        float[] q = Half(Random(hq * sq * d, 41));
        float[] k = Half(Random(hkv * kvLen * d, 42));
        float[] v = Half(Random(hkv * kvLen * d, 43));
        float[] expected = new float[hq * sq * d];
        using (Tensor rq = FromHost(new TensorShape(1, hq, sq, d), q))
        using (Tensor rk = FromHost(new TensorShape(1, hkv, kvLen, d), k))
        using (Tensor rv = FromHost(new TensorShape(1, hkv, kvLen, d), v))
        using (Tensor ro = new(new TensorShape(1, hq, sq, d), DType.F32))
        {
            AttentionReference.FlashAttention(ro, rq, rk, rv, kvLen, hq / hkv, causal: true, qOffset, scale, slidingWindow: window);
            ro.AsReadOnlySpan<float>().CopyTo(expected);
        }
        using Tensor qt = FromHost(new TensorShape(1, hq, sq, d), q);
        using Tensor kt = FromHost(new TensorShape(1, hkv, kvLen, d), k);
        using Tensor vt = FromHost(new TensorShape(1, hkv, kvLen, d), v);
        using Tensor ot = new(new TensorShape(1, hq, sq, d), DType.F32);
        long before = backend.FlashCm2PrefillDispatches;
        backend.FlashAttention(ot, qt, kt, vt, kvLen, hq / hkv, causal: true, qOffset, scale, slidingWindow: window);
        Assert.Equal(before, backend.FlashCm2PrefillDispatches);
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

    /// <summary>The single-stream blocks' attention through <see cref="AttentionLayer.Attend"/>: the token-major config (B.1) gives
    /// the bits of the head-major config's permutes and A.1, the same cm2 kernel read through other strides, with Chroma's
    /// [1, 1, S, S] padding mask too and F32 cast to F16 on both.</summary>
    [Theory]
    [InlineData("flux1-1024", 24, 4608, true, false)]
    [InlineData("flux1-1024-f32", 24, 4608, false, false)]
    [InlineData("chroma-1024", 24, 4107, true, false)]
    [InlineData("chroma-1024-masked", 24, 4107, true, true)]
    [InlineData("flux2-1024", 48, 4608, true, false)]
    [InlineData("hunyuanimage-f32", 28, 4173, false, false)]
    public void SingleStreamAttend_TokenMajorIsBitwiseTheHeadMajorRoute(string name, int h, int s, bool f16, bool masked)
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        if (!backend.SupportsTokenMajorAttention) { output.WriteLine("SKIPPED: no cooperative-matrix-2 device"); return; }
        const int d = 128;
        DType dtype = f16 ? DType.F16 : DType.F32;
        float[] q = Half(Random(s * h * d, 41)), k = Half(Random(s * h * d, 42)), v = Half(Random(s * h * d, 43));
        float[]? bias = masked ? PaddingMask(s, txt: 11, kept: 4) : null;
        TensorShape operand = new(1, s, h, d), flat = new(1, s, h * d);

        float[] Run(AttentionRoute route)
        {
            AttentionLayer layer = new(new AttentionLayerConfig
            {
                Heads = h, KvHeads = h, HeadDim = d, AllowF16 = true, TokenMajorRoute = route, HeadMajorForSage = true,
            });
            using Tensor? mask = bias is null ? null : FromHost(new TensorShape(1, 1, s, s), bias);
            using Tensor o = layer.Attend(backend, Operand(q, operand, dtype), Operand(k, operand, dtype), Operand(v, operand, dtype), flat,
                new AttentionCallContext { Bias = mask });
            Assert.Equal(flat, o.Shape);
            if (o.DType == DType.F32) return o.AsReadOnlySpan<float>().ToArray();
            using Tensor wide = o.CastTo(DType.F32);
            return wide.AsReadOnlySpan<float>().ToArray();
        }

        float[] fromHeadMajor = Run(AttentionRoute.Dense);
        string? headMajorRoute = VulkanBackend.LastAttentionRoute;
        float[] fromTokenMajor = Run(AttentionRoute.TokenMajor);
        string? tokenMajorRoute = VulkanBackend.LastAttentionRoute;
        int differing = 0;
        for (int i = 0; i < fromHeadMajor.Length; i++)
            if (BitConverter.SingleToInt32Bits(fromHeadMajor[i]) != BitConverter.SingleToInt32Bits(fromTokenMajor[i])) differing++;
        output.WriteLine($"{name}: {headMajorRoute} vs {tokenMajorRoute}, {differing} of {fromHeadMajor.Length} elements differ");
        Assert.Equal("A.1", headMajorRoute);
        Assert.Equal("B.1", tokenMajorRoute);
        Assert.Equal(0, differing);
    }

    /// <summary>The double-stream blocks' joint call through <see cref="AttentionLayer.AttendFlat"/>: the token-major config reads the
    /// concatenated [1, S, H·D] rows through B.1 with the bits of the head-major config's permutes and A.1, with Chroma's padding
    /// mask too and F32 cast to F16 on both. AuraFlow's head dim 256, which no cm2 row takes, permutes into the same head-major row.</summary>
    [Theory]
    [InlineData("flux1-1024", 24, 128, 4608, true, false, "A.1", "B.1")]
    [InlineData("flux1-1024-f32", 24, 128, 4608, false, false, "A.1", "B.1")]
    [InlineData("chroma-1024-masked", 24, 128, 4107, true, true, "A.1", "B.1")]
    [InlineData("flux2-1024", 48, 128, 4608, true, false, "A.1", "B.1")]
    [InlineData("qwenimage-1024", 24, 128, 4115, false, false, "A.1", "B.1")]
    [InlineData("hunyuanimage-f32", 28, 128, 1101, false, false, "A.1", "B.1")]
    [InlineData("hidream-f32", 20, 128, 4352, false, false, "A.1", "B.1")]
    [InlineData("auraflow-f32", 12, 256, 4352, false, false, "A.3", "B.2/A.3")]
    public void JointAttendFlat_TokenMajorIsBitwiseTheHeadMajorRoute(string name, int h, int d, int s, bool f16, bool masked,
        string headMajorRoute, string tokenMajorRoute)
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        if (!backend.SupportsTokenMajorAttention) { output.WriteLine("SKIPPED: no cooperative-matrix-2 device"); return; }
        DType dtype = f16 ? DType.F16 : DType.F32;
        float[] q = Half(Random(s * h * d, 51)), k = Half(Random(s * h * d, 52)), v = Half(Random(s * h * d, 53));
        float[]? bias = masked ? PaddingMask(s, txt: 11, kept: 4) : null;
        TensorShape rows = new(1, s, h * d);

        float[] Run(AttentionRoute route)
        {
            AttentionLayer layer = new(new AttentionLayerConfig
            {
                Heads = h, KvHeads = h, HeadDim = d, AllowF16 = true, TokenMajorRoute = route, HeadMajorForSage = true,
            });
            using Tensor? mask = bias is null ? null : FromHost(new TensorShape(1, 1, s, s), bias);
            using Tensor o = layer.AttendFlat(backend, Operand(q, rows, dtype), Operand(k, rows, dtype), Operand(v, rows, dtype), rows,
                new AttentionCallContext { Bias = mask });
            Assert.Equal(rows, o.Shape);
            if (o.DType == DType.F32) return o.AsReadOnlySpan<float>().ToArray();
            using Tensor wide = o.CastTo(DType.F32);
            return wide.AsReadOnlySpan<float>().ToArray();
        }

        float[] fromHeadMajor = Run(AttentionRoute.Dense);
        string? headMajor = VulkanBackend.LastAttentionRoute;
        float[] fromTokenMajor = Run(AttentionRoute.TokenMajor);
        string? tokenMajor = VulkanBackend.LastAttentionRoute;
        int differing = 0;
        for (int i = 0; i < fromHeadMajor.Length; i++)
            if (BitConverter.SingleToInt32Bits(fromHeadMajor[i]) != BitConverter.SingleToInt32Bits(fromTokenMajor[i])) differing++;
        output.WriteLine($"{name}: {headMajor} vs {tokenMajor}, {differing} of {fromHeadMajor.Length} elements differ");
        Assert.Equal(headMajorRoute, headMajor);
        Assert.Equal(tokenMajorRoute, tokenMajor);
        Assert.Equal(0, differing);
    }

    /// <summary>Wan's self-attention rows, the attention call alone: the token-major config reads F32 [S, H·D] Q, K and V through B.1 with
    /// the bits of the head-major config's permutes and A.1, at Wan 2.2 5B's and Wan 2.1 14B's lengths, with Matrix-Game 2's block
    /// mask, and for a context-parallel rank's local queries over the whole sequence's keys.</summary>
    [Theory]
    [InlineData("wan5b-49f", 24, 5070, 5070, false)]
    [InlineData("wan21-14b", 40, 4480, 4480, false)]
    [InlineData("matrixgame2-masked", 12, 2640, 2640, true)]
    [InlineData("wan5b-cp-rank", 24, 2730, 5070, false)]
    public void WanRows_TokenMajorIsBitwiseTheHeadMajorRoute(string name, int h, int sq, int skv, bool masked)
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        if (!backend.SupportsTokenMajorAttention) { output.WriteLine("SKIPPED: no cooperative-matrix-2 device"); return; }
        const int d = 128;
        float[] q = Half(Random(sq * h * d, 61)), k = Half(Random(skv * h * d, 62)), v = Half(Random(skv * h * d, 63));
        float[]? bias = masked ? BlockCausalMask(sq, frames: 3) : null;
        TensorShape qRows = new(sq, h * d), kvRows = new(skv, h * d);

        float[] Run(AttentionRoute route)
        {
            AttentionLayer layer = new(new AttentionLayerConfig
            {
                Heads = h, KvHeads = h, HeadDim = d, AllowF16 = true, TokenMajorRoute = route, HeadMajorForSage = true,
            });
            using Tensor? mask = bias is null ? null : FromHost(new TensorShape(1, 1, sq, skv), bias);
            using Tensor o = layer.AttendFlat(backend, Operand(q, qRows, DType.F32), Operand(k, kvRows, DType.F32), Operand(v, kvRows, DType.F32),
                qRows, new AttentionCallContext { Bias = mask });
            Assert.Equal(qRows, o.Shape);
            return o.AsReadOnlySpan<float>().ToArray();
        }

        float[] fromHeadMajor = Run(AttentionRoute.Dense);
        string? headMajor = VulkanBackend.LastAttentionRoute;
        float[] fromTokenMajor = Run(AttentionRoute.TokenMajor);
        string? tokenMajor = VulkanBackend.LastAttentionRoute;
        int differing = 0;
        for (int i = 0; i < fromHeadMajor.Length; i++)
            if (BitConverter.SingleToInt32Bits(fromHeadMajor[i]) != BitConverter.SingleToInt32Bits(fromTokenMajor[i])) differing++;
        output.WriteLine($"{name}: {headMajor} vs {tokenMajor}, {differing} of {fromHeadMajor.Length} elements differ");
        Assert.Equal("A.1", headMajor);
        Assert.Equal("B.1", tokenMajor);
        Assert.Equal(0, differing);
    }

    /// <summary>The token-major rows read one sequence, so a batched token-major call plans no row there and the layer keeps a batch
    /// head-major on Vulkan; one sequence still plans the cm2 row.</summary>
    [Fact]
    public void ABatchedTokenMajorCall_PlansNoRow()
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        if (!backend.SupportsTokenMajorAttention) { output.WriteLine("SKIPPED: no cooperative-matrix-2 device"); return; }
        const int h = 10, d = 64, s = 256;
        AttentionCall Call(int batch)
        {
            TensorShape rows = new(batch, s, h, d);
            return AttentionCall.FromTokenMajor(new Tensor(new TensorShape(batch, s, h * d), DType.F32), new Tensor(rows, DType.F32),
                new Tensor(rows, DType.F32), new Tensor(rows, DType.F32), null, h, h, d, 0.125f, allowF16: true, batch);
        }

        AttentionCall batched = Call(2), single = Call(1);
        Assert.Null(backend.PlanAttention(in batched).Row);
        Assert.Equal("cm2.flash", backend.PlanAttention(in single).RowId);
    }

    /// <summary>Matrix-Game 2's additive block-causal mask over equal frames: a query sees its own frame and the earlier ones.</summary>
    private static float[] BlockCausalMask(int s, int frames)
    {
        int perFrame = s / frames;
        float[] mask = new float[(long)s * s];
        for (int i = 0; i < s; i++)
            for (int j = 0; j < s; j++)
                mask[(long)i * s + j] = j / perFrame <= i / perFrame ? 0f : -1e9f;
        return mask;
    }

    /// <summary>Chroma's additive mask over a [text; image] sequence: a pair is kept only when both tokens are, and the text past
    /// <paramref name="kept"/> is padding.</summary>
    private static float[] PaddingMask(int s, int txt, int kept)
    {
        float[] mask = new float[(long)s * s];
        for (int i = 0; i < s; i++)
            for (int j = 0; j < s; j++)
                mask[(long)i * s + j] = (i >= kept && i < txt) || (j >= kept && j < txt) ? -1e30f : 0f;
        return mask;
    }

    /// <summary>A host tensor of <paramref name="dtype"/> holding <paramref name="values"/>, already F16-exact.</summary>
    private static Tensor Operand(float[] values, TensorShape shape, DType dtype)
    {
        Tensor f32 = FromHost(shape, values);
        if (dtype == DType.F32) return f32;
        try { return f32.CastTo(dtype); }
        finally { f32.Dispose(); }
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
