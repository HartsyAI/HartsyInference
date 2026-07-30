using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>Correctness tests for Phase 6's LLM decode-graph device state: the small persistent
/// "control" buffers (token id, position, history, counter) and the kernels that read/write them
/// (<c>RopeApplyDecodeStep</c>, <c>EmbedGatherDecodeStep</c>, <c>ArgMaxInto</c>,
/// <c>AppendTokenHistoryStep</c>, <c>ApplyRepetitionPenaltyStep</c>) — the leaf ops a future
/// <c>VulkanStepGraph</c> replay will drive. Each is checked against a from-scratch CPU reference matching
/// the documented math in <c>IBackend.cs</c> / <c>CudaBackend.cs</c>'s decode-graph region, not against the
/// Vulkan implementation itself.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanDecodeGraphTests
{
    private readonly ITestOutputHelper _out;
    public VulkanDecodeGraphTests(ITestOutputHelper output) => _out = output;

    private static bool VulkanAvailable()
    {
        try { using VulkanInstance i = new(); return i.EnumeratePhysicalDevices().Length > 0; }
        catch { return false; }
    }

    [Fact]
    public void DeviceTokenId_WriteThenRead_RoundTrips()
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        ulong handle = backend.AllocDeviceTokenId();
        try
        {
            Assert.NotEqual(0UL, handle);
            backend.WriteDeviceTokenId(handle, 12345);
            Assert.Equal(12345, backend.ReadDeviceTokenId(handle));
            backend.WriteDeviceTokenId(handle, 0);
            Assert.Equal(0, backend.ReadDeviceTokenId(handle));
        }
        finally { backend.FreeDeviceTokenId(handle); }
    }

    [Theory]
    [InlineData(64, 8, 8, false)]   // full rotary, split-half
    [InlineData(64, 8, 8, true)]    // full rotary, interleaved
    [InlineData(64, 16, 8, false)]  // partial rotary (rotaryDim < headDim), split-half
    [InlineData(64, 16, 8, true)]   // partial rotary, interleaved
    public unsafe void RopeApplyDecodeStep_MatchesCpuReference(int maxPos, int headDim, int rotaryDim, bool interleaved)
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        const int numHeads = 4;
        const int position = 17;

        (Tensor cos, Tensor sin) = backend.BuildRopeTableDevice(maxPos, headDim, rotaryDim, 10000f, RopeScaling.None, splitHalfPartial: !interleaved);
        Tensor x = new(new TensorShape(1, numHeads, 1, headDim), DType.F32);
        ulong pos = backend.AllocDevicePos();
        try
        {
            Random rng = new(42);
            Span<float> xs = x.AsSpan<float>();
            float[] original = new float[xs.Length];
            for (int i = 0; i < xs.Length; i++) { xs[i] = (float)(rng.NextDouble() * 2 - 1); original[i] = xs[i]; }

            backend.WriteDevicePos(pos, kvLen: position + 1, qOffset: position);
            backend.RopeApplyDecodeStep(x, cos, sin, rotaryDim, interleaved, pos);

            // CPU reference: same math as ApplyRopeSingle/ApplyRopeInterleaved, evaluated at `position`
            // directly (not read from the cos/sin table, so this also validates BuildRopeTableDevice's
            // table content, not just the decode-step indexing).
            (double[] invFreq, double mscale) = RopeFrequencyBuilder.Build(rotaryDim, 10000.0, RopeScaling.None, maxPos);
            float[] expected = new float[original.Length];
            Array.Copy(original, expected, original.Length);
            int half = rotaryDim / 2;
            for (int h = 0; h < numHeads; h++)
            {
                int baseOff = h * headDim;
                for (int i = 0; i < half; i++)
                {
                    double angle = position * invFreq[i];
                    float c = (float)(Math.Cos(angle) * mscale);
                    float s = (float)(Math.Sin(angle) * mscale);
                    if (interleaved)
                    {
                        float xe = original[baseOff + 2 * i];
                        float xo = original[baseOff + 2 * i + 1];
                        expected[baseOff + 2 * i] = xe * c - xo * s;
                        expected[baseOff + 2 * i + 1] = xo * c + xe * s;
                    }
                    else
                    {
                        float lower = original[baseOff + i];
                        float upper = original[baseOff + i + half];
                        expected[baseOff + i] = lower * c - upper * s;
                        expected[baseOff + i + half] = upper * c + lower * s;
                    }
                }
            }

            ReadOnlySpan<float> actual = x.AsReadOnlySpan<float>();
            for (int i = 0; i < expected.Length; i++)
                Assert.InRange(actual[i] - expected[i], -5e-4f, 5e-4f);
        }
        finally { x.Dispose(); cos.Dispose(); sin.Dispose(); backend.FreeDevicePos(pos); }
    }

    [Fact]
    public unsafe void EmbedGatherDecodeStep_GathersCorrectRow()
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        const int vocab = 37, hidden = 64, tokenId = 19;

        Tensor embed = new(new TensorShape(vocab, hidden), DType.F32);
        Tensor output = new(new TensorShape(1, 1, hidden), DType.F32);
        ulong tokBuf = backend.AllocDeviceTokenId();
        try
        {
            Random rng = new(7);
            Span<float> es = embed.AsSpan<float>();
            for (int i = 0; i < es.Length; i++) es[i] = (float)rng.NextDouble();

            backend.WriteDeviceTokenId(tokBuf, tokenId);
            backend.EmbedGatherDecodeStep(output, embed, tokBuf);

            ReadOnlySpan<float> outS = output.AsReadOnlySpan<float>();
            for (int i = 0; i < hidden; i++)
                Assert.Equal(es[tokenId * hidden + i], outS[i], 5);
        }
        finally { embed.Dispose(); output.Dispose(); backend.FreeDeviceTokenId(tokBuf); }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(300)]    // > WGSIZE (256), exercises the strided multi-pass scan
    [InlineData(4096)]   // realistic vocab-size scale
    public unsafe void ArgMaxInto_MatchesCpuArgmax(int c)
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();

        Tensor logits = new(new TensorShape(1, c), DType.F32);
        ulong tokBuf = backend.AllocDeviceTokenId();
        try
        {
            Random rng = new(3);
            Span<float> ls = logits.AsSpan<float>();
            // Distinct values (no ties) — tie-breaking across the reduction's cross-thread order isn't
            // guaranteed to match a first-index-wins CPU argmax, a documented, measure-zero scope boundary.
            float[] shuffled = Enumerable.Range(0, c).Select(i => (float)i).OrderBy(_ => rng.Next()).ToArray();
            for (int i = 0; i < c; i++) ls[i] = shuffled[i] * 0.01f;

            int expectedIdx = 0;
            for (int i = 1; i < c; i++) if (ls[i] > ls[expectedIdx]) expectedIdx = i;

            backend.ArgMaxInto(tokBuf, logits);
            int actualIdx = backend.ReadDeviceTokenId(tokBuf);

            _out.WriteLine($"C={c}: expected={expectedIdx} actual={actualIdx}");
            Assert.Equal(expectedIdx, actualIdx);
        }
        finally { logits.Dispose(); backend.FreeDeviceTokenId(tokBuf); }
    }

    [Fact]
    public void AppendTokenHistoryStep_AppendsAndIncrementsCounter()
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        const int capacity = 8;

        ulong history = backend.AllocDeviceHistory(capacity);
        ulong counter = backend.AllocDeviceCounter();
        ulong tokBuf = backend.AllocDeviceTokenId();
        try
        {
            backend.WriteDeviceCounter(counter, 0);
            int[] tokens = { 5, 12, 5, 99 };
            foreach (int t in tokens)
            {
                backend.WriteDeviceTokenId(tokBuf, t);
                backend.AppendTokenHistoryStep(history, counter, tokBuf);
            }
            // ReadDeviceTokenId reads any 1-int scalar control buffer by handle, not just a "token id"
            // buffer specifically — reused here to read the counter's post-append value.
            Assert.Equal(tokens.Length, backend.ReadDeviceTokenId(counter));
        }
        finally
        {
            backend.FreeDeviceHistory(history);
            backend.FreeDeviceCounter(counter);
            backend.FreeDeviceTokenId(tokBuf);
        }
    }

    /// <summary>Applies penalty over a history built via <see cref="AppendTokenHistoryStep"/> (not a
    /// hand-written buffer) so this also exercises the append path end-to-end, then checks against the HF
    /// convention (divide positive / multiply negative) with the SAME compounding-on-repeat semantics as
    /// <c>HartsyInference.LLM.Sampling.RepetitionPenaltyStep</c> — repeated tokens divide multiple times.</summary>
    [Fact]
    public void ApplyRepetitionPenaltyStep_MatchesHfConventionWithRepeats()
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        const int vocab = 32;
        const float penalty = 1.3f;
        int[] tokens = { 3, 10, 3, 3, 25 };   // token 3 repeats 3x — must compound

        Tensor logits = new(new TensorShape(1, vocab), DType.F32);
        ulong history = backend.AllocDeviceHistory(tokens.Length);
        ulong counter = backend.AllocDeviceCounter();
        ulong tokBuf = backend.AllocDeviceTokenId();
        try
        {
            Random rng = new(11);
            Span<float> ls = logits.AsSpan<float>();
            for (int i = 0; i < vocab; i++) ls[i] = (float)(rng.NextDouble() * 4 - 2);   // mix of +/-
            float[] expected = new float[vocab];
            Array.Copy(ls.ToArray(), expected, vocab);

            backend.WriteDeviceCounter(counter, 0);
            foreach (int t in tokens)
            {
                backend.WriteDeviceTokenId(tokBuf, t);
                backend.AppendTokenHistoryStep(history, counter, tokBuf);
            }
            foreach (int t in tokens)   // CPU reference: sequential compounding, same order
            {
                float logit = expected[t];
                expected[t] = logit > 0f ? logit / penalty : logit * penalty;
            }

            backend.ApplyRepetitionPenaltyStep(logits, history, counter, penalty);

            ReadOnlySpan<float> actual = logits.AsReadOnlySpan<float>();
            for (int i = 0; i < vocab; i++)
                Assert.InRange(actual[i] - expected[i], -1e-4f, 1e-4f);
        }
        finally
        {
            logits.Dispose();
            backend.FreeDeviceHistory(history);
            backend.FreeDeviceCounter(counter);
            backend.FreeDeviceTokenId(tokBuf);
        }
    }

    /// <summary>Regression gate for the KvCacheAppendDev bug an advisor review caught before this shipped:
    /// the caller (GenericTransformer.ForwardGraphDecodeStep) passes a literal <c>offset: 0</c> placeholder
    /// on every call — the REAL write slot must come from devicePos. A passthrough that forwarded the
    /// placeholder unchanged would write every decode step to slot 0, silently corrupting the KV cache the
    /// instant GraphDecodeSupported is on. Appends 3 steps at devicePos-driven slots 0,1,2 (NOT the
    /// `offset` argument, deliberately left wrong/stale to prove it's ignored) and checks the buffer ends
    /// up with all 3 slots populated, matching a plain host-loop reference.</summary>
    [Fact]
    public unsafe void KvCacheAppendDev_UsesDevicePositionNotHostOffset()
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        const int heads = 2, maxSeq = 8, headDim = 4;

        Tensor cache = new(new TensorShape(1, heads, maxSeq, headDim), DType.F32);
        ulong pos = backend.AllocDevicePos();
        try
        {
            Span<float> cs = cache.AsSpan<float>();
            for (int i = 0; i < cs.Length; i++) cs[i] = -1f;   // sentinel: untouched slots stay -1

            Random rng = new(1);
            float[][] steps = new float[3][];
            for (int step = 0; step < 3; step++)
            {
                Tensor newKv = new(new TensorShape(1, heads, 1, headDim), DType.F32);
                Span<float> ns = newKv.AsSpan<float>();
                steps[step] = new float[heads * headDim];
                for (int i = 0; i < ns.Length; i++) { float v = (float)rng.NextDouble(); ns[i] = v; steps[step][i] = v; }

                backend.WriteDevicePos(pos, kvLen: step + 1, qOffset: step);
                // `offset: 999` is deliberately wrong — if this were used instead of devicePos, every
                // step would write out of bounds (or all to the same wrong slot), not to slot `step`.
                backend.KvCacheAppendDev(cache, newKv, offset: 999, pos);
                newKv.Dispose();
            }

            ReadOnlySpan<float> actual = cache.AsReadOnlySpan<float>();
            for (int step = 0; step < 3; step++)
            {
                for (int h = 0; h < heads; h++)
                {
                    for (int d = 0; d < headDim; d++)
                    {
                        long idx = ((long)h * maxSeq + step) * headDim + d;
                        Assert.Equal(steps[step][h * headDim + d], actual[(int)idx], 5);
                    }
                }
            }
            // Slot 3 (never appended to) must still hold the sentinel — proves this didn't just fill
            // every slot with the last value or otherwise paper over a wrong-index bug.
            for (int h = 0; h < heads; h++)
                for (int d = 0; d < headDim; d++)
                    Assert.Equal(-1f, actual[(int)(((long)h * maxSeq + 3) * headDim + d)]);
        }
        finally { cache.Dispose(); backend.FreeDevicePos(pos); }
    }

    [Theory]
    [InlineData(0)]   // first decode step: no prior context
    [InlineData(9)]   // mid-generation: kvLen < the K/V buffer's over-allocated maxSeq
    public unsafe void FlashAttentionDev_MatchesCpuReference(int priorLen)
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        const int batch = 1, hq = 8, hkv = 2, sq = 1, maxSeq = 32, headDim = 32;
        int kvLen = priorLen + 1;   // this step's own K/V was already appended before attending
        float scale = 1f / MathF.Sqrt(headDim);

        float[] q = VulkanBackendSmokeTests.FillRandom(batch * hq * sq * headDim, 51);
        float[] kFull = VulkanBackendSmokeTests.FillRandom(batch * hkv * maxSeq * headDim, 52);
        float[] vFull = VulkanBackendSmokeTests.FillRandom(batch * hkv * maxSeq * headDim, 53);
        float[] kValid = new float[batch * hkv * kvLen * headDim];
        float[] vValid = new float[batch * hkv * kvLen * headDim];
        for (int h = 0; h < hkv; h++)
        {
            Array.Copy(kFull, h * maxSeq * headDim, kValid, h * kvLen * headDim, kvLen * headDim);
            Array.Copy(vFull, h * maxSeq * headDim, vValid, h * kvLen * headDim, kvLen * headDim);
        }
        float[] expected = VulkanBackendSmokeTests.CpuFlashReference(q, kValid, vValid, null, batch, hq, hkv, sq, kvLen, headDim, scale, causal: false, qOffset: 0, slidingWindow: 0);

        Tensor qT = new(new TensorShape(batch, hq, sq, headDim), DType.F32);
        Tensor kT = new(new TensorShape(batch, hkv, maxSeq, headDim), DType.F32);
        Tensor vT = new(new TensorShape(batch, hkv, maxSeq, headDim), DType.F32);
        Tensor oT = new(new TensorShape(batch, hq, sq, headDim), DType.F32);
        q.CopyTo(qT.AsSpan<float>()); kFull.CopyTo(kT.AsSpan<float>()); vFull.CopyTo(vT.AsSpan<float>());
        ulong pos = backend.AllocDevicePos();
        try
        {
            // Deliberately wrong host placeholders (0,0) — mirrors GenericTransformer's real call
            // convention and proves devicePos, not these arguments, drives the kernel.
            backend.WriteDevicePos(pos, kvLen: kvLen, qOffset: 0);
            backend.FlashAttentionDev(oT, qT, kT, vT, kvLen: 0, kvGroup: hq / hkv, causal: false, qOffset: 0, scale, pos);

            ReadOnlySpan<float> oS = oT.AsReadOnlySpan<float>();
            for (int i = 0; i < expected.Length; i++)
                Assert.InRange(oS[i] - expected[i], -2e-3f, 2e-3f);
        }
        finally { qT.Dispose(); kT.Dispose(); vT.Dispose(); oT.Dispose(); backend.FreeDevicePos(pos); }
    }
}
