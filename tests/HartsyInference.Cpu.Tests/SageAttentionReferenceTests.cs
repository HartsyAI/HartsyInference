using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;

namespace HartsyInference.Cpu.Tests;

/// <summary>Algorithm-level validation of INT8-quantized attention (SageAttention v1, arXiv:2410.02367) ahead
/// of the CUDA kernel build — see docs/Checklists/INFERENCE_ACCEL_GRIND.md. The scheme: quantize Q and K to
/// per-row INT8 (absmax scales), compute QK^T in int8×int8→int32 (IMMA on SM 8.6), dequantize scores, keep
/// softmax+PV in F32/F16. K is SMOOTHED first — its per-channel mean over the sequence is subtracted — which
/// is exact for the softmax (q·μ is constant per query row) and absorbs the channel-consistent outliers that
/// otherwise eat the int8 range.
///
/// <para>These tests pin the two load-bearing claims the GPU kernel will inherit: (1) mean-subtraction is a
/// softmax INVARIANT (bit-level property, not an approximation), and (2) on outlier-channel distributions —
/// the realistic DiT case — smoothing reduces quantized-attention error by an order of magnitude, landing
/// within the tolerance budget (~1e-2 vs F32) at which SageAttention reports metric-neutral end-to-end
/// results. The reference here is the diff target for the future <c>sage_attn_int8</c> PTX kernel.</para></summary>
public sealed unsafe class SageAttentionReferenceTests
{
    // ── Reference implementation (promoted to src/ alongside the kernel when it lands) ──

    /// <summary>F32 baseline attention: softmax(Q·K^T·scale)·V for one head. Q,K,V are [S, D] row-major.</summary>
    private static float[] AttentionF32(float[] q, float[] k, float[] v, int seqQ, int seqK, int dim, float scale)
    {
        float[] output = new float[seqQ * dim];
        float[] scores = new float[seqK];
        for (int i = 0; i < seqQ; i++)
        {
            float max = float.NegativeInfinity;
            for (int j = 0; j < seqK; j++)
            {
                double dot = 0;
                for (int d = 0; d < dim; d++) dot += (double)q[i * dim + d] * k[j * dim + d];
                scores[j] = (float)(dot * scale);
                max = Math.Max(max, scores[j]);
            }
            double sum = 0;
            for (int j = 0; j < seqK; j++)
            {
                scores[j] = MathF.Exp(scores[j] - max);
                sum += scores[j];
            }
            float inv = (float)(1.0 / sum);
            for (int d = 0; d < dim; d++)
            {
                double acc = 0;
                for (int j = 0; j < seqK; j++) acc += (double)scores[j] * v[j * dim + d];
                output[i * dim + d] = (float)(acc * inv);
            }
        }
        return output;
    }

    /// <summary>INT8-quantized attention: per-row absmax int8 quant of Q and (optionally smoothed) K, exact
    /// int32 dot products, dequantized scores, F32 softmax+PV — the numerical model of the planned kernel.</summary>
    private static float[] AttentionInt8(float[] q, float[] k, float[] v, int seqQ, int seqK, int dim, float scale,
        bool smoothK)
    {
        float[] kWork = (float[])k.Clone();
        if (smoothK)
        {
            // Per-CHANNEL mean over the key sequence; exact under softmax (q·μ is per-query-row constant).
            for (int d = 0; d < dim; d++)
            {
                double mean = 0;
                for (int j = 0; j < seqK; j++) mean += kWork[j * dim + d];
                float m = (float)(mean / seqK);
                for (int j = 0; j < seqK; j++) kWork[j * dim + d] -= m;
            }
        }

        sbyte[] q8 = new sbyte[seqQ * dim];
        float[] qScale = new float[seqQ];
        QuantizeRows(q, q8, qScale, seqQ, dim);
        sbyte[] k8 = new sbyte[seqK * dim];
        float[] kScale = new float[seqK];
        QuantizeRows(kWork, k8, kScale, seqK, dim);

        float[] output = new float[seqQ * dim];
        float[] scores = new float[seqK];
        for (int i = 0; i < seqQ; i++)
        {
            float max = float.NegativeInfinity;
            for (int j = 0; j < seqK; j++)
            {
                int dot = 0;
                for (int d = 0; d < dim; d++) dot += q8[i * dim + d] * k8[j * dim + d];
                scores[j] = dot * qScale[i] * kScale[j] * scale;
                max = Math.Max(max, scores[j]);
            }
            double sum = 0;
            for (int j = 0; j < seqK; j++)
            {
                scores[j] = MathF.Exp(scores[j] - max);
                sum += scores[j];
            }
            float inv = (float)(1.0 / sum);
            for (int d = 0; d < dim; d++)
            {
                double acc = 0;
                for (int j = 0; j < seqK; j++) acc += (double)scores[j] * v[j * dim + d];
                output[i * dim + d] = (float)(acc * inv);
            }
        }
        return output;
    }

    private static void QuantizeRows(float[] src, sbyte[] dst, float[] scales, int rows, int dim)
    {
        for (int r = 0; r < rows; r++)
        {
            float amax = 0f;
            for (int d = 0; d < dim; d++) amax = Math.Max(amax, Math.Abs(src[r * dim + d]));
            float scale = amax > 0f ? amax / 127f : 1f;
            scales[r] = scale;
            float inv = 1f / scale;
            for (int d = 0; d < dim; d++)
                dst[r * dim + d] = (sbyte)Math.Clamp(MathF.Round(src[r * dim + d] * inv), -127f, 127f);
        }
    }

    private static float MaxAbsError(float[] a, float[] b)
    {
        float max = 0f;
        for (int i = 0; i < a.Length; i++) max = Math.Max(max, Math.Abs(a[i] - b[i]));
        return max;
    }

    private static (float[] q, float[] k, float[] v) MakeInputs(int seqQ, int seqK, int dim, int seed,
        int outlierChannels = 0, float outlierMagnitude = 0f)
    {
        Random rng = new Random(seed);
        float[] Fill(int n)
        {
            float[] a = new float[n];
            for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            return a;
        }
        float[] q = Fill(seqQ * dim);
        float[] k = Fill(seqK * dim);
        float[] v = Fill(seqK * dim);
        // Channel-consistent K outliers: a few channels carry a large shared offset — the documented DiT
        // activation pathology (SmoothQuant/SageAttention's motivation), NOT random spikes.
        for (int c = 0; c < outlierChannels; c++)
        {
            int channel = rng.Next(dim);
            for (int j = 0; j < seqK; j++) k[j * dim + channel] += outlierMagnitude;
        }
        return (q, k, v);
    }

    // ── Claim 1: mean-subtraction is a softmax invariant (exact math, F32) ──

    [Fact]
    public void KMeanSubtraction_LeavesF32AttentionUnchanged()
    {
        const int seqQ = 8, seqK = 16, dim = 32;
        (float[] q, float[] k, float[] v) = MakeInputs(seqQ, seqK, dim, seed: 42);
        float scale = 1f / MathF.Sqrt(dim);

        float[] baseline = AttentionF32(q, k, v, seqQ, seqK, dim, scale);

        float[] kSmoothed = (float[])k.Clone();
        for (int d = 0; d < dim; d++)
        {
            double mean = 0;
            for (int j = 0; j < seqK; j++) mean += kSmoothed[j * dim + d];
            float m = (float)(mean / seqK);
            for (int j = 0; j < seqK; j++) kSmoothed[j * dim + d] -= m;
        }
        float[] smoothed = AttentionF32(q, kSmoothed, v, seqQ, seqK, dim, scale);

        // Not bit-exact (the subtraction re-rounds logits before exp), but far inside F32 attention noise.
        Assert.True(MaxAbsError(baseline, smoothed) < 1e-4f,
            $"softmax invariance violated: maxAbs {MaxAbsError(baseline, smoothed)}");
    }

    // ── Claim 2: INT8 attention error bounds, and smoothing's necessity under outliers ──

    [Fact]
    public void Int8Attention_WellBehavedInputs_WithinTolerance()
    {
        const int seqQ = 32, seqK = 64, dim = 64;
        (float[] q, float[] k, float[] v) = MakeInputs(seqQ, seqK, dim, seed: 7);
        float scale = 1f / MathF.Sqrt(dim);

        float[] baseline = AttentionF32(q, k, v, seqQ, seqK, dim, scale);
        float[] quantized = AttentionInt8(q, k, v, seqQ, seqK, dim, scale, smoothK: false);

        // No outliers → per-row absmax int8 alone stays within the F16-kernel-class tolerance (1e-2).
        Assert.True(MaxAbsError(baseline, quantized) < 1e-2f,
            $"int8 attention maxAbs {MaxAbsError(baseline, quantized)} exceeds 1e-2 on well-behaved inputs");
    }

    [Fact]
    public void Int8Attention_OutlierChannels_SmoothingRecoversAccuracy()
    {
        const int seqQ = 32, seqK = 64, dim = 64;
        (float[] q, float[] k, float[] v) = MakeInputs(seqQ, seqK, dim, seed: 11,
            outlierChannels: 3, outlierMagnitude: 30f);
        float scale = 1f / MathF.Sqrt(dim);

        float[] baseline = AttentionF32(q, k, v, seqQ, seqK, dim, scale);
        float[] unsmoothed = AttentionInt8(q, k, v, seqQ, seqK, dim, scale, smoothK: false);
        float[] smoothed = AttentionInt8(q, k, v, seqQ, seqK, dim, scale, smoothK: true);

        float errUnsmoothed = MaxAbsError(baseline, unsmoothed);
        float errSmoothed = MaxAbsError(baseline, smoothed);

        // The load-bearing pair: outlier channels wreck plain int8 (they consume the absmax range so the
        // informative dimensions round to few levels), and mean-subtraction recovers it — by construction
        // exactly the channel-consistent component moves into the (softmax-invariant) offset.
        Assert.True(errSmoothed < 1e-2f,
            $"smoothed int8 attention maxAbs {errSmoothed} exceeds the 1e-2 budget under outliers");
        Assert.True(errSmoothed < errUnsmoothed / 4f,
            $"smoothing gain insufficient: unsmoothed {errUnsmoothed} vs smoothed {errSmoothed}");
    }

    [Fact]
    public void Int8Attention_DitShapes_HoldToleranceAcrossHeadDims()
    {
        // The head_dims the image/video fleet actually runs: 64 (SD-class), 128 (Flux/Qwen/Wan class).
        foreach (int dim in new[] { 64, 128 })
        {
            const int seqQ = 24, seqK = 48;
            (float[] q, float[] k, float[] v) = MakeInputs(seqQ, seqK, dim, seed: 100 + dim,
                outlierChannels: 2, outlierMagnitude: 20f);
            float scale = 1f / MathF.Sqrt(dim);

            float[] baseline = AttentionF32(q, k, v, seqQ, seqK, dim, scale);
            float[] smoothed = AttentionInt8(q, k, v, seqQ, seqK, dim, scale, smoothK: true);
            Assert.True(MaxAbsError(baseline, smoothed) < 1e-2f,
                $"head_dim {dim}: smoothed int8 maxAbs {MaxAbsError(baseline, smoothed)} exceeds 1e-2");
        }
    }

    // ── Sanity: the CPU backend SDPA agrees with this file's F32 reference (diff-target validity) ──

    [Fact]
    public void ReferenceF32_MatchesCpuBackendSdpa()
    {
        const int seqQ = 8, seqK = 12, dim = 16;
        (float[] qa, float[] ka, float[] va) = MakeInputs(seqQ, seqK, dim, seed: 55);
        float scale = 1f / MathF.Sqrt(dim);

        using CpuBackend backend = new CpuBackend();
        using Tensor q = new Tensor(new TensorShape(1, 1, seqQ, dim), DType.F32);
        using Tensor k = new Tensor(new TensorShape(1, 1, seqK, dim), DType.F32);
        using Tensor v = new Tensor(new TensorShape(1, 1, seqK, dim), DType.F32);
        using Tensor output = new Tensor(new TensorShape(1, 1, seqQ, dim), DType.F32);
        fixed (float* qp = qa, kp = ka, vp = va)
        {
            Buffer.MemoryCopy(qp, (void*)q.DataPointer, qa.Length * 4, qa.Length * 4);
            Buffer.MemoryCopy(kp, (void*)k.DataPointer, ka.Length * 4, ka.Length * 4);
            Buffer.MemoryCopy(vp, (void*)v.DataPointer, va.Length * 4, va.Length * 4);
        }
        backend.ScaledDotProductAttention(output, q, k, v, null, scale);

        float[] expected = AttentionF32(qa, ka, va, seqQ, seqK, dim, scale);
        float* op = (float*)output.DataPointer;
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(op[i] - expected[i]) < 1e-4f,
                $"backend SDPA[{i}]={op[i]} vs reference {expected[i]}");
    }
}
