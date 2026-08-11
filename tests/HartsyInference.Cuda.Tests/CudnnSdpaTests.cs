using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Validates the cuDNN fused flash-attention SDPA fast path (HARTSY_SDPA_CUDNN) against a CPU
/// reference at the Krea2-style head dim (D=128, no mask, MHA). fp16 I/O so tolerance is fp16-scale.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudnnSdpaTests
{
    private readonly ITestOutputHelper _output;
    public CudnnSdpaTests(ITestOutputHelper output) => _output = output;

    private static uint _rng = 0x1234567u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return (_rng & 0xFFFF) / 65535f - 0.5f; }
    private static Tensor Rnd(int a, int b, int c, int d) { Tensor t = new(new TensorShape(a, b, c, d), DType.F32); float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }

    private bool HasRequiredCudnn()
    {
        CudnnRuntime.EnsureProbed();
        if (CudnnRuntime.SupportsSdpa) return true;
        _output.WriteLine($"SKIPPED: {CudnnRuntime.Reason}");
        return false;
    }

    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    public void CudnnSdpa_MatchesCpuReference(int d)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (!HasRequiredCudnn()) return;
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        const int heads = 4, s = 256;
        float scale = 1f / MathF.Sqrt(d);

        string? prev = Environment.GetEnvironmentVariable("HARTSY_SDPA_CUDNN");
        Environment.SetEnvironmentVariable("HARTSY_SDPA_CUDNN", "1");
        try
        {
            using CudaBackend backend = new(0, ptxDir);
            IBackend b = backend;
            using Tensor q = Rnd(1, heads, s, d);
            using Tensor k = Rnd(1, heads, s, d);
            using Tensor v = Rnd(1, heads, s, d);
            using Tensor outT = new(new TensorShape(1, heads, s, d), DType.F32);
            b.ScaledDotProductAttention(outT, q, k, v, null, scale, allowF16: true);   // allowF16 gates the cuDNN path
            backend.Sync();

            // CPU reference: softmax(QKᵀ·scale) @ V per head
            float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer, op = (float*)outT.DataPointer;
            float maxDiff = 0f;
            float[] row = new float[s];
            for (int h = 0; h < heads; h++)
            {
                for (int i = 0; i < s; i++)
                {
                    long qBase = ((long)h * s + i) * d;
                    float m = float.NegativeInfinity;
                    for (int j = 0; j < s; j++)
                    {
                        long kBase = ((long)h * s + j) * d;
                        float dot = 0f;
                        for (int e = 0; e < d; e++) dot += qp[qBase + e] * kp[kBase + e];
                        dot *= scale;
                        row[j] = dot;
                        if (dot > m) m = dot;
                    }
                    float denom = 0f;
                    for (int j = 0; j < s; j++) { row[j] = MathF.Exp(row[j] - m); denom += row[j]; }
                    long oBase = ((long)h * s + i) * d;
                    for (int e = 0; e < d; e++)
                    {
                        float acc = 0f;
                        for (int j = 0; j < s; j++) acc += row[j] * vp[((long)h * s + j) * d + e];
                        maxDiff = MathF.Max(maxDiff, MathF.Abs(acc / denom - op[oBase + e]));
                    }
                }
            }
            _output.WriteLine($"d={d}: max |CPU ref - cuDNN SDPA| = {maxDiff:E3}  engaged={backend.CudnnSdpaEngaged}");
            Assert.True(backend.CudnnSdpaEngaged, $"cuDNN SDPA fast path did NOT engage (d={d}) — it fell back to the materialized path");
            Assert.Equal(1, backend.CudnnSdpaExecutionCount);
            Assert.True(maxDiff <= 3e-2f, $"cuDNN SDPA (d={d}) diverges from CPU reference by {maxDiff:E3}");
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_SDPA_CUDNN", prev);
        }
    }

    /// <summary>Masked SDPA must also hit the fused engine: the Chroma-style additive [1,1,S,S] outer keep-mask
    /// (0 kept / -1e30 masked, a padded-text span strictly inside the sequence) rides the graph as an fp32 bias
    /// score-modifier. Kept-query rows are compared against the CPU reference; masked-query rows are
    /// uniform-softmax garbage by convention in BOTH paths and are skipped.</summary>
    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    public void CudnnSdpa_Masked_MatchesCpuReference(int d)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (!HasRequiredCudnn()) return;
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        const int heads = 4, s = 256;
        float scale = 1f / MathF.Sqrt(d);

        string? prev = Environment.GetEnvironmentVariable("HARTSY_SDPA_CUDNN");
        Environment.SetEnvironmentVariable("HARTSY_SDPA_CUDNN", "1");
        try
        {
            using CudaBackend backend = new(0, ptxDir);
            IBackend b = backend;
            using Tensor q = Rnd(1, heads, s, d);
            using Tensor k = Rnd(1, heads, s, d);
            using Tensor v = Rnd(1, heads, s, d);
            using Tensor mask = new(new TensorShape(1, 1, s, s), DType.F32);
            bool[] keep = new bool[s];
            for (int i = 0; i < s; i++) keep[i] = i < s / 8 || i >= s / 4;   // masked span strictly inside
            float* mp = (float*)mask.DataPointer;
            for (int i = 0; i < s; i++)
                for (int j = 0; j < s; j++)
                    mp[(long)i * s + j] = keep[i] && keep[j] ? 0f : -1e30f;
            using Tensor outT = new(new TensorShape(1, heads, s, d), DType.F32);
            b.ScaledDotProductAttention(outT, q, k, v, mask, scale, allowF16: true);
            backend.Sync();

            // CPU reference over KEPT query rows only: softmax over kept keys (masked keys excluded)
            float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer, op = (float*)outT.DataPointer;
            float maxDiff = 0f;
            float[] row = new float[s];
            for (int h = 0; h < heads; h++)
            {
                for (int i = 0; i < s; i++)
                {
                    if (!keep[i]) continue;
                    long qBase = ((long)h * s + i) * d;
                    float m = float.NegativeInfinity;
                    for (int j = 0; j < s; j++)
                    {
                        if (!keep[j]) { row[j] = float.NegativeInfinity; continue; }
                        long kBase = ((long)h * s + j) * d;
                        float dot = 0f;
                        for (int e = 0; e < d; e++) dot += qp[qBase + e] * kp[kBase + e];
                        dot *= scale;
                        row[j] = dot;
                        if (dot > m) m = dot;
                    }
                    float denom = 0f;
                    for (int j = 0; j < s; j++) { row[j] = keep[j] ? MathF.Exp(row[j] - m) : 0f; denom += row[j]; }
                    long oBase = ((long)h * s + i) * d;
                    for (int e = 0; e < d; e++)
                    {
                        float acc = 0f;
                        for (int j = 0; j < s; j++) acc += row[j] * vp[((long)h * s + j) * d + e];
                        maxDiff = MathF.Max(maxDiff, MathF.Abs(acc / denom - op[oBase + e]));
                    }
                }
            }
            _output.WriteLine($"d={d} masked: max |CPU ref - cuDNN SDPA| = {maxDiff:E3}  engaged={backend.CudnnSdpaEngaged}");
            Assert.True(backend.CudnnSdpaEngaged, $"masked cuDNN SDPA did NOT engage (d={d}) — it fell back to the materialized path");
            Assert.Equal(1, backend.CudnnSdpaExecutionCount);
            Assert.True(maxDiff <= 3e-2f, $"masked cuDNN SDPA (d={d}) diverges from CPU reference by {maxDiff:E3}");
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_SDPA_CUDNN", prev);
        }
    }

    /// <summary>Plans with identical shapes but different scales must not share the device scale scalar.</summary>
    [Fact]
    public void CudnnSdpa_SameShapeDifferentScale_MatchesEachReference()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (!HasRequiredCudnn()) return;
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        const int Heads = 4, S = 256, D = 64;
        const float ScaleA = 0.125f;
        const float ScaleB = 4.0f;
        using Tensor q = Rnd(1, Heads, S, D);
        using Tensor k = Rnd(1, Heads, S, D);
        using Tensor v = Rnd(1, Heads, S, D);
        using Tensor expectedA = new Tensor(q.Shape, DType.F32);
        using Tensor expectedB = new Tensor(q.Shape, DType.F32);
        using Tensor actualA = new Tensor(q.Shape, DType.F32);
        using Tensor actualB = new Tensor(q.Shape, DType.F32);
        AttentionReference.FlashAttention(expectedA, q, k, v, S, 1, causal: false, qOffset: 0, ScaleA);
        AttentionReference.FlashAttention(expectedB, q, k, v, S, 1, causal: false, qOffset: 0, ScaleB);

        string? previous = Environment.GetEnvironmentVariable("HARTSY_SDPA_CUDNN");
        Environment.SetEnvironmentVariable("HARTSY_SDPA_CUDNN", "1");
        try
        {
            using CudaBackend backend = new CudaBackend(0, ptxDir);
            backend.ScaledDotProductAttention(actualA, q, k, v, null, ScaleA, allowF16: true);
            backend.Sync();
            _ = *(float*)actualA.DataPointer;
            backend.ScaledDotProductAttention(actualB, q, k, v, null, ScaleB, allowF16: true);
            backend.Sync();
            Assert.True(backend.CudnnSdpaEngaged, "cuDNN SDPA did not engage for the scale-cache test.");
            Assert.Equal(2, backend.CudnnSdpaExecutionCount);
            _ = *(float*)actualB.DataPointer;
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_SDPA_CUDNN", previous);
        }

        static float MaxDifference(Tensor expected, Tensor actual)
        {
            float* ep = (float*)expected.DataPointer;
            float* ap = (float*)actual.DataPointer;
            float max = 0f;
            for (long i = 0; i < actual.ElementCount; i++)
            {
                Assert.True(float.IsFinite(ap[i]), $"cuDNN SDPA produced a non-finite value at {i}: {ap[i]}");
                max = MathF.Max(max, MathF.Abs(ep[i] - ap[i]));
            }
            return max;
        }

        float maxA = MaxDifference(expectedA, actualA);
        float maxB = MaxDifference(expectedB, actualB);
        _output.WriteLine($"scale A max diff={maxA:E3}; scale B max diff={maxB:E3}");
        Assert.True(maxA <= 3e-2f, $"cuDNN SDPA scale A diverges by {maxA:E3}.");
        Assert.True(maxB <= 3e-2f, $"cuDNN SDPA scale B diverges by {maxB:E3}.");
    }
}
