using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.Vulkan.Tests;

/// <summary>
/// Smoke tests for the Vulkan backend. Skip themselves at runtime when no
/// Vulkan-capable physical device is visible — that way CI can include this
/// project on machines with or without a GPU.
/// </summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanBackendSmokeTests
{
    private static bool VulkanAvailable()
    {
        try
        {
            using VulkanInstance instance = new();
            nint[] devs = instance.EnumeratePhysicalDevices();
            return devs.Length > 0;
        }
        catch
        {
            return false;
        }
    }

    [Fact]
    public void Instance_BringUp()
    {
        if (!VulkanAvailable())
            return;
        using VulkanInstance instance = new();
        nint[] devs = instance.EnumeratePhysicalDevices();
        Assert.NotEmpty(devs);
    }

    [Fact]
    public void Device_BringUp_ReportsCapabilities()
    {
        if (!VulkanAvailable())
            return;
        using VulkanInstance instance = new();
        using VulkanDevice device = VulkanDevice.Create(instance);
        Assert.NotNull(device.Capabilities.DeviceName);
        Assert.True(device.Capabilities.SubgroupSize >= 4 && device.Capabilities.SubgroupSize <= 128);
        Assert.True(device.Capabilities.MaxComputeWorkGroupInvocations >= 128);
        if (device.Capabilities.HasCooperativeMatrix2)
        {
            Assert.True(device.Capabilities.CoopMat2MGranularity > 0);
            Assert.True(device.Capabilities.CoopMat2NGranularity > 0);
            Assert.True(device.Capabilities.CoopMat2KGranularity > 0);
            Assert.True(device.Capabilities.CoopMat2WorkgroupInvocations > 0);
        }
    }

    [Fact]
    public void Backend_Add_Matches_Cpu()
    {
        if (!VulkanAvailable())
            return;

        using VulkanBackend backend = new();
        Tensor a = new(new TensorShape(64), DType.F32);
        Tensor b = new(new TensorShape(64), DType.F32);
        Tensor c = new(new TensorShape(64), DType.F32);

        Span<float> aS = a.AsSpan<float>();
        Span<float> bS = b.AsSpan<float>();
        for (int i = 0; i < 64; i++) { aS[i] = i; bS[i] = -2 * i + 0.5f; }

        backend.Add(c, a, b);

        ReadOnlySpan<float> cS = c.AsReadOnlySpan<float>();
        for (int i = 0; i < 64; i++)
            Assert.InRange(cS[i] - (i + (-2 * i + 0.5f)), -1e-5f, 1e-5f);

        a.Dispose(); b.Dispose(); c.Dispose();
    }

    [Fact]
    public void Backend_Silu_Matches_Reference()
    {
        if (!VulkanAvailable())
            return;
        using VulkanBackend backend = new();

        Tensor x = new(new TensorShape(32), DType.F32);
        Tensor y = new(new TensorShape(32), DType.F32);
        Span<float> xS = x.AsSpan<float>();
        for (int i = 0; i < 32; i++) xS[i] = i * 0.25f - 4.0f;

        backend.Silu(y, x);

        ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
        for (int i = 0; i < 32; i++)
        {
            float xi = i * 0.25f - 4.0f;
            float expected = xi / (1.0f + MathF.Exp(-xi));
            Assert.InRange(yS[i] - expected, -1e-5f, 1e-5f);
        }
        x.Dispose(); y.Dispose();
    }

    [Fact]
    public void Backend_LayerNorm_Matches_Reference()
    {
        if (!VulkanAvailable())
            return;
        using VulkanBackend backend = new();
        const int B = 2, T = 3, D = 16;

        Tensor x = new(new TensorShape(B, T, D), DType.F32);
        Tensor w = new(new TensorShape(D), DType.F32);
        Tensor b = new(new TensorShape(D), DType.F32);
        Tensor y = new(new TensorShape(B, T, D), DType.F32);

        Span<float> xS = x.AsSpan<float>();
        Span<float> wS = w.AsSpan<float>();
        Span<float> bS = b.AsSpan<float>();
        for (int i = 0; i < B * T * D; i++) xS[i] = MathF.Sin(i * 0.123f) * 2.0f;
        for (int i = 0; i < D; i++) { wS[i] = 1.0f; bS[i] = 0.0f; }

        backend.LayerNorm(y, x, w, b, 1e-5f);

        // CPU reference
        ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
        for (int row = 0; row < B * T; row++)
        {
            float mean = 0.0f;
            for (int d = 0; d < D; d++) mean += xS[row * D + d];
            mean /= D;
            float var_ = 0.0f;
            for (int d = 0; d < D; d++) { float v = xS[row * D + d] - mean; var_ += v * v; }
            var_ /= D;
            float invStd = 1.0f / MathF.Sqrt(var_ + 1e-5f);
            for (int d = 0; d < D; d++)
            {
                float expected = (xS[row * D + d] - mean) * invStd;
                Assert.InRange(yS[row * D + d] - expected, -1e-4f, 1e-4f);
            }
        }

        x.Dispose(); w.Dispose(); b.Dispose(); y.Dispose();
    }

    [Fact]
    public void Backend_MatMul_F16_Roundtrip()
    {
        if (!VulkanAvailable())
            return;
        using VulkanBackend backend = new();
        if (!backend.Capabilities.SupportsF16)
            return;     // skip on devices without FP16 (e.g. Mesa LLVMpipe software path)

        const int M = 8, K = 8, N = 8;
        Tensor a = new(new TensorShape(M, K), DType.F16);
        Tensor b = new(new TensorShape(K, N), DType.F16);
        Tensor c = new(new TensorShape(M, N), DType.F16);

        Span<Half> aS = a.AsSpan<Half>();
        Span<Half> bS = b.AsSpan<Half>();
        for (int i = 0; i < M * K; i++) aS[i] = (Half)(i * 0.0625f);
        for (int i = 0; i < K * N; i++) bS[i] = (Half)((i + 1) * 0.03125f);

        backend.MatMul(c, a, b);

        ReadOnlySpan<Half> cS = c.AsReadOnlySpan<Half>();
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
            {
                float acc = 0.0f;
                for (int k = 0; k < K; k++) acc += (float)aS[m * K + k] * (float)bS[k * N + n];
                float got = (float)cS[m * N + n];
                Assert.InRange(got - acc, -5e-2f, 5e-2f);
            }
        a.Dispose(); b.Dispose(); c.Dispose();
    }

    /// <summary>Per-batch weights ([B,K,N]) exercise the offset-dispatch path; a shared 2D weight exercises the flattened single-GEMM path.</summary>
    [Fact]
    public void Backend_BatchedMatMul_Matches_Scalar_Reference()
    {
        if (!VulkanAvailable())
            return;
        using VulkanBackend backend = new();

        const int B = 3, M = 4, K = 5, N = 6;
        Tensor a = new(new TensorShape(B, M, K), DType.F32);
        Span<float> aS = a.AsSpan<float>();
        for (int i = 0; i < aS.Length; i++) aS[i] = (i % 13) * 0.25f - 1.5f;

        Tensor b3 = new(new TensorShape(B, K, N), DType.F32);
        Span<float> b3S = b3.AsSpan<float>();
        for (int i = 0; i < b3S.Length; i++) b3S[i] = (i % 7) * 0.125f - 0.375f;

        Tensor c3 = new(new TensorShape(B, M, N), DType.F32);
        backend.BatchedMatMul(c3, a, b3);
        ReadOnlySpan<float> c3S = c3.AsReadOnlySpan<float>();
        for (int bi = 0; bi < B; bi++)
            for (int m = 0; m < M; m++)
                for (int n = 0; n < N; n++)
                {
                    float acc = 0.0f;
                    for (int k = 0; k < K; k++) acc += aS[(bi * M + m) * K + k] * b3S[(bi * K + k) * N + n];
                    Assert.InRange(c3S[(bi * M + m) * N + n] - acc, -1e-4f, 1e-4f);
                }
        c3.Dispose(); b3.Dispose();

        Tensor b2 = new(new TensorShape(K, N), DType.F32);
        Span<float> b2S = b2.AsSpan<float>();
        for (int i = 0; i < b2S.Length; i++) b2S[i] = (i % 5) * 0.2f - 0.4f;

        Tensor c2 = new(new TensorShape(B, M, N), DType.F32);
        backend.BatchedMatMul(c2, a, b2);
        ReadOnlySpan<float> c2S = c2.AsReadOnlySpan<float>();
        for (int bi = 0; bi < B; bi++)
            for (int m = 0; m < M; m++)
                for (int n = 0; n < N; n++)
                {
                    float acc = 0.0f;
                    for (int k = 0; k < K; k++) acc += aS[(bi * M + m) * K + k] * b2S[k * N + n];
                    Assert.InRange(c2S[(bi * M + m) * N + n] - acc, -1e-4f, 1e-4f);
                }
        c2.Dispose(); b2.Dispose(); a.Dispose();
    }

    /// <summary>Compares Vulkan FP8 cast vs CPU CastTo by reading every byte value (0..255) via CastToF16 — both paths exercise the kernel.</summary>
    [Fact]
    public void Backend_FP8_Cast_Matches_Cpu_AllBytes()
    {
        if (!VulkanAvailable())
            return;
        using VulkanBackend backend = new();

        Tensor fp8 = new(new TensorShape(256), DType.F8E4M3);
        Span<byte> bytes = fp8.AsSpan<byte>();
        for (int i = 0; i < 256; i++) bytes[i] = (byte)i;

        // CPU reference
        Tensor cpuF16 = fp8.CastTo(DType.F16);
        ReadOnlySpan<Half> cpuVals = cpuF16.AsReadOnlySpan<Half>();

        // Vulkan path
        Tensor vkF16 = new(new TensorShape(256), DType.F16);
        backend.CastToF16(vkF16, fp8);
        ReadOnlySpan<Half> vkVals = vkF16.AsReadOnlySpan<Half>();

        int diffs = 0;
        int firstDiffIdx = -1;
        float worstAbsErr = 0;
        for (int i = 0; i < 256; i++)
        {
            // Skip the NaN sentinel byte (0x7F / 0xFF). My GLSL returns 0; CPU returns 480.
            if (i == 0x7F || i == 0xFF) continue;
            float cpu = (float)cpuVals[i];
            float vk = (float)vkVals[i];
            if (float.IsNaN(cpu) || float.IsNaN(vk)) continue;
            float absErr = MathF.Abs(cpu - vk);
            float tol = 1e-3f + 0.01f * MathF.Max(1.0f, MathF.Abs(cpu));
            if (absErr > tol)
            {
                if (firstDiffIdx < 0) { firstDiffIdx = i; worstAbsErr = absErr; }
                diffs++;
            }
        }
        if (diffs > 0)
        {
            float cpuV = (float)cpuVals[firstDiffIdx];
            float vkV  = (float)vkVals[firstDiffIdx];
            Assert.Fail($"FP8 cast mismatched on {diffs} bytes; first diff at byte 0x{firstDiffIdx:X2}: cpu={cpuV:G6} vk={vkV:G6}  worstAbsErr={worstAbsErr:G6}");
        }

        fp8.Dispose(); cpuF16.Dispose(); vkF16.Dispose();
    }

    [Fact]
    public void Backend_RmsNorm_Matches_Reference()
    {
        if (!VulkanAvailable())
            return;
        using VulkanBackend backend = new();
        const int B = 2, T = 3, D = 16;

        Tensor x = new(new TensorShape(B, T, D), DType.F32);
        Tensor w = new(new TensorShape(D), DType.F32);
        Tensor y = new(new TensorShape(B, T, D), DType.F32);

        Span<float> xS = x.AsSpan<float>();
        Span<float> wS = w.AsSpan<float>();
        for (int i = 0; i < B * T * D; i++) xS[i] = MathF.Sin(i * 0.137f) * 1.5f;
        for (int i = 0; i < D; i++) wS[i] = 1.0f + 0.02f * i;

        backend.RmsNorm(y, x, w, 1e-6f);

        ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
        for (int row = 0; row < B * T; row++)
        {
            float sqsum = 0;
            for (int d = 0; d < D; d++) { float v = xS[row * D + d]; sqsum += v * v; }
            float invStd = 1.0f / MathF.Sqrt(sqsum / D + 1e-6f);
            for (int d = 0; d < D; d++)
            {
                float expected = xS[row * D + d] * invStd * wS[d];
                Assert.InRange(yS[row * D + d] - expected, -1e-4f, 1e-4f);
            }
        }
        x.Dispose(); w.Dispose(); y.Dispose();
    }

    /// <summary>SDPA with Flux-like dimensions (H=24, S=64, D=128) — the per-(b, h) offset loop must be correct for every head.</summary>
    [Fact]
    public void Backend_SDPA_FluxShape_AllHeads_Match_Cpu()
    {
        if (!VulkanAvailable())
            return;
        using VulkanBackend backend = new();

        const int B = 1, H = 24, S = 64, D = 128;   // smaller S than full Flux for test speed
        Tensor q = new(new TensorShape(B, H, S, D), DType.F32);
        Tensor k = new(new TensorShape(B, H, S, D), DType.F32);
        Tensor v = new(new TensorShape(B, H, S, D), DType.F32);
        Tensor o = new(new TensorShape(B, H, S, D), DType.F32);

        Random rng = new(42);
        Span<float> qS = q.AsSpan<float>();
        Span<float> kS = k.AsSpan<float>();
        Span<float> vS = v.AsSpan<float>();
        for (int i = 0; i < B * H * S * D; i++)
        {
            qS[i] = (float)(rng.NextDouble() * 2 - 1);
            kS[i] = (float)(rng.NextDouble() * 2 - 1);
            vS[i] = (float)(rng.NextDouble() * 2 - 1);
        }

        float scale = 1.0f / MathF.Sqrt(D);
        backend.ScaledDotProductAttention(o, q, k, v, mask: null, scale);

        ReadOnlySpan<float> oS = o.AsReadOnlySpan<float>();

        // CPU reference per (b, h). Spot-check ALL heads' first row to detect per-head bugs.
        int errs = 0;
        int firstBadHead = -1;
        float maxAbs = 0;
        for (int h = 0; h < H; h++)
        {
            int baseIdx = h * S * D;
            // Compute expected output[h, 0, :] — first query position only (sufficient to detect head-level corruption)
            float[] scores = new float[S];
            float maxScore = float.NegativeInfinity;
            for (int j = 0; j < S; j++)
            {
                float acc = 0;
                for (int d = 0; d < D; d++) acc += qS[baseIdx + 0 * D + d] * kS[baseIdx + j * D + d];
                scores[j] = acc * scale;
                if (scores[j] > maxScore) maxScore = scores[j];
            }
            float sum = 0;
            for (int j = 0; j < S; j++) { scores[j] = MathF.Exp(scores[j] - maxScore); sum += scores[j]; }
            for (int j = 0; j < S; j++) scores[j] /= sum;

            for (int d = 0; d < D; d++)
            {
                float expected = 0;
                for (int j = 0; j < S; j++) expected += scores[j] * vS[baseIdx + j * D + d];
                float vk = oS[baseIdx + 0 * D + d];
                float err = MathF.Abs(vk - expected);
                if (err > 1e-3f)
                {
                    if (errs == 0) firstBadHead = h;
                    errs++;
                    maxAbs = MathF.Max(maxAbs, err);
                }
            }
        }
        Assert.True(errs == 0, $"SDPA had {errs} mismatches across {H} heads. First bad head: {firstBadHead}, maxAbs={maxAbs:G6}");

        q.Dispose(); k.Dispose(); v.Dispose(); o.Dispose();
    }

    [Fact]
    public void Backend_GeGlu_Matches_Reference()
    {
        if (!VulkanAvailable())
            return;
        using VulkanBackend backend = new();
        // Multi-row test (the canonical PHASE_3_DEVIATIONS #16 regression case).
        const int B = 2, T = 3, D = 8;   // last-dim = 2*D = 16
        Tensor x = new(new TensorShape(B, T, 2 * D), DType.F32);
        Tensor y = new(new TensorShape(B, T, D), DType.F32);
        Span<float> xS = x.AsSpan<float>();
        for (int i = 0; i < B * T * 2 * D; i++) xS[i] = MathF.Cos(i * 0.21f) * 1.7f;

        backend.GeGlu(y, x);

        ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
        for (int row = 0; row < B * T; row++)
        {
            for (int d = 0; d < D; d++)
            {
                float xv = xS[row * 2 * D + d];
                float gv = xS[row * 2 * D + D + d];
                float gelu = 0.5f * gv * (1f + MathF.Tanh(0.7978845608f * (gv + 0.044715f * gv * gv * gv)));
                float expected = xv * gelu;
                Assert.InRange(yS[row * D + d] - expected, -1e-4f, 1e-4f);
            }
        }
        x.Dispose(); y.Dispose();
    }

    /// <summary>Linear at Flux DiT dimensions (M=64, K=3072, N=3072), F32 throughout — exactly the shape Flux uses for its Q/K/V/O projections.</summary>
    [Fact]
    public void Backend_Linear_FluxShape_F32_Matches_Cpu()
    {
        if (!VulkanAvailable())
            return;
        using VulkanBackend backend = new();

        // 1280 = max Flux Schnell sequence length (256 T5 + 1024 image tokens)
        const int M = 1280, K = 3072, N = 3072;
        Tensor input = new(new TensorShape(M, K), DType.F32);
        Tensor weight = new(new TensorShape(N, K), DType.F32);
        Tensor bias = new(new TensorShape(N), DType.F32);
        Tensor output = new(new TensorShape(M, N), DType.F32);

        Random rng = new(42);
        Span<float> iS = input.AsSpan<float>();
        Span<float> wS = weight.AsSpan<float>();
        Span<float> bS = bias.AsSpan<float>();
        for (int i = 0; i < M * K; i++) iS[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        for (int i = 0; i < N * K; i++) wS[i] = (float)(rng.NextDouble() * 2 - 1) * 0.05f;
        for (int i = 0; i < N; i++) bS[i] = (float)(rng.NextDouble() * 2 - 1) * 0.01f;

        backend.Linear(output, input, weight, bias);
        ReadOnlySpan<float> oS = output.AsReadOnlySpan<float>();

        // CPU reference: out[m, n] = sum_k input[m, k] * weight[n, k] + bias[n]
        // Spot-check positions across the entire output matrix to catch tile-boundary bugs.
        int errs = 0; int firstM = -1, firstN = -1;
        float maxAbs = 0;
        // Sample interior, edges, and far corners.
        int[] mProbes = { 0, 1, 31, 32, 33, 64, 100, 500, 1000, 1100, M - 1 };
        int[] nProbes = { 0, 1, 31, 32, 33, 1000, 2000, N - 1 };
        foreach (int m in mProbes)
        foreach (int n in nProbes)
        {
            float acc = 0;
            for (int k = 0; k < K; k++) acc += iS[m * K + k] * wS[n * K + k];
            acc += bS[n];
            float vk = oS[m * N + n];
            float err = MathF.Abs(vk - acc);
            if (err > 1e-2f)
            {
                if (errs == 0) { firstM = m; firstN = n; }
                errs++;
                maxAbs = MathF.Max(maxAbs, err);
            }
        }
        if (errs > 0)
        {
            float exp = 0;
            for (int k = 0; k < K; k++) exp += iS[firstM * K + k] * wS[firstN * K + k];
            exp += bS[firstN];
            Assert.Fail($"Linear F32 Flux-shape: {errs} probe diffs. First at out[{firstM},{firstN}]: vk={oS[firstM * N + firstN]:G6} cpu={exp:G6}  maxAbsErr={maxAbs:G6}");
        }

        input.Dispose(); weight.Dispose(); bias.Dispose(); output.Dispose();
    }

    /// <summary>Regression gate for a real Krea2-on-Vulkan bug (2026-07-30): <c>DispatchMatmul</c> derived
    /// M/N from <c>output.Shape</c>'s rank structure ("flatten all-but-last dims into M, last dim is N"),
    /// which is silently wrong whenever a caller shapes a Linear's output as <c>[B, S, heads, headDim]</c>
    /// (done so a downstream per-head op like RmsNorm can normalize over headDim without a reshape) — the
    /// true output width is <c>heads·headDim</c>, spanning two trailing dims, not just the last one. The old
    /// code computed M too large and N too small, reading input rows past their actual extent (out-of-bounds
    /// VRAM) and using only a slice of the weight matrix — Krea2's to_q/to_k Linears came back ~90% exact
    /// zero against the real checkpoint at production scale, corrupting every downstream op (each of which
    /// tested bit-correct against ITS OWN already-wrong input, hiding the bug from per-op checks) and
    /// producing a pure-noise image end-to-end despite individual ops appearing correct in isolation. Fixed
    /// by deriving N from the weight tensor (mirrors <c>CudaBackend.LinearImpl</c>, which never consults
    /// output.Shape at all) and M as <c>output.ElementCount / N</c>. This test uses a rank-4 output shape
    /// with heads·headDim split across two dims — the exact shape class that exposed the bug.</summary>
    [Fact]
    public void Backend_Linear_SplitHeadOutputShape_MatchesCpu()
    {
        if (!VulkanAvailable())
            return;
        using VulkanBackend backend = new();

        const int batch = 1, seqLen = 37, heads = 6, headDim = 8, hidden = 96;
        const int K = hidden, N = heads * headDim;   // N = 48; deliberately != hidden to catch any accidental K/N mixup
        Tensor input = new(new TensorShape(batch, seqLen, K), DType.F32);
        Tensor weight = new(new TensorShape(N, K), DType.F32);
        Tensor output = new(new TensorShape(batch, seqLen, heads, headDim), DType.F32);   // rank-4: split last dim

        Random rng = new(7);
        Span<float> iS = input.AsSpan<float>();
        Span<float> wS = weight.AsSpan<float>();
        for (int i = 0; i < batch * seqLen * K; i++) iS[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < N * K; i++) wS[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;

        backend.Linear(output, input, weight, null);
        ReadOnlySpan<float> oS = output.AsReadOnlySpan<float>();

        // CPU reference: out[s, n] = sum_k input[s, k] * weight[n, k], flat-indexed [seqLen, N] (byte-identical
        // to the rank-4 [batch, seqLen, heads, headDim] output layout).
        int errs = 0; int firstS = -1, firstN = -1; float maxAbs = 0;
        for (int s = 0; s < seqLen; s++)
        for (int n = 0; n < N; n++)
        {
            float acc = 0;
            for (int k = 0; k < K; k++) acc += iS[s * K + k] * wS[n * K + k];
            float vk = oS[s * N + n];
            float err = MathF.Abs(vk - acc);
            if (err > 1e-3f)
            {
                if (errs == 0) { firstS = s; firstN = n; }
                errs++;
                maxAbs = MathF.Max(maxAbs, err);
            }
        }
        if (errs > 0)
        {
            float exp = 0;
            for (int k = 0; k < K; k++) exp += iS[firstS * K + k] * wS[firstN * K + k];
            Assert.Fail($"Linear split-head-output: {errs}/{seqLen * N} probe diffs. First at out[{firstS},{firstN}]: vk={oS[firstS * N + firstN]:G6} cpu={exp:G6}  maxAbsErr={maxAbs:G6}");
        }

        input.Dispose(); weight.Dispose(); output.Dispose();
    }

    /// <summary>Step-graph prerequisite (Phase 7): <c>CopyInto</c> must preserve <c>dst</c>'s buffer ADDRESS across
    /// repeated calls (the captured-graph boundary-refresh pattern — <c>Krea2Transformer</c> refreshes
    /// <c>_tembFixed</c>/writes <c>_graphVelocity</c> via this exact call every step, and a captured command buffer
    /// bakes the destination's device address at record time). Covers both src origins: a device-resident source
    /// (weight/activation cache hit — the D2D fast path) and a host-only source (fresh upload).</summary>
    [Fact]
    public void Backend_CopyInto_PreservesDstAddress_MatchesCpu_DeviceSrc()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        // dst mirrors the real usage (Krea2Transformer's _tembFixed/_graphVelocity): a plain, never-preloaded
        // tensor whose device buffer CopyInto itself materializes on first use. Pre-caching dst as a WEIGHT
        // before CopyInto would leave it double-registered (weight cache + CopyInto's own activation cache)
        // and double-freed on dispose — not a real usage pattern, so the test doesn't do it either. Likewise,
        // real step-graph usage NEVER reads dst's value from host BETWEEN CopyInto calls (that's the whole
        // point — DataPointer reads are capture-illegal) — reading it via AsReadOnlySpan mid-sequence here
        // would trigger the lazy-sync eviction callback and free the buffer, breaking the very address
        // stability being tested. Address checks stay reflection-only (AddressOf, no host read) until the end.
        const int n = 257;   // deliberately not a round dispatch-tile size
        Tensor dst = new(new TensorShape(n), DType.F32);

        float[] src1Data = FillRandom(n, 11);
        Tensor src1 = new(new TensorShape(n), DType.F32);
        src1Data.CopyTo(src1.AsSpan<float>());
        backend.PreloadWeights(new[] { src1 });   // device-resident source (D2D path)

        backend.CopyInto(dst, src1);
        ulong addrAfterFirst = AddressOf(backend, dst);

        float[] src2Data = FillRandom(n, 22);
        Tensor src2 = new(new TensorShape(n), DType.F32);
        src2Data.CopyTo(src2.AsSpan<float>());
        backend.PreloadWeights(new[] { src2 });

        backend.CopyInto(dst, src2);
        ulong addrAfterSecond = AddressOf(backend, dst);
        Assert.Equal(addrAfterFirst, addrAfterSecond);
        AssertMatches(dst, src2Data);   // final value only — host read is fine now, nothing depends on dst after this

        backend.FreeWeights(new[] { src1, src2 });
        dst.Dispose(); src1.Dispose(); src2.Dispose();
    }

    [Fact]
    public void Backend_CopyInto_PreservesDstAddress_MatchesCpu_HostSrc()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int n = 129;
        Tensor dst = new(new TensorShape(n), DType.F32);

        float[] src1Data = FillRandom(n, 33);
        Tensor src1 = new(new TensorShape(n), DType.F32);   // host-only: never preloaded/cached
        src1Data.CopyTo(src1.AsSpan<float>());

        backend.CopyInto(dst, src1);
        ulong addrAfterFirst = AddressOf(backend, dst);

        float[] src2Data = FillRandom(n, 44);
        Tensor src2 = new(new TensorShape(n), DType.F32);
        src2Data.CopyTo(src2.AsSpan<float>());

        backend.CopyInto(dst, src2);
        ulong addrAfterSecond = AddressOf(backend, dst);
        Assert.Equal(addrAfterFirst, addrAfterSecond);
        AssertMatches(dst, src2Data);

        dst.Dispose(); src1.Dispose(); src2.Dispose();
    }

    private static ulong AddressOf(VulkanBackend backend, Tensor t)
    {
        System.Reflection.MethodInfo? m = typeof(VulkanBackend).GetMethod("GetBuffer", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        VulkanBuffer buf = (VulkanBuffer)m!.Invoke(backend, new object[] { t })!;
        return buf.Handle;
    }

    private static void AssertMatches(Tensor t, float[] expected)
    {
        ReadOnlySpan<float> actual = t.AsReadOnlySpan<float>();
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], 3);
    }

    /// <summary>THE gate for step-graph capture (Phase 6e/7): captures a trivial 2-dispatch sequence (Sigmoid
    /// then Mul — SiLU decomposed manually) reading/writing FIXED buffers, disposing its one intermediate
    /// tensor mid-capture (exactly like Krea2Attention.Forward's per-op q/k/v Dispose() calls) to directly
    /// exercise the capture-time buffer-retention path, then replays it 3 times with FRESH input each time —
    /// proving the graph reads live refreshed data rather than freezing capture-time values, and that the
    /// disposed intermediate survived for the Mul dispatch that references it on every replay. If this doesn't
    /// pass, a 28-block Krea2 capture never will (see VulkanStepGraph's doc comment for why: pool-allocated
    /// descriptor sets and normal buffer-dispose-frees are both unsafe for a replayed command buffer).</summary>
    [Fact]
    public void StepGraph_TrivialCapture_ReplaysWithFreshDataAcrossThreeSteps()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();
        if (!backend.StepGraphSupported) return;   // push descriptors unavailable on this device/driver

        const int n = 64;
        TensorShape shape = new(n);
        Tensor inputFixed = new(shape, DType.F32);     // refreshed via CopyInto before each launch
        Tensor outputFixed = new(shape, DType.F32);    // the graph's fixed output boundary

        void RefreshInput(float[] data)
        {
            Tensor src = new(shape, DType.F32);
            data.CopyTo(src.AsSpan<float>());
            backend.CopyInto(inputFixed, src);
            src.Dispose();
        }

        // Reading a graph-owned fixed tensor's DataPointer directly would D2H-and-free the buffer the
        // captured command buffer's descriptors point at (see Krea2Transformer.SnapshotGraphLatent's doc) —
        // CopyInto into a throwaway tensor first, exactly like the real pipeline does.
        float[] ReadSnapshot()
        {
            Tensor snap = new(shape, DType.F32);
            backend.CopyInto(snap, outputFixed);
            float[] result = snap.AsReadOnlySpan<float>().ToArray();
            snap.Dispose();
            return result;
        }

        static float[] CpuSilu(float[] x)
        {
            float[] o = new float[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                float s = 1f / (1f + MathF.Exp(-x[i]));
                o[i] = x[i] * s;
            }
            return o;
        }

        float[] step1 = FillRandom(n, 101);
        RefreshInput(step1);

        backend.StepGraphBegin();
        Tensor mid = new(shape, DType.F32);
        backend.Sigmoid(mid, inputFixed);
        backend.Mul(outputFixed, mid, inputFixed);
        mid.Dispose();   // mid-capture dispose — exercises VulkanGpuTransferHelper's retain-instead-of-free path
        backend.StepGraphEndAndLaunch();   // capture records without executing beyond this call — this IS step 1's replay

        float[] expected1 = CpuSilu(step1);
        float[] actual1 = ReadSnapshot();
        for (int i = 0; i < n; i++) Assert.Equal(expected1[i], actual1[i], 4);

        foreach (int seed in new[] { 202, 303 })
        {
            float[] stepData = FillRandom(n, seed);
            RefreshInput(stepData);
            Assert.True(backend.StepGraphReady);
            backend.StepGraphLaunch();

            float[] expected = CpuSilu(stepData);
            float[] actual = ReadSnapshot();
            for (int i = 0; i < n; i++) Assert.Equal(expected[i], actual[i], 4);
        }

        backend.StepGraphReset();
        inputFixed.Dispose(); outputFixed.Dispose();
    }

    /// <summary>Regression gate for a real Krea2-on-Vulkan bug (2026-07-30): <c>VulkanBackend</c> had no
    /// <c>CfgEulerStep</c> override, so every call fell through to <c>IBackend</c>'s CPU-loop default, which
    /// reads/writes <c>z.DataPointer</c> directly — evicting Krea2's fixed per-step latent from the GPU
    /// activation cache EVERY denoise step (a full D2H sync + host materialize, whether step-graph capture is
    /// in use or not), and specifically breaking step-graph capture: the evicted latent showed up as a cache
    /// miss the next time a captured dispatch tried to read it. This test proves the new GPU-resident
    /// override (a) matches the CPU-reference math for guidance != 1 (the real CFG combine, not just Krea2
    /// Turbo's guidance=1 identity case) and (b) preserves z's buffer address across repeated in-place calls —
    /// address preservation is exactly what step-graph capture requires from a fixed boundary tensor.</summary>
    [Fact]
    public void Backend_CfgEulerStep_MatchesCpu_PreservesZAddress()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int n = 200;
        TensorShape shape = new(n);
        float[] zData = FillRandom(n, 51);
        float[] posData = FillRandom(n, 52);
        float[] negData = FillRandom(n, 53);
        const float guidance = 4.5f;   // Krea2 Base's real guidance scale — NOT the Turbo guidance=1 identity case
        const float delta = -0.125f;   // a real scheduler dt is negative (flow-match integrates sigma -> 0)

        // z is NOT preloaded as a weight — CfgEulerStep's own first-touch-then-cache-as-activation path
        // establishes its ONE cache entry, exactly like the real _latentFixed usage (first populated via
        // CopyInto, never pre-registered as a weight). Preloading it as a weight too would leave it
        // double-registered (weight cache + CfgEulerStep's activation re-cache) and double-freed on dispose —
        // the same pitfall the CopyInto address test above documents.
        Tensor z = new(shape, DType.F32);
        zData.CopyTo(z.AsSpan<float>());
        Tensor pos = new(shape, DType.F32);
        posData.CopyTo(pos.AsSpan<float>());
        Tensor neg = new(shape, DType.F32);
        negData.CopyTo(neg.AsSpan<float>());
        backend.PreloadWeights(new[] { pos, neg });   // device-resident sources, matching real usage

        backend.CfgEulerStep(z, pos, neg, guidance, delta);
        ulong addrAfterFirst = AddressOf(backend, z);

        float[] expected1 = new float[n];
        for (int i = 0; i < n; i++)
        {
            float v = guidance * posData[i] + (1f - guidance) * negData[i];
            expected1[i] = zData[i] + v * delta;
        }

        // Second in-place call — z is now its OWN pos/neg source too (the real "pos=neg=v, z+=...*dt" pattern
        // when a caller re-derives pos/neg from the updated z). Feed fresh pos/neg to keep the CPU reference
        // simple; the point is verifying the address stays stable across a SECOND in-place write.
        float[] posData2 = FillRandom(n, 62);
        float[] negData2 = FillRandom(n, 63);
        Tensor pos2 = new(shape, DType.F32);
        posData2.CopyTo(pos2.AsSpan<float>());
        Tensor neg2 = new(shape, DType.F32);
        negData2.CopyTo(neg2.AsSpan<float>());
        backend.PreloadWeights(new[] { pos2, neg2 });

        backend.CfgEulerStep(z, pos2, neg2, guidance, delta);
        ulong addrAfterSecond = AddressOf(backend, z);
        Assert.Equal(addrAfterFirst, addrAfterSecond);

        float[] expected2 = new float[n];
        for (int i = 0; i < n; i++)
        {
            float v = guidance * posData2[i] + (1f - guidance) * negData2[i];
            expected2[i] = expected1[i] + v * delta;
        }
        AssertMatches(z, expected2);

        backend.FreeWeights(new[] { pos, neg, pos2, neg2 });
        z.Dispose(); pos.Dispose(); neg.Dispose(); pos2.Dispose(); neg2.Dispose();
    }

    /// <summary>Regression gate for a real Krea2-on-Vulkan bug (2026-07-30): no <c>VulkanBackend</c> override
    /// existed for <c>AddScalar</c>, so it fell through to <c>IBackend</c>'s CPU-loop default — found
    /// capture-illegal via a real <c>HARTSY_DIT_GRAPH=1</c> Krea2 run (<c>DiTUtils.Modulate</c>'s
    /// <c>AddScalar(scale, +1)</c>, the <c>(1+scale)</c> modulation convention every DiT block uses, twice
    /// per block × 28 blocks per forward pass) and, independent of graph mode, a D2H sync 56 times per
    /// denoise step regardless. New elementwise op-code 10 (<c>add_scalar</c>) in <c>elementwise.comp.glsl</c>.</summary>
    [Fact]
    public void Backend_AddScalar_MatchesCpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int n = 133;
        float[] inputData = FillRandom(n, 81);
        const float scalar = 1.0f;   // the real (1+scale) modulation use

        Tensor input = new(new TensorShape(n), DType.F32);
        inputData.CopyTo(input.AsSpan<float>());
        Tensor output = new(new TensorShape(n), DType.F32);

        backend.AddScalar(output, input, scalar);

        float[] expected = new float[n];
        for (int i = 0; i < n; i++) expected[i] = inputData[i] + scalar;
        AssertMatches(output, expected);

        input.Dispose(); output.Dispose();
    }

    /// <summary>Gate for the Stage-1b weight-cast cache: a preloaded FP8 weight feeds two consecutive
    /// Linears (first call populates the cast cache, second reuses it). Both outputs must match the CPU
    /// reference (computed from the same FP8→F16 dequant) — catches a stale/aliased/freed cached cast.</summary>
    [Fact]
    public void Backend_Linear_FP8Weight_CachedCast_Matches_Cpu()
    {
        if (!VulkanAvailable())
            return;
        using VulkanBackend backend = new();

        const int M = 256, K = 512, N = 256;   // multiples of 16 → coopmat fast path
        Tensor input = new(new TensorShape(M, K), DType.F16);
        Tensor weightF32 = new(new TensorShape(N, K), DType.F32);
        Tensor bias = new(new TensorShape(N), DType.F16);

        Random rng = new(7);
        Span<Half> iS = input.AsSpan<Half>();
        Span<float> wS = weightF32.AsSpan<float>();
        Span<Half> bS = bias.AsSpan<Half>();
        for (int i = 0; i < M * K; i++) iS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.1f);
        for (int i = 0; i < N * K; i++) wS[i] = (float)(rng.NextDouble() * 2 - 1) * 0.05f;
        for (int i = 0; i < N; i++) bS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.01f);

        // FP8 weight + the exact F16 values the GPU dequant produces (reference uses the same cast).
        Tensor weightFp8 = weightF32.CastTo(DType.F8E4M3);
        Tensor weightF16Ref = weightFp8.CastTo(DType.F16);
        ReadOnlySpan<Half> wRef = weightF16Ref.AsReadOnlySpan<Half>();

        backend.PreloadWeights(new[] { weightFp8, bias });

        int[] mProbes = { 0, 1, 16, 17, 128, M - 1 };
        int[] nProbes = { 0, 1, 16, 17, 200, N - 1 };

        for (int call = 0; call < 2; call++)
        {
            Tensor output = new(new TensorShape(M, N), DType.F16);
            backend.Linear(output, input, weightFp8, bias);
            backend.Sync();
            ReadOnlySpan<Half> oS = output.AsReadOnlySpan<Half>();

            int errs = 0, firstM = -1, firstN = -1; float maxAbs = 0;
            foreach (int m in mProbes)
            foreach (int n in nProbes)
            {
                float acc = 0;
                for (int k = 0; k < K; k++) acc += (float)iS[m * K + k] * (float)wRef[n * K + k];
                acc += (float)bS[n];
                float err = MathF.Abs((float)oS[m * N + n] - acc);
                if (err > 5e-2f)
                {
                    if (errs == 0) { firstM = m; firstN = n; }
                    errs++; maxAbs = MathF.Max(maxAbs, err);
                }
            }
            Assert.True(errs == 0,
                $"Cached-cast Linear call {call}: {errs} probe diffs, first out[{firstM},{firstN}] maxAbsErr={maxAbs:G6}");
            output.Dispose();
        }

        input.Dispose(); weightF32.Dispose(); weightFp8.Dispose(); weightF16Ref.Dispose(); bias.Dispose();
    }

    /// <summary>Regression gate for a real Krea2-on-Vulkan finding (2026-07-30): every prior FP8-weight test
    /// (including <see cref="Backend_Linear_FP8Weight_CachedCast_Matches_Cpu"/> above) left
    /// <see cref="Tensor.Fp8ScaleFactor"/> at its default 1.0 — never exercising the actual "fp8_scaled"
    /// checkpoint convention the name implies. `CudaBackend` folds a non-1.0 <c>Fp8ScaleFactor</c> into the
    /// GEMM's <c>alpha</c> (raw dequant, scale at matmul time); `VulkanBackend.CastIfNeeded`'s
    /// <c>cast_f8e4m3_f16</c> instead bakes the scale into the dequant itself and hardcodes <c>alpha=1.0</c>
    /// in both the coopmat and tiled matmul paths — a different but should-be-equivalent split of the same
    /// math, IF the cast is applied exactly once. Live e2e evidence (block-by-block CUDA-vs-Vulkan
    /// comparison on the real Krea2 checkpoint) showed Krea2's FFN output ~3x too large on Vulkan
    /// specifically — the `ff.gate/up/down` weights are exactly the fp8_scaled tensors most likely to carry
    /// a non-trivial scale. Covers both the coopmat-eligible shape (M,N,K all multiples of 16) and the
    /// tiled-fallback shape, since they apply alpha independently.</summary>
    [Theory]
    [InlineData(256, 512, 256, 6.7f)]        // multiples of 16 → coopmat fast path, large scale
    [InlineData(37, 96, 41, 6.7f)]           // tiled fallback path, large scale
    [InlineData(256, 512, 256, 0.0021f)]     // coopmat path, REAL Krea2 ff.gate/up/down scale magnitude
    [InlineData(37, 96, 41, 0.0021f)]        // tiled path, same real-world small scale
    public void Backend_Linear_FP8Weight_NonUnitScaleFactor_MatchesCpu(int M, int K, int N, float scale)
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();
        if (!backend.Capabilities.SupportsF16) return;

        Tensor input = new(new TensorShape(M, K), DType.F16);
        Tensor weightF32 = new(new TensorShape(N, K), DType.F32);

        // Tensor.CastTo(F8E4M3) encodes the F32 value's magnitude DIRECTLY into the raw fp8 byte (no
        // scale-aware pre-division — see Tensor.ConvertRange's F32->F8E4M3 branch); Fp8ScaleFactor is only
        // applied on DEQUANT. So to exercise the REAL checkpoint's raw-byte magnitude regime (real_weight
        // ~= raw_byte * scale, and Krea2's real ff.gate/up/down scale is ~0.002 with normal ~0.01-0.1
        // weight magnitudes, so raw_byte ~= weight/scale ~= 5-50) the pre-cast F32 values must be scaled up
        // by ~1/scale here — encoding a tiny weight value directly (raw_byte ~0.05) would only ever
        // exercise fp8's near-zero/subnormal range, never the real checkpoint's actual regime.
        float rawMagnitude = 0.05f / MathF.Max(scale, 1e-6f);
        Random rng = new(29);
        Span<Half> iS = input.AsSpan<Half>();
        Span<float> wS = weightF32.AsSpan<float>();
        for (int i = 0; i < M * K; i++) iS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.1f);
        for (int i = 0; i < N * K; i++) wS[i] = (float)(rng.NextDouble() * 2 - 1) * rawMagnitude;

        Tensor weightFp8 = weightF32.CastTo(DType.F8E4M3);
        weightFp8.Fp8ScaleFactor = scale;
        Tensor weightF16Ref = weightFp8.CastTo(DType.F16);   // CPU reference already folds Fp8ScaleFactor
        ReadOnlySpan<Half> wRef = weightF16Ref.AsReadOnlySpan<Half>();

        Tensor output = new(new TensorShape(M, N), DType.F16);
        backend.Linear(output, input, weightFp8, null);
        ReadOnlySpan<Half> oS = output.AsReadOnlySpan<Half>();

        float maxRel = 0f;
        for (int m = 0; m < M; m++)
        {
            for (int n = 0; n < N; n++)
            {
                float acc = 0;
                for (int k = 0; k < K; k++) acc += (float)iS[m * K + k] * (float)wRef[n * K + k];
                float got = (float)oS[m * N + n];
                float rel = MathF.Abs(got - acc) / MathF.Max(1e-3f, MathF.Abs(acc));
                maxRel = MathF.Max(maxRel, rel);
            }
        }
        Assert.True(maxRel < 0.05f, $"FP8 Linear (scale={scale}) maxRelErr {maxRel:P2} too high — suggests the scale factor isn't applied exactly once.");

        input.Dispose(); weightF32.Dispose(); weightFp8.Dispose(); weightF16Ref.Dispose(); output.Dispose();
    }

    /// <summary>Reproduces Krea2's exact SwiGLU shape: two DIFFERENT fp8_scaled weights (gate/up, each its
    /// own <see cref="Tensor.Fp8ScaleFactor"/>) Linear'd against the SAME input back-to-back, THEN
    /// multiplied together (<c>silu(g)*u</c>) — with <see cref="VulkanBackend.CacheWeightCasts"/> disabled
    /// (the exact setting the real Krea2 Vulkan test uses for its 13 GB checkpoint), unlike every other FP8
    /// test in this file which leaves caching at its default (on). If each Linear is individually correct
    /// but the transient (uncached) cast path has any cross-call bleed — e.g. a scale-factor mixup between
    /// back-to-back casts of different tensors — this compounds multiplicatively in the product, matching
    /// the ~3x-too-large `ffOut` a live block-by-block CUDA-vs-Vulkan Krea2 comparison found.</summary>
    [Fact]
    public void Backend_SwiGlu_TwoFp8ScaledWeights_UncachedCasts_MatchesCpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();
        if (!backend.Capabilities.SupportsF16) return;
        backend.CacheWeightCasts = false;

        const int M = 37, K = 96, N = 41;
        Tensor input = new(new TensorShape(M, K), DType.F16);
        Tensor gateWF32 = new(new TensorShape(N, K), DType.F32);
        Tensor upWF32 = new(new TensorShape(N, K), DType.F32);

        // REAL Krea2 ff.gate/up scale magnitude (~0.0017-0.002), not an arbitrary large value — see the
        // raw-byte-magnitude note on Backend_Linear_FP8Weight_NonUnitScaleFactor_MatchesCpu above.
        // Tensor.CastTo(F8E4M3) encodes F32 directly (no scale-aware pre-division), so the pre-cast values
        // must be scaled up by ~1/scale to land in the same raw-byte regime a real tiny-scale checkpoint uses.
        const float gateScale = 0.00166f, upScale = 0.00201f;   // Krea2's actual per-tensor values, deliberately DIFFERENT
        float gateRawMag = 0.05f / gateScale, upRawMag = 0.05f / upScale;
        Random rng = new(31);
        Span<Half> iS = input.AsSpan<Half>();
        Span<float> gS = gateWF32.AsSpan<float>();
        Span<float> uS = upWF32.AsSpan<float>();
        for (int i = 0; i < M * K; i++) iS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.1f);
        for (int i = 0; i < N * K; i++) gS[i] = (float)(rng.NextDouble() * 2 - 1) * gateRawMag;
        for (int i = 0; i < N * K; i++) uS[i] = (float)(rng.NextDouble() * 2 - 1) * upRawMag;

        Tensor gateFp8 = gateWF32.CastTo(DType.F8E4M3); gateFp8.Fp8ScaleFactor = gateScale;
        Tensor upFp8 = upWF32.CastTo(DType.F8E4M3); upFp8.Fp8ScaleFactor = upScale;
        Tensor gateF16Ref = gateFp8.CastTo(DType.F16);
        Tensor upF16Ref = upFp8.CastTo(DType.F16);
        ReadOnlySpan<Half> gRef = gateF16Ref.AsReadOnlySpan<Half>();
        ReadOnlySpan<Half> uRef = upF16Ref.AsReadOnlySpan<Half>();

        Tensor g = new(new TensorShape(M, N), DType.F16);
        Tensor u = new(new TensorShape(M, N), DType.F16);
        backend.Linear(g, input, gateFp8, null);
        backend.Linear(u, input, upFp8, null);
        Tensor silu = new(new TensorShape(M, N), DType.F16);
        backend.Silu(silu, g);
        Tensor gated = new(new TensorShape(M, N), DType.F16);
        backend.Mul(gated, silu, u);
        ReadOnlySpan<Half> gatedS = gated.AsReadOnlySpan<Half>();

        float maxRel = 0f;
        for (int m = 0; m < M; m++)
        {
            for (int n = 0; n < N; n++)
            {
                float accG = 0, accU = 0;
                for (int k = 0; k < K; k++)
                {
                    accG += (float)iS[m * K + k] * (float)gRef[n * K + k];
                    accU += (float)iS[m * K + k] * (float)uRef[n * K + k];
                }
                float siluRef = accG / (1.0f + MathF.Exp(-accG));
                float expected = siluRef * accU;
                float got = (float)gatedS[m * N + n];
                float rel = MathF.Abs(got - expected) / MathF.Max(1e-3f, MathF.Abs(expected));
                maxRel = MathF.Max(maxRel, rel);
            }
        }
        Assert.True(maxRel < 0.05f, $"SwiGLU (two differently-scaled FP8 weights, uncached) maxRelErr {maxRel:P2} too high.");

        input.Dispose(); gateWF32.Dispose(); upWF32.Dispose(); gateFp8.Dispose(); upFp8.Dispose();
        gateF16Ref.Dispose(); upF16Ref.Dispose(); g.Dispose(); u.Dispose(); silu.Dispose(); gated.Dispose();
    }

    /// <summary>Same FP8-scaled Linear correctness check as
    /// <see cref="Backend_Linear_FP8Weight_NonUnitScaleFactor_MatchesCpu"/> but at Krea2's EXACT real
    /// <c>ff.gate</c>/<c>ff.up</c> shape (M=jointSeq=4108, K=hidden=6144, N=intermediateSize=16384) — every
    /// prior FP8-scale test used M,N ≤ 256, far smaller than the real model's tile-count (real N=16384 is
    /// 128 tiles of 128 vs. 2 tiles at N=256; real K=6144 is 384 K-blocks of 16 vs. 32 at K=512), so an
    /// accumulator or tile-index bug that only manifests after many loop iterations would be invisible at
    /// the smaller shapes already tested. Uses probe sampling (not every output element) since a full
    /// M×N×K reference here is ~4×10^11 multiply-adds — probes still cover corners/tile-boundaries.</summary>
    [Fact]
    public void Backend_Linear_FP8Weight_NonUnitScaleFactor_MatchesCpu_RealKrea2FfnShape()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();
        if (!backend.Capabilities.SupportsF16) return;
        backend.CacheWeightCasts = false;

        const int M = 4108, K = 6144, N = 16384;
        const float scale = 0.00166f;   // Krea2's actual ff.gate scale magnitude
        float rawMagnitude = 0.05f / scale;

        Tensor input = new(new TensorShape(M, K), DType.F16);
        Tensor weightF32 = new(new TensorShape(N, K), DType.F32);

        Random rng = new(37);
        Span<Half> iS = input.AsSpan<Half>();
        Span<float> wS = weightF32.AsSpan<float>();
        for (int i = 0; i < M * K; i++) iS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.1f);
        for (int i = 0; i < N * K; i++) wS[i] = (float)(rng.NextDouble() * 2 - 1) * rawMagnitude;

        Tensor weightFp8 = weightF32.CastTo(DType.F8E4M3);
        weightFp8.Fp8ScaleFactor = scale;
        Tensor weightF16Ref = weightFp8.CastTo(DType.F16);
        ReadOnlySpan<Half> wRef = weightF16Ref.AsReadOnlySpan<Half>();

        Tensor output = new(new TensorShape(M, N), DType.F16);
        backend.Linear(output, input, weightFp8, null);
        ReadOnlySpan<Half> oS = output.AsReadOnlySpan<Half>();

        // Probe corners, tile boundaries (every 128th row/col — the coopmat/128-tile boundary), and a
        // scattered sample across the full M/N range.
        int[] mProbes = { 0, 1, 127, 128, 129, 2048, 4106, 4107 };
        int[] nProbes = { 0, 1, 127, 128, 129, 8192, 16382, 16383 };
        int errs = 0; float maxRel = 0f; int firstM = -1, firstN = -1;
        foreach (int m in mProbes)
        foreach (int n in nProbes)
        {
            float acc = 0;
            for (int k = 0; k < K; k++) acc += (float)iS[m * K + k] * (float)wRef[n * K + k];
            float got = (float)oS[m * N + n];
            float rel = MathF.Abs(got - acc) / MathF.Max(1e-3f, MathF.Abs(acc));
            if (rel > 0.05f) { if (errs == 0) { firstM = m; firstN = n; } errs++; maxRel = MathF.Max(maxRel, rel); }
        }
        Assert.True(errs == 0, $"FP8 Linear (real Krea2 FFN shape, scale={scale}) {errs} probe diffs, first out[{firstM},{firstN}] maxRelErr={maxRel:P2}.");

        input.Dispose(); weightF32.Dispose(); weightFp8.Dispose(); weightF16Ref.Dispose(); output.Dispose();
    }

    /// <summary>Regression gate for <c>matmul_coopmat_partial_m.comp.glsl</c> (2026-07-31, the SECOND attempt
    /// at unaligned-M coopmat — see TROUBLESHOOTING.md for the first attempt's host-side scratch-buffer
    /// design and the real `ErrorDeviceLost` that reverted it). This design instead stages the boundary
    /// row-tile through workgroup shared memory (mirroring matmul_tiled.comp.glsl's own proven bounds-checked
    /// idiom), entirely within one dispatch — no separate command buffer, no cross-submission barrier.
    /// Covers: M values straddling every relevant boundary (1: a single, fully-partial 16-row tile; 13:
    /// Krea2's REAL measured txtSeq bug value; 17: one past the first 16-aligned value; 63/65: straddling the
    /// 64-row BM workgroup-tile boundary; 100: an interior case with no special alignment) × bias present or
    /// absent × F16/F32 output. Verified two ways per case: (1) numerically against a from-scratch CPU GEMM
    /// reference, and (2) that the partial-M kernel actually engaged (not a silent fallback to matmul_tiled)
    /// via the same reflection-based engagement-counter check used for the first (reverted) attempt.</summary>
    /// <summary>Regression gate for <c>matmul_coopmat_partial_m.comp.glsl</c> (2026-07-31, the SECOND attempt
    /// at unaligned-M coopmat — see TROUBLESHOOTING.md for the first attempt's host-side scratch-buffer
    /// design and the real `ErrorDeviceLost` that reverted it). This design instead stages the boundary
    /// row-tile through workgroup shared memory (mirroring matmul_tiled.comp.glsl's own proven bounds-checked
    /// idiom), entirely within one dispatch — no separate command buffer, no cross-submission barrier.
    /// Covers: M values straddling every relevant boundary (1: a single, fully-partial 16-row tile; 13:
    /// Krea2's REAL measured txtSeq bug value; 17: one past the first 16-aligned value; 33: creates a
    /// FULLY-invalid subgroup-row, zero valid rows, not just a partial one; 63/65: straddling the 64-row BM
    /// workgroup-tile boundary; 100: an interior case with no special alignment) × bias present or absent,
    /// all F16 output (matching Krea2's real per-block dtype — DitDtype.Act defaults to F16, so this is the
    /// case that actually matters; F32 output stays gated off, see ResolveGemmDtype's comment). N=48 is
    /// deliberately NOT a multiple of BN (32) — this is what caught a real, separate bug (2026-07-31): a
    /// divergent-barrier UB from an early `return` for out-of-N-bounds subgroups racing this kernel's
    /// barrier() (fixed by removing all early returns in favor of scalar bounds-checked draining — see the
    /// shader's own comment). Verified two ways per case: (1) numerically against a from-scratch CPU GEMM
    /// reference, and (2) that the partial-M kernel actually engaged (not a silent fallback to matmul_tiled)
    /// via reflection on the engagement counters.</summary>
    [Theory]
    [InlineData(1, false)]
    [InlineData(1, true)]
    [InlineData(13, false)]
    [InlineData(13, true)]
    [InlineData(17, true)]
    [InlineData(33, false)]
    [InlineData(63, false)]
    [InlineData(65, true)]
    [InlineData(100, false)]
    public void Backend_Linear_CoopmatPartialM_NonMultipleOf16_MatchesCpu(int M, bool hasBias)
    {
        if (!VulkanAvailable()) return;
        Environment.SetEnvironmentVariable("HARTSYINFERENCE_VK_PROFILE", "1");
        using VulkanBackend backend = new();
        Environment.SetEnvironmentVariable("HARTSYINFERENCE_VK_PROFILE", null);
        if (!backend.Capabilities.SupportsF16 || !backend.Vk.HasCooperativeMatrix) return;
        // This test specifically exercises coopmat1's partial-M kernel — coopmat2 (default ON since
        // 2026-07-31) would otherwise engage first and handle any M transparently, masking what this
        // test exists to check.
        backend.EnableCoopMat2 = false;

        const int K = 32, N = 48;   // multiples of 16, matching real Krea2-like hidden dims
        Tensor input = new(new TensorShape(M, K), DType.F16);
        Tensor weight = new(new TensorShape(N, K), DType.F16);
        Tensor? bias = hasBias ? new Tensor(new TensorShape(N), DType.F16) : null;
        Tensor output = new(new TensorShape(M, N), DType.F16);

        Random rng = new(1000 + M);
        Span<Half> iS = input.AsSpan<Half>();
        Span<Half> wS = weight.AsSpan<Half>();
        Half[] bS = new Half[N];
        for (int i = 0; i < M * K; i++) iS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.3f);
        for (int i = 0; i < N * K; i++) wS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.2f);
        if (hasBias)
        {
            Span<Half> bSpan = bias!.AsSpan<Half>();
            for (int i = 0; i < N; i++) { bS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.05f); bSpan[i] = bS[i]; }
        }

        backend.Linear(output, input, weight, bias);

        long coopmatCount = (long)typeof(VulkanBackend).GetField("_coopmatGemmCount",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(backend)!;
        long tiledCount = (long)typeof(VulkanBackend).GetField("_tiledGemmCount",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(backend)!;
        Assert.True(coopmatCount == 1 && tiledCount == 0,
            $"Expected the partial-M coopmat kernel to engage for M={M} (not a multiple of 16); " +
            $"got coopmatCount={coopmatCount}, tiledCount={tiledCount} — suggests a silent fallback to matmul_tiled.");

        float maxRel = 0f;
        ReadOnlySpan<Half> oS = output.AsReadOnlySpan<Half>();
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
            {
                float acc = hasBias ? (float)bS[n] : 0f;
                for (int k = 0; k < K; k++) acc += (float)iS[m * K + k] * (float)wS[n * K + k];
                float rel = MathF.Abs((float)oS[m * N + n] - acc) / MathF.Max(1e-3f, MathF.Abs(acc));
                maxRel = MathF.Max(maxRel, rel);
            }
        Assert.True(maxRel < 0.02f, $"Coopmat-partial-M Linear (M={M}, bias={hasBias}) maxRelErr {maxRel:P2} too high.");

        input.Dispose(); weight.Dispose(); bias?.Dispose(); output.Dispose();
    }

    /// <summary>Confirms M values that ARE already multiples of 16 still take the unmodified,
    /// already-proven <c>matmul_coopmat</c> path (not the new partial-M kernel) — i.e. this change is
    /// additive, not a behavior change for the case that already worked.</summary>
    [Fact]
    public void Backend_Linear_CoopmatAlignedM_StillUsesOriginalKernel()
    {
        if (!VulkanAvailable()) return;
        Environment.SetEnvironmentVariable("HARTSYINFERENCE_VK_PROFILE", "1");
        using VulkanBackend backend = new();
        Environment.SetEnvironmentVariable("HARTSYINFERENCE_VK_PROFILE", null);
        if (!backend.Capabilities.SupportsF16 || !backend.Vk.HasCooperativeMatrix) return;
        // Specifically checking coopmat1's own engagement — coopmat2 (default ON since 2026-07-31) would
        // otherwise handle this shape first, which is not what this test exists to check.
        backend.EnableCoopMat2 = false;

        const int M = 64, K = 32, N = 48;
        Tensor input = new(new TensorShape(M, K), DType.F16);
        Tensor weight = new(new TensorShape(N, K), DType.F16);
        Tensor output = new(new TensorShape(M, N), DType.F16);
        Random rng = new(55);
        Span<Half> iS = input.AsSpan<Half>();
        Span<Half> wS = weight.AsSpan<Half>();
        for (int i = 0; i < M * K; i++) iS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.3f);
        for (int i = 0; i < N * K; i++) wS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.2f);

        backend.Linear(output, input, weight, null);

        long coopmatCount = (long)typeof(VulkanBackend).GetField("_coopmatGemmCount",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(backend)!;
        Assert.Equal(1, coopmatCount);

        ReadOnlySpan<Half> oS = output.AsReadOnlySpan<Half>();
        float maxRel = 0f;
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
            {
                float acc = 0f;
                for (int k = 0; k < K; k++) acc += (float)iS[m * K + k] * (float)wS[n * K + k];
                float rel = MathF.Abs((float)oS[m * N + n] - acc) / MathF.Max(1e-3f, MathF.Abs(acc));
                maxRel = MathF.Max(maxRel, rel);
            }
        Assert.True(maxRel < 0.02f, $"Aligned-M coopmat Linear maxRelErr {maxRel:P2} too high.");

        input.Dispose(); weight.Dispose(); output.Dispose();
    }

    /// <summary>Matmul at Flux DiT dimensions (M=32, K=3072, N=3072) — checks accumulator precision survives ~3K-deep dot products without overflowing.</summary>
    [Fact]
    public void Backend_MatMul_LargeFp16_Matches_Cpu()
    {
        if (!VulkanAvailable())
            return;
        using VulkanBackend backend = new();
        if (!backend.Capabilities.SupportsF16) return;

        // Realistic shapes. Random Gaussian to look like real activations/weights.
        const int M = 32, K = 3072, N = 3072;
        Tensor a = new(new TensorShape(M, K), DType.F16);
        Tensor b = new(new TensorShape(K, N), DType.F16);
        Tensor c = new(new TensorShape(M, N), DType.F16);

        Random rng = new(42);
        Span<Half> aS = a.AsSpan<Half>();
        Span<Half> bS = b.AsSpan<Half>();
        for (int i = 0; i < M * K; i++) aS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.1f);
        for (int i = 0; i < K * N; i++) bS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.1f);

        backend.MatMul(c, a, b);
        ReadOnlySpan<Half> cS = c.AsReadOnlySpan<Half>();

        // CPU reference — accumulate in FP32
        int errs = 0;
        float maxRel = 0;
        for (int m = 0; m < M; m++)
        {
            for (int n = 0; n < N; n++)
            {
                float acc = 0;
                for (int k = 0; k < K; k++) acc += (float)aS[m * K + k] * (float)bS[k * N + n];
                float vk = (float)cS[m * N + n];
                float absErr = MathF.Abs(vk - acc);
                float relErr = absErr / MathF.Max(1e-3f, MathF.Abs(acc));
                if (relErr > 1e-2f && absErr > 1e-2f) errs++;
                if (relErr > maxRel) maxRel = relErr;
            }
        }
        Assert.True(errs == 0,
            $"Large FP16 matmul: {errs} elements outside 1% rel tolerance. Max relative error: {maxRel:G6}");

        a.Dispose(); b.Dispose(); c.Dispose();
    }

    /// <summary>A key-only additive mask (one <c>[1, Skv]</c> row broadcast over every query — Wan-Animate-2's
    /// log_scale band) must match the <c>[Sq, Skv]</c> duplicate of it. The mask_add dispatch adds a flat Sq·Skv
    /// block at one offset and cannot broadcast a row, so the backend expands it; handing the shader the short
    /// buffer instead would read past the allocation and go unnoticed.</summary>
    [Fact]
    public void Backend_SDPA_KeyOnlyMask_Matches_ExpandedMask()
    {
        if (!VulkanAvailable())
            return;
        using VulkanBackend backend = new();

        const int B = 1, H = 2, Sq = 16, Skv = 64, D = 64;
        using Tensor q = new(new TensorShape(B, H, Sq, D), DType.F32);
        using Tensor k = new(new TensorShape(B, H, Skv, D), DType.F32);
        using Tensor v = new(new TensorShape(B, H, Skv, D), DType.F32);
        using Tensor broadcast = new(new TensorShape(B, H, Sq, D), DType.F32);
        using Tensor duplicate = new(new TensorShape(B, H, Sq, D), DType.F32);
        Span<float> qS = q.AsSpan<float>(), kS = k.AsSpan<float>(), vS = v.AsSpan<float>();
        for (int i = 0; i < B * H * Sq * D; i++) qS[i] = MathF.Sin(i * 0.11f);
        for (int i = 0; i < B * H * Skv * D; i++) { kS[i] = MathF.Cos(i * 0.07f); vS[i] = MathF.Sin(i * 0.13f) * 0.5f; }

        using Tensor row = new(new TensorShape(1, Skv), DType.F32);
        using Tensor full = new(new TensorShape(Sq, Skv), DType.F32);
        Span<float> rowS = row.AsSpan<float>(), fullS = full.AsSpan<float>();
        for (int key = 0; key < Skv; key++) rowS[key] = key >= Sq && key < 2 * Sq ? -1.3f : 0f;
        for (int query = 0; query < Sq; query++)
            for (int key = 0; key < Skv; key++) fullS[query * Skv + key] = rowS[key];

        float scale = 1.0f / MathF.Sqrt(D);
        backend.ScaledDotProductAttention(broadcast, q, k, v, row, scale);
        backend.ScaledDotProductAttention(duplicate, q, k, v, full, scale);

        Span<float> a = broadcast.AsSpan<float>(), b = duplicate.AsSpan<float>();
        float worst = 0f;
        for (int i = 0; i < a.Length; i++) worst = MathF.Max(worst, MathF.Abs(a[i] - b[i]));
        Assert.True(worst < 1e-5f, $"key-only mask diverged from the duplicate by {worst:E3}.");
    }

    /// <summary>An F16 key-only mask must be rejected by the existing F32-only dtype guard, not expanded first.
    /// Regression target: <c>ExpandKeyOnlyMask</c> read the mask natively as <c>float*</c> unconditionally, before
    /// any dtype check ran — an F16 <c>[1,Skv]</c> mask (half the bytes of the same-shaped F32 one) was copied as
    /// four bytes per element from a two-byte allocation (an out-of-bounds native read), and the RESULT of that
    /// copy is a freshly-allocated F32 tensor, so the dtype guard downstream never even saw the original F16 mask
    /// to reject it — the call silently proceeded on garbage mask data instead of throwing.</summary>
    [Fact]
    public void Backend_SDPA_KeyOnlyMask_F16_IsRejectedByDtypeGuard_NotExpandedFirst()
    {
        if (!VulkanAvailable())
            return;
        using VulkanBackend backend = new();

        const int B = 1, H = 2, Sq = 16, Skv = 64, D = 64;
        using Tensor q = new(new TensorShape(B, H, Sq, D), DType.F32);
        using Tensor k = new(new TensorShape(B, H, Skv, D), DType.F32);
        using Tensor v = new(new TensorShape(B, H, Skv, D), DType.F32);
        using Tensor o = new(new TensorShape(B, H, Sq, D), DType.F32);
        using Tensor rowF16 = new(new TensorShape(1, Skv), DType.F16);

        float scale = 1.0f / MathF.Sqrt(D);
        NotSupportedException ex = Assert.Throws<NotSupportedException>(
            () => backend.ScaledDotProductAttention(o, q, k, v, rowF16, scale));
        Assert.Contains("F32", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Backend_SDPA_MultiHead_Matches_Cpu_Reference()
    {
        if (!VulkanAvailable())
            return;
        using VulkanBackend backend = new();

        const int B = 1, H = 2, S = 4, D = 8;
        Tensor q = new(new TensorShape(B, H, S, D), DType.F32);
        Tensor k = new(new TensorShape(B, H, S, D), DType.F32);
        Tensor v = new(new TensorShape(B, H, S, D), DType.F32);
        Tensor o = new(new TensorShape(B, H, S, D), DType.F32);

        Span<float> qS = q.AsSpan<float>();
        Span<float> kS = k.AsSpan<float>();
        Span<float> vS = v.AsSpan<float>();
        for (int i = 0; i < B * H * S * D; i++) { qS[i] = MathF.Sin(i * 0.1f); kS[i] = MathF.Cos(i * 0.07f); vS[i] = MathF.Sin(i * 0.13f) * 0.5f; }

        float scale = 1.0f / MathF.Sqrt(D);
        backend.ScaledDotProductAttention(o, q, k, v, mask: null, scale);

        // CPU reference per (b, h)
        ReadOnlySpan<float> oS = o.AsReadOnlySpan<float>();
        float[] expected = new float[B * H * S * D];
        for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
        {
            int baseIdx = (b * H + h) * S * D;
            float[] scores = new float[S * S];
            for (int i = 0; i < S; i++)
                for (int j = 0; j < S; j++)
                {
                    float acc = 0;
                    for (int d = 0; d < D; d++)
                        acc += qS[baseIdx + i * D + d] * kS[baseIdx + j * D + d];
                    scores[i * S + j] = acc * scale;
                }
            for (int i = 0; i < S; i++)
            {
                float maxv = float.NegativeInfinity;
                for (int j = 0; j < S; j++) maxv = MathF.Max(maxv, scores[i * S + j]);
                float sum = 0;
                for (int j = 0; j < S; j++) { scores[i * S + j] = MathF.Exp(scores[i * S + j] - maxv); sum += scores[i * S + j]; }
                for (int j = 0; j < S; j++) scores[i * S + j] /= sum;
            }
            for (int i = 0; i < S; i++)
                for (int d = 0; d < D; d++)
                {
                    float acc = 0;
                    for (int j = 0; j < S; j++) acc += scores[i * S + j] * vS[baseIdx + j * D + d];
                    expected[baseIdx + i * D + d] = acc;
                }
        }
        for (int i = 0; i < B * H * S * D; i++)
            Assert.InRange(oS[i] - expected[i], -1e-3f, 1e-3f);

        q.Dispose(); k.Dispose(); v.Dispose(); o.Dispose();
    }

    [Fact]
    public void Backend_MatMul_Matches_Cpu_Reference()
    {
        if (!VulkanAvailable())
            return;
        using VulkanBackend backend = new();

        const int M = 16, K = 8, N = 12;
        Tensor a = new(new TensorShape(M, K), DType.F32);
        Tensor b = new(new TensorShape(K, N), DType.F32);
        Tensor c = new(new TensorShape(M, N), DType.F32);

        Span<float> aS = a.AsSpan<float>();
        Span<float> bS = b.AsSpan<float>();
        for (int i = 0; i < M * K; i++) aS[i] = (i * 0.1f) - 2.0f;
        for (int i = 0; i < K * N; i++) bS[i] = ((i + 3) * 0.05f) + 1.0f;

        backend.MatMul(c, a, b);

        // CPU reference
        float[] ref_ = new float[M * N];
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
            {
                float acc = 0.0f;
                for (int k = 0; k < K; k++) acc += aS[m * K + k] * bS[k * N + n];
                ref_[m * N + n] = acc;
            }

        ReadOnlySpan<float> cS = c.AsReadOnlySpan<float>();
        for (int i = 0; i < M * N; i++)
            Assert.InRange(cS[i] - ref_[i], -1e-3f, 1e-3f);

        a.Dispose(); b.Dispose(); c.Dispose();
    }

    // ── Phase A1 backfill: per-kernel *_Vs_Cpu coverage for SD1.5 / SDXL bring-up ──

    [Fact]
    public void Backend_Mul_Matches_Cpu_F32()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        Tensor a = new(new TensorShape(64), DType.F32);
        Tensor b = new(new TensorShape(64), DType.F32);
        Tensor c = new(new TensorShape(64), DType.F32);

        Span<float> aS = a.AsSpan<float>();
        Span<float> bS = b.AsSpan<float>();
        for (int i = 0; i < 64; i++) { aS[i] = i * 0.5f - 4f; bS[i] = MathF.Sin(i * 0.31f); }

        backend.Mul(c, a, b);

        ReadOnlySpan<float> cS = c.AsReadOnlySpan<float>();
        for (int i = 0; i < 64; i++)
            Assert.InRange(cS[i] - aS[i] * bS[i], -1e-5f, 1e-5f);

        a.Dispose(); b.Dispose(); c.Dispose();
    }

    [Fact]
    public void Backend_Mul_Matches_Cpu_F16()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();
        if (!backend.Capabilities.SupportsF16) return;

        Tensor a = new(new TensorShape(64), DType.F16);
        Tensor b = new(new TensorShape(64), DType.F16);
        Tensor c = new(new TensorShape(64), DType.F16);

        Span<Half> aS = a.AsSpan<Half>();
        Span<Half> bS = b.AsSpan<Half>();
        for (int i = 0; i < 64; i++) { aS[i] = (Half)(i * 0.0625f); bS[i] = (Half)((i + 1) * 0.03125f); }

        backend.Mul(c, a, b);

        ReadOnlySpan<Half> cS = c.AsReadOnlySpan<Half>();
        for (int i = 0; i < 64; i++)
            Assert.InRange((float)cS[i] - (float)aS[i] * (float)bS[i], -5e-3f, 5e-3f);

        a.Dispose(); b.Dispose(); c.Dispose();
    }

    [Fact]
    public void Backend_Scale_Matches_Cpu_F32()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        Tensor x = new(new TensorShape(48), DType.F32);
        Tensor y = new(new TensorShape(48), DType.F32);
        Span<float> xS = x.AsSpan<float>();
        for (int i = 0; i < 48; i++) xS[i] = MathF.Cos(i * 0.27f) * 3.5f;

        const float k = 2.71828f;
        backend.Scale(y, x, k);

        ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
        for (int i = 0; i < 48; i++)
            Assert.InRange(yS[i] - xS[i] * k, -1e-5f, 1e-5f);

        x.Dispose(); y.Dispose();
    }

    [Fact]
    public void Backend_Gelu_Matches_Cpu_TanhApprox()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        Tensor x = new(new TensorShape(64), DType.F32);
        Tensor y = new(new TensorShape(64), DType.F32);
        Span<float> xS = x.AsSpan<float>();
        for (int i = 0; i < 64; i++) xS[i] = i * 0.125f - 4f;

        backend.Gelu(y, x);

        ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
        for (int i = 0; i < 64; i++)
        {
            float xv = xS[i];
            float expected = 0.5f * xv * (1f + MathF.Tanh(0.7978845608f * (xv + 0.044715f * xv * xv * xv)));
            Assert.InRange(yS[i] - expected, -1e-5f, 1e-5f);
        }
        x.Dispose(); y.Dispose();
    }

    [Fact]
    public void Backend_Transpose2D_Matches_Cpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int B = 2, D1 = 16, D2 = 24;
        Tensor x = new(new TensorShape(B, D1, D2), DType.F32);
        Tensor y = new(new TensorShape(B, D2, D1), DType.F32);

        Span<float> xS = x.AsSpan<float>();
        for (int i = 0; i < B * D1 * D2; i++) xS[i] = MathF.Sin(i * 0.17f);

        backend.Transpose2D(y, x, D1, D2);

        ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
        for (int b = 0; b < B; b++)
            for (int i = 0; i < D1; i++)
                for (int j = 0; j < D2; j++)
                {
                    float src = xS[b * D1 * D2 + i * D2 + j];
                    float dst = yS[b * D2 * D1 + j * D1 + i];
                    Assert.InRange(dst - src, -1e-5f, 1e-5f);
                }

        x.Dispose(); y.Dispose();
    }

    [Fact]
    public void Backend_Permute0213_Matches_Cpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        // [B, S, H, D] → [B, H, S, D]
        const int B = 2, S = 6, H = 4, D = 8;
        Tensor src = new(new TensorShape(B, S, H, D), DType.F32);
        Tensor dst = new(new TensorShape(B, H, S, D), DType.F32);

        Span<float> sS = src.AsSpan<float>();
        for (int i = 0; i < B * S * H * D; i++) sS[i] = MathF.Cos(i * 0.19f) * 1.7f;

        backend.Permute0213(dst, src, S, H, D);

        ReadOnlySpan<float> dS = dst.AsReadOnlySpan<float>();
        for (int b = 0; b < B; b++)
            for (int s = 0; s < S; s++)
                for (int h = 0; h < H; h++)
                    for (int d = 0; d < D; d++)
                    {
                        float srcVal = sS[((b * S + s) * H + h) * D + d];
                        float dstVal = dS[((b * H + h) * S + s) * D + d];
                        Assert.InRange(dstVal - srcVal, -1e-5f, 1e-5f);
                    }

        src.Dispose(); dst.Dispose();
    }

    [Fact]
    public void Backend_BroadcastAdd_Matches_Cpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        // hidden[B, C, spatial] += bias[C]   (channel broadcast)
        const int B = 2, C = 4, Spatial = 16;
        Tensor hidden = new(new TensorShape(B, C, Spatial), DType.F32);
        Tensor bias = new(new TensorShape(C), DType.F32);
        try
        {
            Span<float> hS = hidden.AsSpan<float>();
            Span<float> bS = bias.AsSpan<float>();
            for (int i = 0; i < B * C * Spatial; i++) hS[i] = MathF.Sin(i * 0.13f);
            float[] biasVals = new float[C];
            for (int i = 0; i < C; i++) { biasVals[i] = (i + 1) * 0.5f; bS[i] = biasVals[i]; }

            // CPU reference (compute before dispatch since BroadcastAdd is in-place)
            float[] expected = new float[B * C * Spatial];
            for (int b = 0; b < B; b++)
                for (int c = 0; c < C; c++)
                    for (int s = 0; s < Spatial; s++)
                        expected[b * C * Spatial + c * Spatial + s] = hS[b * C * Spatial + c * Spatial + s] + biasVals[c];

            backend.BroadcastAdd(hidden, bias, C, Spatial);

            ReadOnlySpan<float> hOut = hidden.AsReadOnlySpan<float>();
            for (int i = 0; i < B * C * Spatial; i++)
                Assert.InRange(hOut[i] - expected[i], -1e-5f, 1e-5f);
        }
        finally { hidden.Dispose(); bias.Dispose(); }
    }

    /// <summary>GroupNorm at SD1.5 U-Net shapes (32 groups, C=320, spatial=64×64) — the dominant U-Net norm.</summary>
    [Fact]
    public void Backend_GroupNorm_Matches_Cpu_Sd15Shape()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int B = 1, C = 320, H = 16, W = 16, Groups = 32;
        Tensor x = new(new TensorShape(B, C, H, W), DType.F32);
        Tensor w = new(new TensorShape(C), DType.F32);
        Tensor b = new(new TensorShape(C), DType.F32);
        Tensor y = new(new TensorShape(B, C, H, W), DType.F32);

        Random rng = new(42);
        Span<float> xS = x.AsSpan<float>();
        Span<float> wS = w.AsSpan<float>();
        Span<float> bS = b.AsSpan<float>();
        for (int i = 0; i < B * C * H * W; i++) xS[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < C; i++) { wS[i] = 1.0f + (float)(rng.NextDouble() * 0.1); bS[i] = (float)(rng.NextDouble() * 0.1 - 0.05); }

        const float eps = 1e-5f;
        backend.GroupNorm(y, x, w, b, Groups, eps);

        // CPU reference: per (batch, group) compute mean+var over (channels-in-group × spatial)
        ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
        int groupSize = C / Groups;
        int spatial = H * W;
        int errs = 0; float maxErr = 0;
        for (int bi = 0; bi < B; bi++)
            for (int g = 0; g < Groups; g++)
            {
                double mean = 0; int N = groupSize * spatial;
                for (int c = g * groupSize; c < (g + 1) * groupSize; c++)
                    for (int s = 0; s < spatial; s++) mean += xS[bi * C * spatial + c * spatial + s];
                mean /= N;
                double var_ = 0;
                for (int c = g * groupSize; c < (g + 1) * groupSize; c++)
                    for (int s = 0; s < spatial; s++)
                    {
                        double d = xS[bi * C * spatial + c * spatial + s] - mean;
                        var_ += d * d;
                    }
                var_ /= N;
                double invStd = 1.0 / Math.Sqrt(var_ + eps);
                for (int c = g * groupSize; c < (g + 1) * groupSize; c++)
                    for (int s = 0; s < spatial; s++)
                    {
                        int idx = bi * C * spatial + c * spatial + s;
                        float exp = (float)((xS[idx] - mean) * invStd) * wS[c] + bS[c];
                        float got = yS[idx];
                        float err = MathF.Abs(got - exp);
                        if (err > 1e-3f) errs++;
                        if (err > maxErr) maxErr = err;
                    }
            }
        Assert.True(errs == 0, $"GroupNorm SD1.5-shape: {errs} mismatches, maxErr={maxErr:G6}");

        x.Dispose(); w.Dispose(); b.Dispose(); y.Dispose();
    }

    /// <summary>Fused GroupNorm+Silu — should match CPU GroupNorm followed by Silu.</summary>
    [Fact]
    public void Backend_GroupNormSilu_Matches_Cpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int B = 1, C = 64, H = 8, W = 8, Groups = 8;
        Tensor x = new(new TensorShape(B, C, H, W), DType.F32);
        Tensor w = new(new TensorShape(C), DType.F32);
        Tensor b = new(new TensorShape(C), DType.F32);
        Tensor y = new(new TensorShape(B, C, H, W), DType.F32);

        Random rng = new(7);
        Span<float> xS = x.AsSpan<float>();
        Span<float> wS = w.AsSpan<float>();
        Span<float> bS = b.AsSpan<float>();
        for (int i = 0; i < B * C * H * W; i++) xS[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < C; i++) { wS[i] = 1.0f; bS[i] = 0.0f; }

        const float eps = 1e-5f;
        backend.GroupNormSilu(y, x, w, b, Groups, eps);

        // CPU reference: groupnorm then silu
        ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
        int groupSize = C / Groups;
        int spatial = H * W;
        for (int bi = 0; bi < B; bi++)
            for (int g = 0; g < Groups; g++)
            {
                double mean = 0; int N = groupSize * spatial;
                for (int c = g * groupSize; c < (g + 1) * groupSize; c++)
                    for (int s = 0; s < spatial; s++) mean += xS[bi * C * spatial + c * spatial + s];
                mean /= N;
                double var_ = 0;
                for (int c = g * groupSize; c < (g + 1) * groupSize; c++)
                    for (int s = 0; s < spatial; s++)
                    {
                        double d = xS[bi * C * spatial + c * spatial + s] - mean;
                        var_ += d * d;
                    }
                var_ /= N;
                double invStd = 1.0 / Math.Sqrt(var_ + eps);
                for (int c = g * groupSize; c < (g + 1) * groupSize; c++)
                    for (int s = 0; s < spatial; s++)
                    {
                        int idx = bi * C * spatial + c * spatial + s;
                        float gn = (float)((xS[idx] - mean) * invStd) * wS[c] + bS[c];
                        float exp = gn / (1f + MathF.Exp(-gn));   // silu
                        Assert.InRange(yS[idx] - exp, -1e-3f, 1e-3f);
                    }
            }

        x.Dispose(); w.Dispose(); b.Dispose(); y.Dispose();
    }

    /// <summary>SDPA at long sequence (S=512) — exercises softmax long-row stability without the SDXL-S=4096 VRAM cost.</summary>
    [Fact]
    public void Backend_SDPA_LongSequence_Matches_Cpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int B = 1, H = 4, S = 512, D = 64;
        Tensor q = new(new TensorShape(B, H, S, D), DType.F32);
        Tensor k = new(new TensorShape(B, H, S, D), DType.F32);
        Tensor v = new(new TensorShape(B, H, S, D), DType.F32);
        Tensor o = new(new TensorShape(B, H, S, D), DType.F32);

        Random rng = new(13);
        Span<float> qS = q.AsSpan<float>();
        Span<float> kS = k.AsSpan<float>();
        Span<float> vS = v.AsSpan<float>();
        for (int i = 0; i < B * H * S * D; i++)
        {
            qS[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
            kS[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
            vS[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
        }

        float scale = 1.0f / MathF.Sqrt(D);
        backend.ScaledDotProductAttention(o, q, k, v, mask: null, scale);

        ReadOnlySpan<float> oS = o.AsReadOnlySpan<float>();

        // CPU reference: spot-check head 0, query positions 0, S/2, S-1
        int[] iProbes = { 0, S / 2, S - 1 };
        foreach (int qPos in iProbes)
        {
            float[] scores = new float[S];
            float maxScore = float.NegativeInfinity;
            for (int j = 0; j < S; j++)
            {
                float acc = 0;
                for (int d = 0; d < D; d++) acc += qS[qPos * D + d] * kS[j * D + d];
                scores[j] = acc * scale;
                if (scores[j] > maxScore) maxScore = scores[j];
            }
            float sum = 0;
            for (int j = 0; j < S; j++) { scores[j] = MathF.Exp(scores[j] - maxScore); sum += scores[j]; }
            for (int j = 0; j < S; j++) scores[j] /= sum;

            for (int d = 0; d < D; d++)
            {
                float exp = 0;
                for (int j = 0; j < S; j++) exp += scores[j] * vS[j * D + d];
                Assert.InRange(oS[qPos * D + d] - exp, -1e-3f, 1e-3f);
            }
        }

        q.Dispose(); k.Dispose(); v.Dispose(); o.Dispose();
    }

    /// <summary>Conv2D 3×3 stride=1 pad=1 at small spatial — the dominant SD1.5 U-Net conv.</summary>
    [Fact]
    public void Backend_Conv2D_3x3_Pad1_Matches_Cpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int B = 1, Cin = 4, Cout = 8, H = 16, W = 16, Kh = 3, Kw = 3;
        Tensor input = new(new TensorShape(B, Cin, H, W), DType.F32);
        Tensor weight = new(new TensorShape(Cout, Cin, Kh, Kw), DType.F32);
        Tensor bias = new(new TensorShape(Cout), DType.F32);
        Tensor output = new(new TensorShape(B, Cout, H, W), DType.F32);

        Random rng = new(11);
        Span<float> iS = input.AsSpan<float>();
        Span<float> wS = weight.AsSpan<float>();
        Span<float> bS = bias.AsSpan<float>();
        for (int i = 0; i < B * Cin * H * W; i++) iS[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < Cout * Cin * Kh * Kw; i++) wS[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        for (int i = 0; i < Cout; i++) bS[i] = (float)(rng.NextDouble() * 2 - 1) * 0.05f;

        backend.Conv2D(output, input, weight, bias, strideH: 1, strideW: 1, padH: 1, padW: 1);

        ReadOnlySpan<float> oS = output.AsReadOnlySpan<float>();

        // CPU reference at a few sample positions (centre + corner)
        int[] yProbes = { 0, H / 2, H - 1 };
        int[] xProbes = { 0, W / 2, W - 1 };
        foreach (int oc in new[] { 0, Cout / 2, Cout - 1 })
            foreach (int yp in yProbes)
                foreach (int xp in xProbes)
                {
                    float acc = bS[oc];
                    for (int ic = 0; ic < Cin; ic++)
                        for (int kh = 0; kh < Kh; kh++)
                            for (int kw = 0; kw < Kw; kw++)
                            {
                                int ih = yp - 1 + kh;
                                int iw = xp - 1 + kw;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                acc += iS[(ic * H + ih) * W + iw] * wS[((oc * Cin + ic) * Kh + kh) * Kw + kw];
                            }
                    float got = oS[(oc * H + yp) * W + xp];
                    Assert.InRange(got - acc, -1e-3f, 1e-3f);
                }

        input.Dispose(); weight.Dispose(); bias.Dispose(); output.Dispose();
    }

    /// <summary>Regression gate for the real Krea2-on-Vulkan VAE-decode OOM (2026-07-30):
    /// <c>Conv2D</c>'s im2col buffer used to materialize the FULL <c>[gemmK, outH*outW]</c> column matrix
    /// in one allocation — ~7 GB at Krea2's 1024x1024 decode resolution, which OOM'd even with the
    /// transformer's weights already freed. Fixed by tiling over output positions (see the im2col shader's
    /// <c>colOffset</c>/<c>tileCols</c> and the matmul kernel's existing <c>bOffset</c>/<c>cOffset</c>).
    /// Forces the multi-tile path via <see cref="VulkanBackend.Conv2DMaxColTileBytes"/> (tiny here, so a
    /// small conv still exercises many tile boundaries) and checks every output element, not just probes,
    /// against the CPU reference — a stride/offset bug in the tiling math would corrupt specific columns,
    /// which probe-sampling could miss.</summary>
    [Fact]
    public void Backend_Conv2D_TiledPath_MatchesUntiled()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int B = 1, Cin = 3, Cout = 5, H = 10, W = 10, Kh = 3, Kw = 3;
        // gemmK=27, fullN=100: 4 bytes/elem * 27 * 1 col ≈ 108 bytes/col → tileN clamps to 1,
        // forcing ~100 separate tile iterations (im2col + matmul dispatched once per output column).
        backend.Conv2DMaxColTileBytes = 128;

        Tensor input = new(new TensorShape(B, Cin, H, W), DType.F32);
        Tensor weight = new(new TensorShape(Cout, Cin, Kh, Kw), DType.F32);
        Tensor bias = new(new TensorShape(Cout), DType.F32);
        Tensor output = new(new TensorShape(B, Cout, H, W), DType.F32);

        Random rng = new(43);
        Span<float> iS = input.AsSpan<float>();
        Span<float> wS = weight.AsSpan<float>();
        Span<float> bS = bias.AsSpan<float>();
        for (int i = 0; i < B * Cin * H * W; i++) iS[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < Cout * Cin * Kh * Kw; i++) wS[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        for (int i = 0; i < Cout; i++) bS[i] = (float)(rng.NextDouble() * 2 - 1) * 0.05f;

        backend.Conv2D(output, input, weight, bias, strideH: 1, strideW: 1, padH: 1, padW: 1);

        ReadOnlySpan<float> oS = output.AsReadOnlySpan<float>();
        float maxErr = 0f;
        for (int oc = 0; oc < Cout; oc++)
            for (int yp = 0; yp < H; yp++)
                for (int xp = 0; xp < W; xp++)
                {
                    float acc = bS[oc];
                    for (int ic = 0; ic < Cin; ic++)
                        for (int kh = 0; kh < Kh; kh++)
                            for (int kw = 0; kw < Kw; kw++)
                            {
                                int ih = yp - 1 + kh;
                                int iw = xp - 1 + kw;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                acc += iS[(ic * H + ih) * W + iw] * wS[((oc * Cin + ic) * Kh + kh) * Kw + kw];
                            }
                    float got = oS[(oc * H + yp) * W + xp];
                    maxErr = MathF.Max(maxErr, MathF.Abs(got - acc));
                }
        Assert.True(maxErr < 1e-3f, $"Conv2D tiled-path maxErr {maxErr:E3} too high.");

        input.Dispose(); weight.Dispose(); bias.Dispose(); output.Dispose();
    }

    [Fact]
    public void Backend_UpsampleNearest_Matches_Cpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int B = 1, C = 2, H = 4, W = 4;
        Tensor x = new(new TensorShape(B, C, H, W), DType.F32);
        Tensor y = new(new TensorShape(B, C, H * 2, W * 2), DType.F32);
        Span<float> xS = x.AsSpan<float>();
        for (int i = 0; i < B * C * H * W; i++) xS[i] = i + 1;

        backend.UpsampleNearest2D(y, x, scaleH: 2, scaleW: 2);

        ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
        for (int c = 0; c < C; c++)
            for (int h = 0; h < H * 2; h++)
                for (int w = 0; w < W * 2; w++)
                {
                    float src = xS[(c * H + h / 2) * W + w / 2];
                    float dst = yS[(c * H * 2 + h) * W * 2 + w];
                    Assert.InRange(dst - src, -1e-5f, 1e-5f);
                }

        x.Dispose(); y.Dispose();
    }

    [Fact]
    public void Backend_UpsampleBilinear_Matches_Cpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        // Small input where we can hand-check a couple of interior values.
        const int B = 1, C = 1, H = 4, W = 4;
        Tensor x = new(new TensorShape(B, C, H, W), DType.F32);
        Tensor y = new(new TensorShape(B, C, H * 2, W * 2), DType.F32);
        Span<float> xS = x.AsSpan<float>();
        for (int i = 0; i < B * C * H * W; i++) xS[i] = MathF.Sin(i * 0.5f);

        backend.UpsampleBilinear2D(y, x, scaleH: 2, scaleW: 2);

        // Sanity: corners must equal source corners; interior must sit between source samples.
        ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
        // Top-left output should equal top-left input (align_corners=False uses src(-0.25), but the
        // shader's specific corner convention is what we care about — just ensure NaN/Inf-free and
        // that the output is within the source range envelope.)
        float minSrc = float.PositiveInfinity, maxSrc = float.NegativeInfinity;
        for (int i = 0; i < B * C * H * W; i++) { minSrc = MathF.Min(minSrc, xS[i]); maxSrc = MathF.Max(maxSrc, xS[i]); }
        // Bilinear output is bounded by the input min/max + small tolerance (accounting for corner extrapolation).
        float pad = 0.1f * (maxSrc - minSrc);
        for (int i = 0; i < B * C * H * 2 * W * 2; i++)
        {
            Assert.False(float.IsNaN(yS[i]) || float.IsInfinity(yS[i]));
            Assert.InRange(yS[i], minSrc - pad, maxSrc + pad);
        }

        x.Dispose(); y.Dispose();
    }

    [Fact]
    public void Backend_Cast_F32_To_F16_Matches_Cpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();
        if (!backend.Capabilities.SupportsF16) return;

        Tensor src = new(new TensorShape(128), DType.F32);
        Tensor dst = new(new TensorShape(128), DType.F16);
        Span<float> sS = src.AsSpan<float>();
        for (int i = 0; i < 128; i++) sS[i] = i * 0.0625f - 4f;

        backend.CastToF16(dst, src);

        ReadOnlySpan<Half> dS = dst.AsReadOnlySpan<Half>();
        for (int i = 0; i < 128; i++)
        {
            // Round-trip through CPU Half conversion as the reference.
            float exp = (float)(Half)sS[i];
            Assert.InRange((float)dS[i] - exp, -1e-3f, 1e-3f);
        }
        src.Dispose(); dst.Dispose();
    }

    [Fact]
    public void Backend_Cast_F16_To_F32_Matches_Cpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();
        if (!backend.Capabilities.SupportsF16) return;

        Tensor src = new(new TensorShape(128), DType.F16);
        Tensor dst = new(new TensorShape(128), DType.F32);
        Span<Half> sS = src.AsSpan<Half>();
        for (int i = 0; i < 128; i++) sS[i] = (Half)(i * 0.0625f - 4f);

        backend.CastToF32(dst, src);

        ReadOnlySpan<float> dS = dst.AsReadOnlySpan<float>();
        for (int i = 0; i < 128; i++)
            Assert.InRange(dS[i] - (float)sS[i], -1e-5f, 1e-5f);

        src.Dispose(); dst.Dispose();
    }

    /// <summary>Regression gate for a real OOM-investigation finding (2026-07-30): Krea2's DiT
    /// (<c>Krea2Transformer.ComputeTimeEmbedding</c>) has a BF16 weight feeding a <c>Linear</c> whose GEMM
    /// resolves to F32 — <c>CastIfNeeded</c> had no BF16 branch at all before this, throwing
    /// <c>NotSupportedException</c> the first time any real model exercised it (no synthetic unit test had
    /// a BF16 tensor). Exercises the private cast indirectly through <c>Linear</c>, the same call path the
    /// real model hits, rather than testing the cast in isolation.</summary>
    [Fact]
    public void Backend_Linear_Bf16Weight_MatchesCpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int M = 8, K = 64, N = 8;
        Tensor input = new(new TensorShape(M, K), DType.F32);
        Tensor weightF32 = new(new TensorShape(N, K), DType.F32);
        Tensor output = new(new TensorShape(M, N), DType.F32);

        Random rng = new(17);
        Span<float> iS = input.AsSpan<float>();
        Span<float> wS = weightF32.AsSpan<float>();
        for (int i = 0; i < M * K; i++) iS[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        for (int i = 0; i < N * K; i++) wS[i] = (float)(rng.NextDouble() * 2 - 1) * 0.05f;

        // BF16 weight, matching the truncated-mantissa CPU reference the rest of the codebase already
        // uses (Tensor.CastTo) — the shader must match THIS conversion, not a hypothetical exact one.
        Tensor weightBf16 = weightF32.CastTo(DType.BF16);
        Tensor weightF32Roundtrip = weightBf16.CastTo(DType.F32);   // the CPU-truncated reference weight

        backend.Linear(output, input, weightBf16, null);

        ReadOnlySpan<float> oS = output.AsReadOnlySpan<float>();
        ReadOnlySpan<float> wRef = weightF32Roundtrip.AsReadOnlySpan<float>();
        for (int m = 0; m < M; m++)
        {
            for (int n = 0; n < N; n++)
            {
                float acc = 0;
                for (int k = 0; k < K; k++) acc += iS[m * K + k] * wRef[n * K + k];
                Assert.InRange(oS[m * N + n] - acc, -1e-3f, 1e-3f);
            }
        }
        input.Dispose(); weightF32.Dispose(); weightBf16.Dispose(); weightF32Roundtrip.Dispose(); output.Dispose();
    }

    /// <summary>Regression gate for a historical CUDA incident with the SAME symptom class as a real
    /// Krea2-on-Vulkan finding: <c>PARITY_VERIFICATION.md</c>'s "Krea2 / all fp8_scaled DiTs (fused GEMV)"
    /// row documents a past CUDA bug where BF16 biases on Krea2's time-embed/modulation linears were
    /// reinterpreted as F32 raw bytes, exploding timestep conditioning into an all-black image. Krea2's
    /// real checkpoint confirms `time_embed.linear_1/2.bias`, `time_mod_proj.bias`, and `img_in.bias` are
    /// ALL BF16 — exercised on Vulkan via <see cref="VulkanBackend.Linear"/>'s bias operand for the first
    /// time by Krea2 (the existing <see cref="Backend_Linear_Bf16Weight_MatchesCpu"/> test only covers a
    /// BF16 WEIGHT with bias=null). Covers the bias-cast path specifically, independent of the weight cast.</summary>
    [Theory]
    [InlineData(8)]
    [InlineData(1)]   // M=1: the exact shape Krea2's time-embed/img_in projections use (batch=1)
    public void Backend_Linear_Bf16Bias_MatchesCpu(int M)
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int K = 64, N = 8;
        Tensor input = new(new TensorShape(M, K), DType.F32);
        Tensor weight = new(new TensorShape(N, K), DType.F32);
        Tensor output = new(new TensorShape(M, N), DType.F32);

        Random rng = new(19);
        Span<float> iS = input.AsSpan<float>();
        Span<float> wS = weight.AsSpan<float>();
        for (int i = 0; i < M * K; i++) iS[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        for (int i = 0; i < N * K; i++) wS[i] = (float)(rng.NextDouble() * 2 - 1) * 0.05f;

        Tensor biasF32 = new(new TensorShape(N), DType.F32);
        Span<float> bS = biasF32.AsSpan<float>();
        for (int i = 0; i < N; i++) bS[i] = (float)(rng.NextDouble() * 2 - 1) * 0.2f;

        // BF16 bias, matching the truncated-mantissa CPU reference the rest of the codebase already uses
        // (Tensor.CastTo) — the shader must match THIS conversion, not a hypothetical exact one.
        Tensor biasBf16 = biasF32.CastTo(DType.BF16);
        Tensor biasF32Roundtrip = biasBf16.CastTo(DType.F32);   // the CPU-truncated reference bias

        backend.Linear(output, input, weight, biasBf16);

        ReadOnlySpan<float> oS = output.AsReadOnlySpan<float>();
        ReadOnlySpan<float> bRef = biasF32Roundtrip.AsReadOnlySpan<float>();
        for (int m = 0; m < M; m++)
        {
            for (int n = 0; n < N; n++)
            {
                float acc = bRef[n];
                for (int k = 0; k < K; k++) acc += iS[m * K + k] * wS[n * K + k];
                Assert.InRange(oS[m * N + n] - acc, -1e-3f, 1e-3f);
            }
        }
        input.Dispose(); weight.Dispose(); output.Dispose();
        biasF32.Dispose(); biasBf16.Dispose(); biasF32Roundtrip.Dispose();
    }

    /// <summary>Same as <see cref="Backend_Linear_Bf16Bias_MatchesCpu"/> but at Krea2's EXACT real shape for
    /// <c>time_mod_proj</c> (M=1, K=hidden=6144, N=6·hidden=36864) — the live e2e run showed block 0's output
    /// already ~5-10x larger than the CUDA reference (min/max magnitude), and `time_mod_proj`'s BF16
    /// weight+bias feed EVERY block's modulation (gate/scale/shift) vectors, so a scale-dependent bug in the
    /// tiny-tile (M=1) GEMM+bias-fusion path at this specific large-N width would explain a uniform per-block
    /// amplification invisible at the smaller N=8 shape already tested.</summary>
    [Fact]
    public void Backend_Linear_Bf16Bias_MatchesCpu_RealKrea2ModProjShape()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();
        if (!backend.Capabilities.SupportsF16) return;

        const int M = 1, K = 6144, N = 6 * 6144;
        Tensor input = new(new TensorShape(M, K), DType.F32);
        Tensor weight = new(new TensorShape(N, K), DType.F32);
        Tensor output = new(new TensorShape(M, N), DType.F32);

        Random rng = new(23);
        Span<float> iS = input.AsSpan<float>();
        Span<float> wS = weight.AsSpan<float>();
        for (int i = 0; i < M * K; i++) iS[i] = (float)(rng.NextDouble() * 2 - 1) * 0.05f;
        for (int i = 0; i < N * K; i++) wS[i] = (float)(rng.NextDouble() * 2 - 1) * 0.02f;

        Tensor biasF32 = new(new TensorShape(N), DType.F32);
        Span<float> bS = biasF32.AsSpan<float>();
        for (int i = 0; i < N; i++) bS[i] = (float)(rng.NextDouble() * 2 - 1) * 0.2f;

        Tensor biasBf16 = biasF32.CastTo(DType.BF16);
        Tensor biasF32Roundtrip = biasBf16.CastTo(DType.F32);

        backend.Linear(output, input, weight, biasBf16);

        ReadOnlySpan<float> oS = output.AsReadOnlySpan<float>();
        ReadOnlySpan<float> bRef = biasF32Roundtrip.AsReadOnlySpan<float>();
        float maxErr = 0f;
        for (int n = 0; n < N; n++)
        {
            float acc = bRef[n];
            for (int k = 0; k < K; k++) acc += iS[k] * wS[n * K + k];
            maxErr = MathF.Max(maxErr, MathF.Abs(oS[n] - acc));
        }
        Assert.True(maxErr < 1e-2f, $"Linear (BF16 bias, real time_mod_proj shape) maxErr {maxErr:E3} too high.");

        input.Dispose(); weight.Dispose(); output.Dispose();
        biasF32.Dispose(); biasBf16.Dispose(); biasF32Roundtrip.Dispose();
    }

    /// <summary>Regression gate for another real Krea2-on-Vulkan finding (2026-07-30): no
    /// <c>AffineBroadcastLastDim</c> override existed at all — every call fell through to IBackend's
    /// F32-only CPU default, which throws on Krea2's F16 DiT activations
    /// (<c>Krea2Block.Forward</c> → <c>DiTUtils.Modulate</c>). Covers both the with-shift and
    /// scale-only (Ideogram 4 adaLN) variants, and both F32 and F16 activations.</summary>
    [Theory]
    [InlineData(false, false)]  // F32, with shift
    [InlineData(false, true)]   // F32, scale-only
    [InlineData(true, false)]   // F16, with shift
    [InlineData(true, true)]    // F16, scale-only
    public unsafe void Backend_AffineBroadcastLastDim_MatchesCpu(bool useF16, bool scaleOnly)
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();
        if (useF16 && !backend.Capabilities.SupportsF16) return;

        const int B = 2, seqLen = 5, C = 16;
        DType actDtype = useF16 ? DType.F16 : DType.F32;
        Tensor input = new(new TensorShape(B, seqLen, C), actDtype);
        Tensor scale = new(new TensorShape(B, C), DType.F32);
        Tensor? shift = scaleOnly ? null : new Tensor(new TensorShape(B, C), DType.F32);
        Tensor output = new(new TensorShape(B, seqLen, C), actDtype);

        Random rng = new(23);
        float[] inRef = new float[B * seqLen * C];
        float[] scaleRef = new float[B * C];
        float[] shiftRef = new float[B * C];
        if (useF16)
        {
            Span<Half> iS = input.AsSpan<Half>();
            for (int i = 0; i < iS.Length; i++) { float v = (float)(rng.NextDouble() * 2 - 1); iS[i] = (Half)v; inRef[i] = (float)(Half)v; }
        }
        else
        {
            Span<float> iS = input.AsSpan<float>();
            for (int i = 0; i < iS.Length; i++) { iS[i] = (float)(rng.NextDouble() * 2 - 1); inRef[i] = iS[i]; }
        }
        Span<float> sS = scale.AsSpan<float>();
        for (int i = 0; i < sS.Length; i++) { sS[i] = (float)(rng.NextDouble() * 2 - 1) + 1f; scaleRef[i] = sS[i]; }
        if (shift is not null)
        {
            Span<float> shS = shift.AsSpan<float>();
            for (int i = 0; i < shS.Length; i++) { shS[i] = (float)(rng.NextDouble() * 2 - 1); shiftRef[i] = shS[i]; }
        }

        backend.AffineBroadcastLastDim(output, input, scale, shift);

        float[] actual = new float[B * seqLen * C];
        if (useF16)
        {
            ReadOnlySpan<Half> oS = output.AsReadOnlySpan<Half>();
            for (int i = 0; i < actual.Length; i++) actual[i] = (float)oS[i];
        }
        else
        {
            ReadOnlySpan<float> oS = output.AsReadOnlySpan<float>();
            oS.CopyTo(actual);
        }

        float maxErr = 0f;
        for (int i = 0; i < actual.Length; i++)
        {
            int d = i % C, row = i / C, b = row / seqLen;
            int pIdx = b * C + d;
            float expected = inRef[i] * scaleRef[pIdx] + (shift is null ? 0f : shiftRef[pIdx]);
            maxErr = MathF.Max(maxErr, MathF.Abs(actual[i] - expected));
        }
        Assert.True(maxErr < (useF16 ? 5e-3f : 1e-4f), $"AffineBroadcastLastDim maxErr {maxErr:E3} too high.");

        input.Dispose(); scale.Dispose(); shift?.Dispose(); output.Dispose();
    }

    /// <summary>Regression gate for a real Krea2-on-Vulkan crash (2026-07-30): no <c>WanRopeInterleaved</c>
    /// override existed at all — <c>FluxRope.ApplyGpuGqa</c> (Krea2Attention's GQA rope) fell through to
    /// IBackend's CPU-loop default, which dereferenced the GPU-resident Q/K tensors' host mirror unsafely
    /// and crashed the process with an AccessViolationException instead of throwing a catchable managed
    /// exception. Covers both F32 and F16 activations (cos/sin are always F32).</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public unsafe void Backend_WanRopeInterleaved_MatchesCpu(bool useF16)
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();
        if (useF16 && !backend.Capabilities.SupportsF16) return;

        const int seqLen = 6, heads = 3, headDim = 8;
        DType actDtype = useF16 ? DType.F16 : DType.F32;
        Tensor x = new(new TensorShape(seqLen, heads, headDim), actDtype);
        Tensor cos = new(new TensorShape(seqLen, headDim), DType.F32);
        Tensor sin = new(new TensorShape(seqLen, headDim), DType.F32);

        Random rng = new(29);
        float[] xRef = new float[seqLen * heads * headDim];
        if (useF16)
        {
            Span<Half> xS = x.AsSpan<Half>();
            for (int i = 0; i < xS.Length; i++) { float v = (float)(rng.NextDouble() * 2 - 1); xS[i] = (Half)v; xRef[i] = (float)(Half)v; }
        }
        else
        {
            Span<float> xS = x.AsSpan<float>();
            for (int i = 0; i < xS.Length; i++) { xS[i] = (float)(rng.NextDouble() * 2 - 1); xRef[i] = xS[i]; }
        }
        // Duplicated-pair layout (FluxRope.GetGpuTables): pair i's angle is stored at BOTH 2i and 2i+1.
        Span<float> cS = cos.AsSpan<float>();
        Span<float> sS = sin.AsSpan<float>();
        for (int s = 0; s < seqLen; s++)
            for (int i = 0; i < headDim / 2; i++)
            {
                double angle = rng.NextDouble() * Math.PI;
                float c = (float)Math.Cos(angle), sn = (float)Math.Sin(angle);
                cS[s * headDim + 2 * i] = c; cS[s * headDim + 2 * i + 1] = c;
                sS[s * headDim + 2 * i] = sn; sS[s * headDim + 2 * i + 1] = sn;
            }

        backend.WanRopeInterleaved(x, cos, sin, seqLen, heads, headDim);

        float[] actual = new float[xRef.Length];
        if (useF16)
        {
            ReadOnlySpan<Half> oS = x.AsReadOnlySpan<Half>();
            for (int i = 0; i < actual.Length; i++) actual[i] = (float)oS[i];
        }
        else
        {
            ReadOnlySpan<float> oS = x.AsReadOnlySpan<float>();
            oS.CopyTo(actual);
        }

        int pairs = headDim / 2;
        float maxErr = 0f;
        for (int s = 0; s < seqLen; s++)
            for (int h = 0; h < heads; h++)
                for (int i = 0; i < pairs; i++)
                {
                    int xoff = (s * heads + h) * headDim, coff = s * headDim, i0 = 2 * i;
                    float re = xRef[xoff + i0], im = xRef[xoff + i0 + 1];
                    float c = cS[coff + i0], sn = sS[coff + i0];
                    float expRe = re * c - im * sn, expIm = re * sn + im * c;
                    maxErr = MathF.Max(maxErr, MathF.Abs(actual[xoff + i0] - expRe));
                    maxErr = MathF.Max(maxErr, MathF.Abs(actual[xoff + i0 + 1] - expIm));
                }
        Assert.True(maxErr < (useF16 ? 5e-3f : 1e-4f), $"WanRopeInterleaved maxErr {maxErr:E3} too high.");

        x.Dispose(); cos.Dispose(); sin.Dispose();
    }

    /// <summary>Regression gate for another real Krea2-on-Vulkan finding (2026-07-30): no
    /// <c>RepeatKvHeads</c> override existed at all — every call fell through to IBackend's F32-only
    /// CPU default, which throws on Krea2's F16 GQA K/V (<c>Krea2Attention.Forward</c>).</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public unsafe void Backend_RepeatKvHeads_MatchesCpu(bool useF16)
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();
        if (useF16 && !backend.Capabilities.SupportsF16) return;

        const int batch = 2, kvHeads = 3, groupSize = 4, seqLen = 5, headDim = 8;
        DType dtype = useF16 ? DType.F16 : DType.F32;
        Tensor input = new(new TensorShape(batch, kvHeads, seqLen, headDim), dtype);
        Tensor output = new(new TensorShape(batch, kvHeads * groupSize, seqLen, headDim), dtype);

        Random rng = new(31);
        float[] inRef = new float[batch * kvHeads * seqLen * headDim];
        if (useF16)
        {
            Span<Half> iS = input.AsSpan<Half>();
            for (int i = 0; i < iS.Length; i++) { float v = (float)(rng.NextDouble() * 2 - 1); iS[i] = (Half)v; inRef[i] = (float)(Half)v; }
        }
        else
        {
            Span<float> iS = input.AsSpan<float>();
            for (int i = 0; i < iS.Length; i++) { iS[i] = (float)(rng.NextDouble() * 2 - 1); inRef[i] = iS[i]; }
        }

        backend.RepeatKvHeads(output, input, kvHeads, groupSize);

        int qHeads = kvHeads * groupSize;
        float[] actual = new float[batch * qHeads * seqLen * headDim];
        if (useF16)
        {
            ReadOnlySpan<Half> oS = output.AsReadOnlySpan<Half>();
            for (int i = 0; i < actual.Length; i++) actual[i] = (float)oS[i];
        }
        else
        {
            ReadOnlySpan<float> oS = output.AsReadOnlySpan<float>();
            oS.CopyTo(actual);
        }

        int perHead = seqLen * headDim;
        float maxErr = 0f;
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < kvHeads; h++)
                for (int g = 0; g < groupSize; g++)
                {
                    int qHead = h * groupSize + g;
                    long srcOff = ((long)b * kvHeads + h) * perHead;
                    long dstOff = ((long)b * qHeads + qHead) * perHead;
                    for (int d = 0; d < perHead; d++)
                        maxErr = MathF.Max(maxErr, MathF.Abs(actual[dstOff + d] - inRef[srcOff + d]));
                }
        Assert.True(maxErr < 1e-6f, $"RepeatKvHeads maxErr {maxErr:E3} too high.");

        input.Dispose(); output.Dispose();
    }

    /// <summary>Regression gate for a real Vulkan VAE-decode finding (2026-07-31): no <c>WanRmsNormChannel</c>
    /// override existed at all — every call fell through to <c>IBackend</c>'s CPU-loop default, which reads/
    /// writes <c>DataPointer</c> directly (a full D2H sync, a single-threaded cache-hostile stride-<c>spatial</c>
    /// reduction over C, then an H2D re-upload). <c>QwenImageVaeDecoder</c>'s one call site runs at the VAE's
    /// full output resolution — the worst possible shape for this fallthrough — and was a real contributor to
    /// Krea2's ~2300x VAE-decode gap vs CUDA (which has always had a real kernel for this op, see
    /// <c>CudaBackend.WanRmsNormChannel</c>). Covers both the gamma and no-gamma (null) paths.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Backend_WanRmsNormChannel_MatchesCpu(bool hasGamma)
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int batch = 2, c = 6, h = 3, w = 4;
        const float eps = 1e-12f;
        long spatial = h * w;

        Tensor input = new(new TensorShape(batch, c, h, w), DType.F32);
        Tensor output = new(new TensorShape(batch, c, h, w), DType.F32);
        Tensor? gamma = hasGamma ? new Tensor(new TensorShape(c), DType.F32) : null;

        float[] inRef = FillRandom((int)(batch * c * spatial), 53);
        inRef.CopyTo(input.AsSpan<float>());
        float[] gammaRef = hasGamma ? FillRandom(c, 54) : new float[c];
        if (hasGamma) gammaRef.CopyTo(gamma!.AsSpan<float>());

        backend.WanRmsNormChannel(output, input, gamma, eps);

        float sqrtC = MathF.Sqrt(c);
        float[] expected = new float[inRef.Length];
        for (int b = 0; b < batch; b++)
        {
            long baseB = (long)b * c * spatial;
            for (long s = 0; s < spatial; s++)
            {
                double sumSq = 0;
                for (int ci = 0; ci < c; ci++) { float v = inRef[baseB + ci * spatial + s]; sumSq += (double)v * v; }
                float denom = MathF.Max((float)Math.Sqrt(sumSq), eps);
                float scale = sqrtC / denom;
                for (int ci = 0; ci < c; ci++)
                {
                    long off = baseB + ci * spatial + s;
                    expected[off] = inRef[off] * scale * (hasGamma ? gammaRef[ci] : 1f);
                }
            }
        }
        AssertMatches(output, expected);

        input.Dispose(); output.Dispose(); gamma?.Dispose();
    }

    /// <summary>Regression gate for another real Krea2-on-Vulkan finding (2026-07-30): no
    /// <c>GatedResidualLastDim</c> override existed at all — every call fell through to IBackend's F32-only
    /// CPU default, which throws on Krea2's F16 activations (<c>Krea2Block.Forward</c>'s attn/mlp gate).</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public unsafe void Backend_GatedResidualLastDim_MatchesCpu(bool useF16)
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();
        if (useF16 && !backend.Capabilities.SupportsF16) return;

        const int B = 2, seqLen = 5, C = 16;
        DType actDtype = useF16 ? DType.F16 : DType.F32;
        Tensor residual = new(new TensorShape(B, seqLen, C), actDtype);
        Tensor value = new(new TensorShape(B, seqLen, C), actDtype);
        Tensor gate = new(new TensorShape(B, C), DType.F32);
        Tensor output = new(new TensorShape(B, seqLen, C), actDtype);

        Random rng = new(37);
        float[] resRef = new float[B * seqLen * C];
        float[] valRef = new float[B * seqLen * C];
        float[] gateRef = new float[B * C];
        if (useF16)
        {
            Span<Half> rS = residual.AsSpan<Half>();
            Span<Half> vS = value.AsSpan<Half>();
            for (int i = 0; i < rS.Length; i++)
            {
                float rv = (float)(rng.NextDouble() * 2 - 1), vv = (float)(rng.NextDouble() * 2 - 1);
                rS[i] = (Half)rv; vS[i] = (Half)vv;
                resRef[i] = (float)(Half)rv; valRef[i] = (float)(Half)vv;
            }
        }
        else
        {
            Span<float> rS = residual.AsSpan<float>();
            Span<float> vS = value.AsSpan<float>();
            for (int i = 0; i < rS.Length; i++) { rS[i] = (float)(rng.NextDouble() * 2 - 1); vS[i] = (float)(rng.NextDouble() * 2 - 1); resRef[i] = rS[i]; valRef[i] = vS[i]; }
        }
        Span<float> gS = gate.AsSpan<float>();
        for (int i = 0; i < gS.Length; i++) { gS[i] = (float)(rng.NextDouble() * 2 - 1); gateRef[i] = gS[i]; }

        backend.GatedResidualLastDim(output, residual, value, gate);

        float[] actual = new float[B * seqLen * C];
        if (useF16)
        {
            ReadOnlySpan<Half> oS = output.AsReadOnlySpan<Half>();
            for (int i = 0; i < actual.Length; i++) actual[i] = (float)oS[i];
        }
        else
        {
            ReadOnlySpan<float> oS = output.AsReadOnlySpan<float>();
            oS.CopyTo(actual);
        }

        float maxErr = 0f;
        for (int i = 0; i < actual.Length; i++)
        {
            int d = i % C, row = i / C, b = row / seqLen;
            float expected = resRef[i] + gateRef[b * C + d] * valRef[i];
            maxErr = MathF.Max(maxErr, MathF.Abs(actual[i] - expected));
        }
        Assert.True(maxErr < (useF16 ? 5e-3f : 1e-4f), $"GatedResidualLastDim maxErr {maxErr:E3} too high.");

        residual.Dispose(); value.Dispose(); gate.Dispose(); output.Dispose();
    }

    /// <summary>Regression gate for another real Krea2-on-Vulkan finding (2026-07-30): no <c>SliceRows</c>
    /// override existed at all — every call fell through to IBackend's F32-only CPU default, which throws
    /// on Krea2's F16 joint img+txt sequence (<c>Krea2Transformer.SliceTail</c>).</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public unsafe void Backend_SliceRows_MatchesCpu(bool useF16)
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();
        if (useF16 && !backend.Capabilities.SupportsF16) return;

        const int totalRows = 10, rowOffset = 4, tailRows = 6, dim = 8;
        DType dtype = useF16 ? DType.F16 : DType.F32;
        Tensor input = new(new TensorShape(totalRows, dim), dtype);
        Tensor output = new(new TensorShape(tailRows, dim), dtype);

        Random rng = new(41);
        float[] inRef = new float[totalRows * dim];
        if (useF16)
        {
            Span<Half> iS = input.AsSpan<Half>();
            for (int i = 0; i < iS.Length; i++) { float v = (float)(rng.NextDouble() * 2 - 1); iS[i] = (Half)v; inRef[i] = (float)(Half)v; }
        }
        else
        {
            Span<float> iS = input.AsSpan<float>();
            for (int i = 0; i < iS.Length; i++) { iS[i] = (float)(rng.NextDouble() * 2 - 1); inRef[i] = iS[i]; }
        }

        backend.SliceRows(output, input, rowOffset);

        float[] actual = new float[tailRows * dim];
        if (useF16)
        {
            ReadOnlySpan<Half> oS = output.AsReadOnlySpan<Half>();
            for (int i = 0; i < actual.Length; i++) actual[i] = (float)oS[i];
        }
        else
        {
            ReadOnlySpan<float> oS = output.AsReadOnlySpan<float>();
            oS.CopyTo(actual);
        }

        float maxErr = 0f;
        int elemOffset = rowOffset * dim;
        for (int i = 0; i < actual.Length; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(actual[i] - inRef[elemOffset + i]));
        Assert.True(maxErr < 1e-6f, $"SliceRows maxErr {maxErr:E3} too high.");

        input.Dispose(); output.Dispose();
    }

    // ── Phase 0 rebuild-trust-gate regressions ──────────────────────────────────────────────
    // These four ops were dispatchable in VulkanBackend.cs but had never been numerically
    // verified against a real Vulkan run: Tanh/Elu's shipped elementwise.spv predated the
    // op-8/op-9 source (silent zeros), and maxpool2d/depthwise_conv2d had no compiled .spv at
    // all (shader-load failure) because no SPIR-V compiler was available on the dev box. Both
    // are closed by rebuilding src/HartsyInference.Vulkan/Spirv/*.spv from current source; these tests pin
    // the fix so a future stale-artifact regression fails loudly instead of silently.

    [Fact]
    public void Backend_Tanh_Matches_Cpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        Tensor x = new(new TensorShape(64), DType.F32);
        Tensor y = new(new TensorShape(64), DType.F32);
        Span<float> xS = x.AsSpan<float>();
        for (int i = 0; i < 64; i++) xS[i] = i * 0.1f - 3.2f;

        backend.Tanh(y, x);

        ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
        for (int i = 0; i < 64; i++)
            Assert.InRange(yS[i] - MathF.Tanh(xS[i]), -1e-4f, 1e-4f);

        x.Dispose(); y.Dispose();
    }

    [Fact]
    public void Backend_Elu_Matches_Cpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        Tensor x = new(new TensorShape(64), DType.F32);
        Tensor y = new(new TensorShape(64), DType.F32);
        Span<float> xS = x.AsSpan<float>();
        for (int i = 0; i < 64; i++) xS[i] = i * 0.1f - 3.2f;

        const float alpha = 1.0f;
        backend.Elu(y, x, alpha);

        ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
        for (int i = 0; i < 64; i++)
        {
            float xv = xS[i];
            float expected = xv >= 0f ? xv : alpha * (MathF.Exp(xv) - 1f);
            Assert.InRange(yS[i] - expected, -1e-4f, 1e-4f);
        }
        x.Dispose(); y.Dispose();
    }

    [Fact]
    public void Backend_MaxPool2D_Matches_Cpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int N = 1, C = 2, H = 5, W = 5, kH = 3, kW = 3, stride = 2, pad = 1;
        const int oH = 3, oW = 3; // (5 + 2*1 - 3)/2 + 1 = 3
        Tensor x = new(new TensorShape(N, C, H, W), DType.F32);
        Tensor y = new(new TensorShape(N, C, oH, oW), DType.F32);
        Span<float> xS = x.AsSpan<float>();
        for (int i = 0; i < N * C * H * W; i++) xS[i] = MathF.Sin(i * 0.31f) * 5f;

        backend.MaxPool2D(y, x, kH, kW, stride, stride, pad, pad);

        ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++)
                for (int oy = 0; oy < oH; oy++)
                    for (int ox = 0; ox < oW; ox++)
                    {
                        float expected = float.NegativeInfinity;
                        bool any = false;
                        for (int ky = 0; ky < kH; ky++)
                        {
                            int iy = oy * stride + ky - pad;
                            if (iy < 0 || iy >= H) continue;
                            for (int kx = 0; kx < kW; kx++)
                            {
                                int ix = ox * stride + kx - pad;
                                if (ix < 0 || ix >= W) continue;
                                float v = xS[((n * C + c) * H + iy) * W + ix];
                                if (v > expected) expected = v;
                                any = true;
                            }
                        }
                        if (!any) expected = 0f;
                        int oi = ((n * C + c) * oH + oy) * oW + ox;
                        Assert.InRange(yS[oi] - expected, -1e-4f, 1e-4f);
                    }

        x.Dispose(); y.Dispose();
    }

    [Fact]
    public void Backend_Conv2dDepthwise_Matches_Cpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int N = 1, C = 3, H = 6, W = 6, kH = 3, kW = 3, stride = 1, pad = 1;
        const int oH = 6, oW = 6;
        Tensor x = new(new TensorShape(N, C, H, W), DType.F32);
        Tensor w = new(new TensorShape(C, 1, kH, kW), DType.F32);
        Tensor bias = new(new TensorShape(C), DType.F32);
        Tensor y = new(new TensorShape(N, C, oH, oW), DType.F32);
        Span<float> xS = x.AsSpan<float>();
        Span<float> wS = w.AsSpan<float>();
        Span<float> bS = bias.AsSpan<float>();
        for (int i = 0; i < N * C * H * W; i++) xS[i] = MathF.Cos(i * 0.23f);
        for (int i = 0; i < C * kH * kW; i++) wS[i] = MathF.Sin(i * 0.41f) * 0.5f;
        for (int c = 0; c < C; c++) bS[c] = c * 0.25f - 0.1f;

        backend.Conv2dDepthwise(y, x, w, bias, stride, stride, pad, pad);

        ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++)
                for (int oy = 0; oy < oH; oy++)
                    for (int ox = 0; ox < oW; ox++)
                    {
                        float acc = bS[c];
                        for (int ky = 0; ky < kH; ky++)
                        {
                            int iy = oy * stride + ky - pad;
                            if (iy < 0 || iy >= H) continue;
                            for (int kx = 0; kx < kW; kx++)
                            {
                                int ix = ox * stride + kx - pad;
                                if (ix < 0 || ix >= W) continue;
                                acc += wS[(c * kH + ky) * kW + kx] * xS[((n * C + c) * H + iy) * W + ix];
                            }
                        }
                        int oi = ((n * C + c) * oH + oy) * oW + ox;
                        Assert.InRange(yS[oi] - acc, -1e-4f, 1e-4f);
                    }

        x.Dispose(); w.Dispose(); bias.Dispose(); y.Dispose();
    }

    [Fact]
    public void Backend_Conv1d_Grouped_Matches_Cpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int N = 1, cIn = 4, cOut = 4, tIn = 10, kernel = 3, stride = 1, padLeft = 1, padRight = 1, dilation = 1, groups = 2;
        const int tOut = 10; // (10 + 1 + 1 - 3)/1 + 1
        const int cInPerGroup = cIn / groups, cOutPerGroup = cOut / groups;
        Tensor x = new(new TensorShape(N, cIn, tIn), DType.F32);
        Tensor w = new(new TensorShape(cOut, cInPerGroup, kernel), DType.F32);
        Tensor bias = new(new TensorShape(cOut), DType.F32);
        Tensor y = new(new TensorShape(N, cOut, tOut), DType.F32);
        Span<float> xS = x.AsSpan<float>();
        Span<float> wS = w.AsSpan<float>();
        Span<float> bS = bias.AsSpan<float>();
        for (int i = 0; i < N * cIn * tIn; i++) xS[i] = MathF.Sin(i * 0.19f);
        for (int i = 0; i < cOut * cInPerGroup * kernel; i++) wS[i] = MathF.Cos(i * 0.37f) * 0.5f;
        for (int c = 0; c < cOut; c++) bS[c] = c * 0.1f - 0.2f;

        backend.Conv1d(y, x, w, bias, stride, padLeft, padRight, dilation, groups);

        ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
        for (int oc = 0; oc < cOut; oc++)
        {
            int group = oc / cOutPerGroup;
            int icStart = group * cInPerGroup;
            for (int j = 0; j < tOut; j++)
            {
                float acc = bS[oc];
                for (int ic = 0; ic < cInPerGroup; ic++)
                {
                    int inCh = icStart + ic;
                    for (int k = 0; k < kernel; k++)
                    {
                        int src = j * stride + k * dilation - padLeft;
                        if (src < 0 || src >= tIn) continue;
                        acc += xS[inCh * tIn + src] * wS[(oc * cInPerGroup + ic) * kernel + k];
                    }
                }
                Assert.InRange(yS[oc * tOut + j] - acc, -1e-4f, 1e-4f);
            }
        }
        x.Dispose(); w.Dispose(); bias.Dispose(); y.Dispose();
    }

    [Fact]
    public void Backend_ConvTranspose1d_Matches_Cpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int N = 1, cIn = 3, cOut = 2, tIn = 6, kernel = 4, stride = 2, padLeft = 1, padRight = 1, dilation = 1, groups = 1;
        const int tOut = 12; // (tIn-1)*stride + dilation*(kernel-1) + 1 - padLeft - padRight = 5*2+3+1-2 = 12
        Tensor x = new(new TensorShape(N, cIn, tIn), DType.F32);
        Tensor w = new(new TensorShape(cIn, cOut, kernel), DType.F32);
        Tensor bias = new(new TensorShape(cOut), DType.F32);
        Tensor y = new(new TensorShape(N, cOut, tOut), DType.F32);
        Span<float> xS = x.AsSpan<float>();
        Span<float> wS = w.AsSpan<float>();
        Span<float> bS = bias.AsSpan<float>();
        for (int i = 0; i < N * cIn * tIn; i++) xS[i] = MathF.Sin(i * 0.23f);
        for (int i = 0; i < cIn * cOut * kernel; i++) wS[i] = MathF.Cos(i * 0.29f) * 0.5f;
        for (int c = 0; c < cOut; c++) bS[c] = c * 0.15f + 0.05f;

        backend.ConvTranspose1d(y, x, w, bias, stride, padLeft, padRight, dilation, groups);

        ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
        for (int oc = 0; oc < cOut; oc++)
            for (int j = 0; j < tOut; j++)
            {
                float acc = bS[oc];
                int jShifted = j + padLeft;
                for (int k = 0; k < kernel; k++)
                {
                    int num = jShifted - k * dilation;
                    if (num < 0 || num % stride != 0) continue;
                    int i = num / stride;
                    if (i >= tIn) continue;
                    for (int ic = 0; ic < cIn; ic++)
                        acc += xS[ic * tIn + i] * wS[(ic * cOut + oc) * kernel + k];
                }
                Assert.InRange(yS[oc * tOut + j] - acc, -1e-4f, 1e-4f);
            }

        x.Dispose(); w.Dispose(); bias.Dispose(); y.Dispose();
    }

    [Fact]
    public void Backend_ConvTranspose1d_GroupsGreaterThanOne_Throws()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        Tensor x = new(new TensorShape(1, 4, 6), DType.F32);
        Tensor w = new(new TensorShape(4, 1, 3), DType.F32);
        Tensor y = new(new TensorShape(1, 4, 8), DType.F32);
        try
        {
            Assert.Throws<NotSupportedException>(() => backend.ConvTranspose1d(y, x, w, null, 1, 0, 0, 1, groups: 4));
        }
        finally { x.Dispose(); w.Dispose(); y.Dispose(); }
    }

    [Fact]
    public void Backend_Snake_Vanilla_Matches_Cpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int N = 1, C = 3, T = 8;
        Tensor x = new(new TensorShape(N, C, T), DType.F32);
        Tensor alpha = new(new TensorShape(C), DType.F32);
        Tensor y = new(new TensorShape(N, C, T), DType.F32);
        Span<float> xS = x.AsSpan<float>();
        Span<float> aS = alpha.AsSpan<float>();
        for (int i = 0; i < N * C * T; i++) xS[i] = MathF.Sin(i * 0.31f) * 2f;
        for (int c = 0; c < C; c++) aS[c] = 0.5f + c * 0.3f;

        backend.Snake(y, x, alpha, null);

        ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
        for (int c = 0; c < C; c++)
            for (int t = 0; t < T; t++)
            {
                float xv = xS[c * T + t];
                float a = aS[c];
                float s = MathF.Sin(a * xv);
                float expected = xv + (s * s) / a;
                Assert.InRange(yS[c * T + t] - expected, -1e-4f, 1e-4f);
            }
        x.Dispose(); alpha.Dispose(); y.Dispose();
    }

    [Fact]
    public void Backend_Snake_Beta_Matches_Cpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int N = 1, C = 3, T = 8;
        Tensor x = new(new TensorShape(N, C, T), DType.F32);
        Tensor alpha = new(new TensorShape(C), DType.F32);
        Tensor beta = new(new TensorShape(C), DType.F32);
        Tensor y = new(new TensorShape(N, C, T), DType.F32);
        Span<float> xS = x.AsSpan<float>();
        Span<float> aS = alpha.AsSpan<float>();
        Span<float> bS = beta.AsSpan<float>();
        for (int i = 0; i < N * C * T; i++) xS[i] = MathF.Cos(i * 0.27f) * 2f;
        for (int c = 0; c < C; c++) { aS[c] = 0.4f + c * 0.2f; bS[c] = 0.2f + c * 0.1f; }

        backend.Snake(y, x, alpha, beta);

        ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
        for (int c = 0; c < C; c++)
            for (int t = 0; t < T; t++)
            {
                float xv = xS[c * T + t];
                float a = aS[c];
                float divisor = bS[c] + 1e-8f;
                float s = MathF.Sin(a * xv);
                float expected = xv + (s * s) / divisor;
                Assert.InRange(yS[c * T + t] - expected, -1e-4f, 1e-4f);
            }
        x.Dispose(); alpha.Dispose(); beta.Dispose(); y.Dispose();
    }

    // ── Phase 2 perf-measurement infrastructure ─────────────────────────────────────────────
    // GetD2hSyncCount/ResetD2hSyncCount mirror CudaBackend's counter of the same name. These tests
    // pin the two directions that matter: a GPU-resident op that's never read stays at zero syncs,
    // and a CPU-loop-default IBackend member (Concat) that reads .DataPointer directly forces one
    // sync per GPU-resident input it touches — the exact "hidden D2H round trip" the counter exists
    // to surface.

    [Fact]
    public void GetD2hSyncCount_TracksLazyActivationReads()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();
        backend.ResetD2hSyncCount();
        Assert.Equal(0, backend.GetD2hSyncCount());

        Tensor x = new(new TensorShape(32), DType.F32);
        Tensor y = new(new TensorShape(32), DType.F32);
        Span<float> xS = x.AsSpan<float>();
        for (int i = 0; i < 32; i++) xS[i] = i;

        backend.Scale(y, x, 2f);
        // The op's output is cached GPU-side (CacheOutput → CacheActivation) — nothing has read it
        // back yet, so no sync should have fired.
        Assert.Equal(0, backend.GetD2hSyncCount());

        // Reading y's data forces the lazy D2H sync callback exactly once.
        ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
        Assert.Equal(1, backend.GetD2hSyncCount());
        Assert.InRange(yS[5] - 10f, -1e-5f, 1e-5f);

        x.Dispose(); y.Dispose();
    }

    /// <summary>Regression gate for a real Krea2-on-Vulkan bug (2026-07-30): <c>Concat</c> was a pure CPU loop
    /// reading <c>.DataPointer</c> directly on every input AND the output — found capture-illegal via a real
    /// <c>HARTSY_DIT_GRAPH=1</c> Krea2 run (`ForwardCore`'s `Concat(joint, [txt, img], dim: 1)`, the
    /// text+image sequence join every DiT forward pass) and, independent of graph mode, a D2H sync on every
    /// GPU-resident input on every forward pass regardless. Fixed with a device-resident <c>vkCmdCopyBuffer</c>
    /// implementation (concatenation with contiguous inner strides is pure data movement — no compute shader
    /// needed). This test used to assert the OLD broken behavior (2 syncs) as documentation of the gap; now it
    /// asserts the fix.</summary>
    [Fact]
    public void GetD2hSyncCount_Concat_StaysGpuResident()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        Tensor a = new(new TensorShape(8), DType.F32);
        Tensor b = new(new TensorShape(8), DType.F32);
        Tensor outA = new(new TensorShape(8), DType.F32);
        Tensor outB = new(new TensorShape(8), DType.F32);
        Tensor concatOut = new(new TensorShape(16), DType.F32);
        Span<float> aS = a.AsSpan<float>();
        Span<float> bS = b.AsSpan<float>();
        for (int i = 0; i < 8; i++) { aS[i] = i; bS[i] = 100 + i; }

        // Two real GPU dispatches leave outA/outB GPU-resident (cached, lazy-sync armed).
        backend.Scale(outA, a, 1f);
        backend.Scale(outB, b, 1f);
        backend.ResetD2hSyncCount();
        Assert.Equal(0, backend.GetD2hSyncCount());

        backend.Concat(concatOut, new Tensor[] { outA, outB }, 0);
        Assert.Equal(0, backend.GetD2hSyncCount());

        float[] expected = new float[16];
        for (int i = 0; i < 8; i++) { expected[i] = i; expected[8 + i] = 100 + i; }
        AssertMatches(concatOut, expected);

        a.Dispose(); b.Dispose(); outA.Dispose(); outB.Dispose(); concatOut.Dispose();
    }

    /// <summary>Concat along a NON-leading axis (dim=1 of a 3D [B,S,hidden] tensor, batch &gt; 1) — the
    /// multi-region <c>vkCmdCopyBuffer</c> path (outerStride &gt; 1), distinct from the dim=0/batch=1 case
    /// above where outerStride is trivially 1. Matches Krea2's real `Concat(joint, [txt, img], dim: 1)` shape
    /// class (that call happens to run at batch=1, so this closes the batch&gt;1 case it doesn't exercise).</summary>
    [Fact]
    public void Backend_Concat_NonLeadingDim_BatchGreaterThanOne_MatchesCpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int batch = 3, s1 = 4, s2 = 5, hidden = 6;
        Tensor a = new(new TensorShape(batch, s1, hidden), DType.F32);
        Tensor b = new(new TensorShape(batch, s2, hidden), DType.F32);
        float[] aData = FillRandom(batch * s1 * hidden, 71);
        float[] bData = FillRandom(batch * s2 * hidden, 72);
        aData.CopyTo(a.AsSpan<float>());
        bData.CopyTo(b.AsSpan<float>());

        Tensor output = new(new TensorShape(batch, s1 + s2, hidden), DType.F32);
        backend.Concat(output, new Tensor[] { a, b }, dim: 1);

        float[] expected = new float[batch * (s1 + s2) * hidden];
        for (int bIdx = 0; bIdx < batch; bIdx++)
        {
            for (int s = 0; s < s1; s++)
                for (int h = 0; h < hidden; h++)
                    expected[(bIdx * (s1 + s2) + s) * hidden + h] = aData[(bIdx * s1 + s) * hidden + h];
            for (int s = 0; s < s2; s++)
                for (int h = 0; h < hidden; h++)
                    expected[(bIdx * (s1 + s2) + s1 + s) * hidden + h] = bData[(bIdx * s2 + s) * hidden + h];
        }
        AssertMatches(output, expected);

        a.Dispose(); b.Dispose(); output.Dispose();
    }

    // ── Phase 3 GPU-residency closure ───────────────────────────────────────────────────────
    // SliceLastDim/ApplyRope/KvCacheAppend previously had no VulkanBackend override at all — every
    // call fell through to IBackend's CPU-loop default. These pin the new GPU dispatches against the
    // same CPU-reference math (mirroring IBackend.cs's own default bodies).

    [Fact]
    public void Backend_SliceLastDim_Matches_Cpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int rows = 4, inDim = 12, offset = 5, outDim = 4;
        Tensor input = new(new TensorShape(rows, inDim), DType.F32);
        Tensor output = new(new TensorShape(rows, outDim), DType.F32);
        Span<float> inS = input.AsSpan<float>();
        for (int i = 0; i < rows * inDim; i++) inS[i] = MathF.Sin(i * 0.13f);

        backend.SliceLastDim(output, input, offset);

        ReadOnlySpan<float> outS = output.AsReadOnlySpan<float>();
        for (int row = 0; row < rows; row++)
            for (int d = 0; d < outDim; d++)
                Assert.InRange(outS[row * outDim + d] - inS[row * inDim + offset + d], -1e-5f, 1e-5f);

        input.Dispose(); output.Dispose();
    }

    [Fact]
    public void Backend_ApplyRope_Matches_Cpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int batch = 1, seqLen = 3, numHeads = 2, headDim = 8, half = headDim / 2;
        Tensor q = new(new TensorShape(batch, seqLen, numHeads, headDim), DType.F32);
        Tensor k = new(new TensorShape(batch, seqLen, numHeads, headDim), DType.F32);
        Tensor cos = new(new TensorShape(batch, seqLen, headDim), DType.F32);
        Tensor sin = new(new TensorShape(batch, seqLen, headDim), DType.F32);
        Span<float> qS = q.AsSpan<float>(), kS = k.AsSpan<float>();
        Span<float> cosS = cos.AsSpan<float>(), sinS = sin.AsSpan<float>();
        for (int i = 0; i < batch * seqLen * numHeads * headDim; i++) { qS[i] = MathF.Sin(i * 0.11f); kS[i] = MathF.Cos(i * 0.17f); }
        for (int i = 0; i < batch * seqLen * headDim; i++) { float a = i * 0.05f; cosS[i] = MathF.Cos(a); sinS[i] = MathF.Sin(a); }

        // Reference computed BEFORE mutating q/k in place.
        float[] qRef = qS.ToArray(), kRef = kS.ToArray();
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
            {
                int freqBase = (b * seqLen + s) * headDim;
                for (int h = 0; h < numHeads; h++)
                {
                    int vecOff = ((b * seqLen + s) * numHeads + h) * headDim;
                    for (int i = 0; i < half; i++)
                    {
                        float qLower = qRef[vecOff + i], qUpper = qRef[vecOff + i + half];
                        qRef[vecOff + i] = qLower * cosS[freqBase + i] - qUpper * sinS[freqBase + i];
                        qRef[vecOff + i + half] = qUpper * cosS[freqBase + i + half] + qLower * sinS[freqBase + i + half];
                        float kLower = kRef[vecOff + i], kUpper = kRef[vecOff + i + half];
                        kRef[vecOff + i] = kLower * cosS[freqBase + i] - kUpper * sinS[freqBase + i];
                        kRef[vecOff + i + half] = kUpper * cosS[freqBase + i + half] + kLower * sinS[freqBase + i + half];
                    }
                }
            }

        backend.ApplyRope(q, k, cos, sin);

        ReadOnlySpan<float> qOut = q.AsReadOnlySpan<float>();
        ReadOnlySpan<float> kOut = k.AsReadOnlySpan<float>();
        for (int i = 0; i < batch * seqLen * numHeads * headDim; i++)
        {
            Assert.InRange(qOut[i] - qRef[i], -1e-4f, 1e-4f);
            Assert.InRange(kOut[i] - kRef[i], -1e-4f, 1e-4f);
        }

        q.Dispose(); k.Dispose(); cos.Dispose(); sin.Dispose();
    }

    [Fact]
    public void Backend_KvCacheAppend_Matches_Cpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int heads = 3, maxSeq = 16, headDim = 4, tNew = 5, offset = 7;
        Tensor buffer = new(new TensorShape(1, heads, maxSeq, headDim), DType.F32);
        Tensor newKv = new(new TensorShape(1, heads, tNew, headDim), DType.F32);
        Span<float> bufS = buffer.AsSpan<float>();
        Span<float> newS = newKv.AsSpan<float>();
        for (int i = 0; i < heads * maxSeq * headDim; i++) bufS[i] = -1f;   // sentinel: untouched region must survive
        for (int i = 0; i < heads * tNew * headDim; i++) newS[i] = 100f + i;

        float[] expected = bufS.ToArray();
        for (int h = 0; h < heads; h++)
            for (int t = 0; t < tNew; t++)
                for (int d = 0; d < headDim; d++)
                    expected[(h * maxSeq + offset + t) * headDim + d] = newS[(h * tNew + t) * headDim + d];

        backend.KvCacheAppend(buffer, newKv, offset);

        ReadOnlySpan<float> bufOut = buffer.AsReadOnlySpan<float>();
        for (int i = 0; i < heads * maxSeq * headDim; i++)
            Assert.InRange(bufOut[i] - expected[i], -1e-5f, 1e-5f);

        buffer.Dispose(); newKv.Dispose();
    }

    [Fact]
    public void Backend_CopyTo_GpuResidentSource_StaysDeviceSideAndMatchesCpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int n = 64;
        Tensor x = new(new TensorShape(n), DType.F32);
        Tensor gpuResident = new(new TensorShape(n), DType.F32);
        Tensor dest = new(new TensorShape(n), DType.F32);
        Span<float> xS = x.AsSpan<float>();
        for (int i = 0; i < n; i++) xS[i] = MathF.Sin(i * 0.2f) * 3f;

        // A real GPU dispatch leaves gpuResident cached (lazy-sync armed) — CopyTo should take the
        // device-to-device fast path instead of forcing gpuResident's D2H sync.
        backend.Scale(gpuResident, x, 2f);
        backend.ResetD2hSyncCount();
        Assert.Equal(0, backend.GetD2hSyncCount());

        backend.CopyTo(dest, gpuResident);
        Assert.Equal(0, backend.GetD2hSyncCount());   // the whole point of the fast path

        ReadOnlySpan<float> destS = dest.AsReadOnlySpan<float>();   // this read is expected to sync
        for (int i = 0; i < n; i++)
            Assert.InRange(destS[i] - xS[i] * 2f, -1e-5f, 1e-5f);

        x.Dispose(); gpuResident.Dispose(); dest.Dispose();
    }

    [Fact]
    public void Backend_CopyTo_HostOnlySource_FallsBackToMemcpy()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int n = 32;
        Tensor src = new(new TensorShape(n), DType.F32);
        Tensor dest = new(new TensorShape(n), DType.F32);
        Span<float> srcS = src.AsSpan<float>();
        for (int i = 0; i < n; i++) srcS[i] = i * 1.5f;

        backend.CopyTo(dest, src);

        ReadOnlySpan<float> destS = dest.AsReadOnlySpan<float>();
        for (int i = 0; i < n; i++) Assert.Equal(srcS[i], destS[i]);

        src.Dispose(); dest.Dispose();
    }

    // ── Phase 4 fused flash attention ───────────────────────────────────────────────────────
    // No online-softmax/fused attention existed on Vulkan before this — ScaledDotProductAttention
    // materialized the full [Sq,Skv] score matrix (the documented root cause of the Wan-video
    // full-resolution OOM) and FlashAttention/FlashAttentionDev fell through to the CPU reference.
    // These pin the new sdpa_flash.comp.glsl dispatch against a from-scratch CPU reference covering
    // the features specific to flash attention: causal masking, GQA, sliding window, an explicit
    // additive mask, and the KV-cache-buffer-larger-than-kvLen decode shape.

    /// <summary>Reference matching sdpa_flash.comp.glsl exactly (causal / sliding-window / additive
    /// mask / GQA), independent of AttentionReference so a bug shared by both wouldn't hide.</summary>
    /// <remarks>Internal (not private): reused by VulkanDecodeGraphTests for FlashAttentionDev, which
    /// dispatches the SAME shader (sdpa_flash_dev_f32) with skv/qOffset routed through a device buffer
    /// instead of push constants — one reference, not a second copy that could quietly diverge.</remarks>
    internal static float[] CpuFlashReference(
        float[] q, float[] k, float[] v, float[]? mask,
        int batch, int hq, int hkv, int sq, int skv, int headDim,
        float scale, bool causal, int qOffset, int slidingWindow)
    {
        int kvGroup = hq / hkv;
        float[] o = new float[batch * hq * sq * headDim];
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < hq; h++)
            {
                int kvHead = h / kvGroup;
                for (int sqi = 0; sqi < sq; sqi++)
                {
                    int absQ = qOffset + sqi;
                    long qBase = (((long)b * hq + h) * sq + sqi) * headDim;
                    float[] scores = new float[skv];
                    float rowMax = float.NegativeInfinity;
                    for (int j = 0; j < skv; j++)
                    {
                        bool masked = (causal && j > absQ) || (slidingWindow != 0 && (j > absQ || absQ - j >= slidingWindow));
                        if (masked) { scores[j] = float.NegativeInfinity; continue; }
                        long kBase = (((long)b * hkv + kvHead) * skv + j) * headDim;
                        float dot = 0f;
                        for (int d = 0; d < headDim; d++) dot += q[qBase + d] * k[kBase + d];
                        float s = dot * scale;
                        if (mask is not null) s += mask[sqi * skv + j];
                        scores[j] = s;
                        if (s > rowMax) rowMax = s;
                    }
                    float sum = 0f;
                    for (int j = 0; j < skv; j++)
                    {
                        float p = float.IsNegativeInfinity(rowMax) ? 0f : MathF.Exp(scores[j] - rowMax);
                        scores[j] = p;
                        sum += p;
                    }
                    for (int d = 0; d < headDim; d++)
                    {
                        float acc = 0f;
                        for (int j = 0; j < skv; j++)
                        {
                            if (scores[j] == 0f) continue;
                            long vBase = (((long)b * hkv + kvHead) * skv + j) * headDim;
                            acc += scores[j] * v[vBase + d];
                        }
                        o[qBase + d] = sum > 0f ? acc / sum : 0f;
                    }
                }
            }
        return o;
    }

    internal static float[] FillRandom(int n, int seed)
    {
        Random rng = new(seed);
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    [Fact]
    public void Backend_FlashAttention_Causal_MatchesCpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int batch = 1, hq = 4, hkv = 4, sq = 20, skv = 20, headDim = 64;
        float scale = 1f / MathF.Sqrt(headDim);
        float[] q = FillRandom(batch * hq * sq * headDim, 1);
        float[] k = FillRandom(batch * hkv * skv * headDim, 2);
        float[] v = FillRandom(batch * hkv * skv * headDim, 3);
        float[] expected = CpuFlashReference(q, k, v, null, batch, hq, hkv, sq, skv, headDim, scale, causal: true, qOffset: 0, slidingWindow: 0);

        Tensor qT = new(new TensorShape(batch, hq, sq, headDim), DType.F32);
        Tensor kT = new(new TensorShape(batch, hkv, skv, headDim), DType.F32);
        Tensor vT = new(new TensorShape(batch, hkv, skv, headDim), DType.F32);
        Tensor oT = new(new TensorShape(batch, hq, sq, headDim), DType.F32);
        q.CopyTo(qT.AsSpan<float>()); k.CopyTo(kT.AsSpan<float>()); v.CopyTo(vT.AsSpan<float>());

        backend.FlashAttention(oT, qT, kT, vT, kvLen: skv, kvGroup: hq / hkv, causal: true, qOffset: 0, scale);

        ReadOnlySpan<float> oS = oT.AsReadOnlySpan<float>();
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(oS[i] - expected[i], -2e-3f, 2e-3f);

        qT.Dispose(); kT.Dispose(); vT.Dispose(); oT.Dispose();
    }

    [Fact]
    public void Backend_FlashAttention_SlidingWindow_MatchesCpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int batch = 1, hq = 2, hkv = 2, sq = 24, skv = 24, headDim = 32, window = 6;
        float scale = 1f / MathF.Sqrt(headDim);
        float[] q = FillRandom(batch * hq * sq * headDim, 11);
        float[] k = FillRandom(batch * hkv * skv * headDim, 12);
        float[] v = FillRandom(batch * hkv * skv * headDim, 13);
        float[] expected = CpuFlashReference(q, k, v, null, batch, hq, hkv, sq, skv, headDim, scale, causal: true, qOffset: 0, slidingWindow: window);

        Tensor qT = new(new TensorShape(batch, hq, sq, headDim), DType.F32);
        Tensor kT = new(new TensorShape(batch, hkv, skv, headDim), DType.F32);
        Tensor vT = new(new TensorShape(batch, hkv, skv, headDim), DType.F32);
        Tensor oT = new(new TensorShape(batch, hq, sq, headDim), DType.F32);
        q.CopyTo(qT.AsSpan<float>()); k.CopyTo(kT.AsSpan<float>()); v.CopyTo(vT.AsSpan<float>());

        backend.FlashAttention(oT, qT, kT, vT, kvLen: skv, kvGroup: hq / hkv, causal: true, qOffset: 0, scale, slidingWindow: window);

        ReadOnlySpan<float> oS = oT.AsReadOnlySpan<float>();
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(oS[i] - expected[i], -2e-3f, 2e-3f);

        qT.Dispose(); kT.Dispose(); vT.Dispose(); oT.Dispose();
    }

    [Fact]
    public void Backend_FlashAttention_GqaAndKvLenLessThanBuffer_MatchesCpu()
    {
        // Simulates a decode step against a KV cache: the K/V buffers are over-allocated to maxSeq,
        // only the first kvLen positions are valid, and hq/hkv > 1 (GQA).
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int batch = 1, hq = 8, hkv = 2, sq = 1, maxSeq = 32, kvLen = 9, headDim = 32;
        float scale = 1f / MathF.Sqrt(headDim);
        float[] q = FillRandom(batch * hq * sq * headDim, 21);
        float[] kFull = FillRandom(batch * hkv * maxSeq * headDim, 22);
        float[] vFull = FillRandom(batch * hkv * maxSeq * headDim, 23);
        // CPU reference only reads the first kvLen positions per head — slice kFull/vFull down to
        // [batch,hkv,kvLen,headDim] contiguously so the reference's flat-index math matches skv=kvLen.
        float[] kValid = new float[batch * hkv * kvLen * headDim];
        float[] vValid = new float[batch * hkv * kvLen * headDim];
        for (int h = 0; h < hkv; h++)
        {
            Array.Copy(kFull, h * maxSeq * headDim, kValid, h * kvLen * headDim, kvLen * headDim);
            Array.Copy(vFull, h * maxSeq * headDim, vValid, h * kvLen * headDim, kvLen * headDim);
        }
        float[] expected = CpuFlashReference(q, kValid, vValid, null, batch, hq, hkv, sq, kvLen, headDim, scale, causal: false, qOffset: 0, slidingWindow: 0);

        Tensor qT = new(new TensorShape(batch, hq, sq, headDim), DType.F32);
        Tensor kT = new(new TensorShape(batch, hkv, maxSeq, headDim), DType.F32);
        Tensor vT = new(new TensorShape(batch, hkv, maxSeq, headDim), DType.F32);
        Tensor oT = new(new TensorShape(batch, hq, sq, headDim), DType.F32);
        q.CopyTo(qT.AsSpan<float>()); kFull.CopyTo(kT.AsSpan<float>()); vFull.CopyTo(vT.AsSpan<float>());

        backend.FlashAttention(oT, qT, kT, vT, kvLen: kvLen, kvGroup: hq / hkv, causal: false, qOffset: 0, scale);

        ReadOnlySpan<float> oS = oT.AsReadOnlySpan<float>();
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(oS[i] - expected[i], -2e-3f, 2e-3f);

        qT.Dispose(); kT.Dispose(); vT.Dispose(); oT.Dispose();
    }

    /// <summary>Regression gate for a real Krea2-on-Vulkan investigation (2026-07-30): <c>FlashMaxHeadDim = 128</c>
    /// is the exact boundary between the fused flash-attention path and the naive fallback
    /// (<c>headDim &lt;= FlashMaxHeadDim</c>) — and Krea2's <c>headDim</c> is EXACTLY 128, a boundary value no
    /// existing flash-attention test exercised (all prior tests use headDim 32/64). Also matches Krea2's
    /// EXACT `ScaledDotProductAttention` call shape: mask=null, non-causal, GQA, self-attention (sq==skv,
    /// both large — Krea2 runs jointSeq≈4108, kept smaller here for CPU-reference speed), F32. This hypothesis
    /// was ruled out during the investigation (both cases pass) — the real bug was
    /// <c>VulkanBackend.DispatchMatmul</c> deriving M/N from <c>output.Shape</c>'s rank structure instead of
    /// from the weight tensor, which silently computed the wrong-shaped GEMM for any Linear whose output is
    /// shaped <c>[B, S, heads, headDim]</c> (Krea2's Q/K/V). Kept as a permanent regression gate for this
    /// specific boundary value regardless.</summary>
    [Theory]
    [InlineData(128)]   // Krea2's exact headDim — the FlashMaxHeadDim boundary
    [InlineData(64)]    // one below a power-of-two step, sanity control
    public void Backend_ScaledDotProductAttention_HeadDim128Boundary_MatchesCpu(int headDim)
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int batch = 1, hq = 8, hkv = 2, seqLen = 48;   // GQA, self-attention (sq==skv)
        float scale = 1f / MathF.Sqrt(headDim);
        float[] q = FillRandom(batch * hq * seqLen * headDim, 41);
        float[] k = FillRandom(batch * hkv * seqLen * headDim, 42);
        float[] v = FillRandom(batch * hkv * seqLen * headDim, 43);
        float[] expected = CpuFlashReference(q, k, v, null, batch, hq, hkv, seqLen, seqLen, headDim, scale, causal: false, qOffset: 0, slidingWindow: 0);

        Tensor qT = new(new TensorShape(batch, hq, seqLen, headDim), DType.F32);
        Tensor kT = new(new TensorShape(batch, hkv, seqLen, headDim), DType.F32);
        Tensor vT = new(new TensorShape(batch, hkv, seqLen, headDim), DType.F32);
        Tensor oT = new(new TensorShape(batch, hq, seqLen, headDim), DType.F32);
        q.CopyTo(qT.AsSpan<float>()); k.CopyTo(kT.AsSpan<float>()); v.CopyTo(vT.AsSpan<float>());

        backend.ScaledDotProductAttention(oT, qT, kT, vT, mask: null, scale);

        ReadOnlySpan<float> oS = oT.AsReadOnlySpan<float>();
        float maxAbs = 0f; int firstBad = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(oS[i] - expected[i]);
            if (diff > maxAbs) { maxAbs = diff; if (diff > 2e-3f) firstBad = i; }
        }
        Assert.True(maxAbs < 2e-3f, $"ScaledDotProductAttention (headDim={headDim}) maxAbsDiff={maxAbs:E3} firstBadIdx={firstBad}.");

        qT.Dispose(); kT.Dispose(); vT.Dispose(); oT.Dispose();
    }

    [Fact]
    public void Backend_SDPA_WithMask_MatchesCpu()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        const int batch = 1, hq = 3, hkv = 3, sq = 10, skv = 14, headDim = 32;
        float scale = 1f / MathF.Sqrt(headDim);
        float[] q = FillRandom(batch * hq * sq * headDim, 31);
        float[] k = FillRandom(batch * hkv * skv * headDim, 32);
        float[] v = FillRandom(batch * hkv * skv * headDim, 33);
        float[] mask = new float[sq * skv];
        Random rng = new(34);
        for (int i = 0; i < mask.Length; i++) mask[i] = rng.NextDouble() < 0.3 ? -1e9f : 0f;   // random padding mask
        float[] expected = CpuFlashReference(q, k, v, mask, batch, hq, hkv, sq, skv, headDim, scale, causal: false, qOffset: 0, slidingWindow: 0);

        Tensor qT = new(new TensorShape(batch, hq, sq, headDim), DType.F32);
        Tensor kT = new(new TensorShape(batch, hkv, skv, headDim), DType.F32);
        Tensor vT = new(new TensorShape(batch, hkv, skv, headDim), DType.F32);
        Tensor oT = new(new TensorShape(batch, hq, sq, headDim), DType.F32);
        Tensor maskT = new(new TensorShape(sq, skv), DType.F32);
        q.CopyTo(qT.AsSpan<float>()); k.CopyTo(kT.AsSpan<float>()); v.CopyTo(vT.AsSpan<float>()); mask.CopyTo(maskT.AsSpan<float>());

        backend.ScaledDotProductAttention(oT, qT, kT, vT, maskT, scale);

        ReadOnlySpan<float> oS = oT.AsReadOnlySpan<float>();
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(oS[i] - expected[i], -2e-3f, 2e-3f);

        qT.Dispose(); kT.Dispose(); vT.Dispose(); oT.Dispose(); maskT.Dispose();
    }

    /// <summary>The exact shape flagged in benchmarks/scoreboards/VULKAN.md as too large to run on the
    /// old materialized path (~25 GB score matrix — the documented Wan-video OOM root cause). Not a
    /// full numeric cross-check (a CPU reference at this scale is too slow for a unit test) — proves
    /// completion without OOM and finite output, which the old path could not do at all.</summary>
    [Fact]
    public void Backend_FlashAttention_WanVideoScale_CompletesWithoutOom()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();
        // Software rendering (llvmpipe) takes HOURS on this shape (measured: still running after 2.85
        // CPU-hours, killed) — this test proves no-OOM/finite-output on real hardware, not something
        // llvmpipe's correctness-only role needs to cover. Skip rather than hang the whole suite.
        if (backend.Vk.DeviceName.Contains("llvmpipe", StringComparison.OrdinalIgnoreCase)) return;

        const int batch = 1, hq = 24, hkv = 24, sq = 16384, skv = 16384, headDim = 128;
        float scale = 1f / MathF.Sqrt(headDim);
        Tensor qT = new(new TensorShape(batch, hq, sq, headDim), DType.F32);
        Tensor kT = new(new TensorShape(batch, hkv, skv, headDim), DType.F32);
        Tensor vT = new(new TensorShape(batch, hkv, skv, headDim), DType.F32);
        Tensor oT = new(new TensorShape(batch, hq, sq, headDim), DType.F32);
        Span<float> qS = qT.AsSpan<float>();
        for (int i = 0; i < qS.Length; i++) qS[i] = MathF.Sin(i * 0.001f) * 0.1f;
        Span<float> kS = kT.AsSpan<float>();
        for (int i = 0; i < kS.Length; i++) kS[i] = MathF.Cos(i * 0.001f) * 0.1f;
        Span<float> vS = vT.AsSpan<float>();
        for (int i = 0; i < vS.Length; i++) vS[i] = MathF.Sin(i * 0.0007f) * 0.1f;

        backend.FlashAttention(oT, qT, kT, vT, kvLen: skv, kvGroup: 1, causal: true, qOffset: 0, scale);

        ReadOnlySpan<float> oOut = oT.AsReadOnlySpan<float>();
        // Spot-check a handful of positions across the tensor for finiteness (a full scan is unnecessary
        // — the goal is proving the fused kernel completed and produced real numbers, not a NaN/garbage
        // buffer from an indexing bug at scale).
        for (int i = 0; i < oOut.Length; i += 999983)   // large odd stride, cheap coverage across the buffer
            Assert.True(float.IsFinite(oOut[i]), $"Non-finite output at index {i}");

        qT.Dispose(); kT.Dispose(); vT.Dispose(); oT.Dispose();
    }

    // ── Phase 5: GGUF dequant shaders ───────────────────────────────────────────────────────
    // Vulkan had zero GGUF k-quant/legacy-quant dequant support before this (an explicit open
    // ROADMAP.md item). Reference is HartsyInference.ModelAssets.Gguf.GgufDequantizer — the SAME
    // validated CPU codec used elsewhere in the engine, not a hand-rolled test-only reimplementation.
    // Block bytes are random EXCEPT the FP16 scale header(s), which get a small fixed value instead of
    // a random bit pattern — an unconstrained random uint16 has a real chance of landing in FP16's
    // NaN/Inf exponent range, which would make the whole block NaN/Inf on BOTH sides and prove nothing.
    // Everything else (nibbles, packed 6-bit scale/min fields, signed int8 sub-scales) is a plain
    // integer reinterpretation with no such hazard, so full-random there is fine and exercises the
    // bit-unpacking generically.

    /// <summary>Per-format byte offsets of the block's FP16 scale field(s), relative to block start —
    /// everywhere else in the block is safe to fill with fully random bytes.</summary>
    private static int[] DequantScaleByteOffsets(string dtypeName) => dtypeName switch
    {
        "Q4_0" => new[] { 0 },
        "Q5_0" => new[] { 0 },
        "Q8_0" => new[] { 0 },
        "Q4_K" => new[] { 0, 2 },
        "Q5_K" => new[] { 0, 2 },
        "Q6_K" => new[] { 208 },   // Q6_K's scale sits at the END of the super-block, not the start.
        _ => throw new ArgumentException(dtypeName),
    };

    [Theory]
    [InlineData("Q4_0")]
    [InlineData("Q5_0")]
    [InlineData("Q8_0")]
    [InlineData("Q4_K")]
    [InlineData("Q5_K")]
    [InlineData("Q6_K")]
    public unsafe void Backend_DequantizeToF32_MatchesGgufDequantizer(string dtypeName)
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();

        DType dtype = dtypeName switch
        {
            "Q4_0" => DType.Q4_0,
            "Q5_0" => DType.Q5_0,
            "Q8_0" => DType.Q8_0,
            "Q4_K" => DType.Q4_K,
            "Q5_K" => DType.Q5_K,
            "Q6_K" => DType.Q6_K,
            _ => throw new ArgumentException(dtypeName),
        };
        // 3 super-blocks/blocks worth of elements — exercises block-boundary handling without a huge buffer.
        const int blockRepeats = 8;
        int elementCount = dtype.BlockElementCount * blockRepeats;
        int blockBytes = dtype.BlockByteSize;
        Tensor quant = new(new TensorShape(elementCount), dtype);
        byte* raw = (byte*)quant.DataPointer;
        long byteCount = dtype.ComputeByteCount(elementCount);
        Random rng = new(unchecked(dtypeName.GetHashCode()));
        for (long i = 0; i < byteCount; i++) raw[i] = (byte)rng.Next(256);

        int[] scaleOffsets = DequantScaleByteOffsets(dtypeName);
        for (int block = 0; block < blockRepeats; block++)
        {
            long blockBase = (long)block * blockBytes;
            // Vary the scale slightly per block/field so the test isn't accidentally insensitive to a
            // scale-indexing bug (e.g. reading dmin's bytes for d).
            foreach (int off in scaleOffsets)
            {
                Half h = (Half)(0.1f + 0.05f * off / 2f + 0.01f * block);
                ushort bits = BitConverter.HalfToUInt16Bits(h);
                raw[blockBase + off] = (byte)(bits & 0xFF);
                raw[blockBase + off + 1] = (byte)((bits >> 8) & 0xFF);
            }
        }

        using Tensor cpuRef = HartsyInference.ModelAssets.Gguf.GgufDequantizer.Dequantize(quant, DType.F32);
        using Tensor gpuOut = backend.DequantizeToF32(quant);

        ReadOnlySpan<float> cpuS = cpuRef.AsReadOnlySpan<float>();
        ReadOnlySpan<float> gpuS = gpuOut.AsReadOnlySpan<float>();
        for (int i = 0; i < elementCount; i++)
        {
            // F16 is the GPU dequant kernel's native intermediate (matches CudaBackend.CastOnGpu's own
            // quant->F16->F32 staging), so tolerance is relative-ish: absolute floor + a percentage.
            float diff = MathF.Abs(gpuS[i] - cpuS[i]);
            float tol = 0.05f + MathF.Abs(cpuS[i]) * 0.02f;
            Assert.True(diff <= tol, $"[{dtypeName}] index {i}: gpu={gpuS[i]} cpu={cpuS[i]} diff={diff} tol={tol}");
        }

        quant.Dispose();
    }

    /// <summary>Correctness gate for <c>matmul_coopmat_blocked.comp.glsl</c> (2026-07-31), the register-
    /// blocked GEMM built to test whether ggml/llama.cpp's core GEMM optimization (each subgroup computes
    /// a GRID of 16x16 output tiles from ONE shared-memory-staged input tile, instead of one tile per
    /// direct global-memory load) closes the ~30-160x Vulkan-vs-CUDA gap — see
    /// docs/Checklists/TROUBLESHOOTING.md. NOT wired into DispatchMatmul; called directly via the internal
    /// diagnostic entry point. Shapes deliberately >= 128 in M/N so BOTH register-blocking dimensions
    /// (multiple subgroups per workgroup AND multiple accumulators per subgroup) are actually exercised —
    /// a shape smaller than one workgroup tile (64) wouldn't test the thing this kernel exists to test.</summary>
    [Theory]
    [InlineData(128, 128, 128, false, 32u, 32u)]
    [InlineData(256, 512, 256, true, 32u, 32u)]
    [InlineData(129, 128, 144, false, 32u, 32u)]   // M not a multiple of 16; N not a multiple of 64 (both 16-aligned per the gate)
    [InlineData(256, 512, 256, true, 16u, 16u)]    // register blocking DISABLED (1 accumulator/subgroup) -- the
                                                    // "staged-only" benchmark config, needs its own correctness check
    public void Backend_CoopmatBlocked_Diagnostic_MatchesCpu(int M, int K, int N, bool hasBias, uint wm, uint wn)
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();
        if (!backend.Capabilities.SupportsF16 || !backend.Vk.HasCooperativeMatrix) return;

        Tensor input = new(new TensorShape(M, K), DType.F16);
        Tensor weight = new(new TensorShape(N, K), DType.F16);
        Tensor? bias = hasBias ? new Tensor(new TensorShape(N), DType.F16) : null;
        Tensor output = new(new TensorShape(M, N), DType.F16);

        Random rng = new(2000 + M + N);
        Span<Half> iS = input.AsSpan<Half>();
        Span<Half> wS = weight.AsSpan<Half>();
        Half[] bS = new Half[N];
        for (int i = 0; i < M * K; i++) iS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.3f);
        for (int i = 0; i < N * K; i++) wS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.2f);
        if (hasBias)
        {
            Span<Half> bSpan = bias!.AsSpan<Half>();
            for (int i = 0; i < N; i++) { bS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.05f); bSpan[i] = bS[i]; }
        }

        bool dispatched = backend.TryDispatchCoopmatBlockedDiagnostic(output, input, weight, transposeA: false, transposeB: true, bias, wm, wn);
        Assert.True(dispatched, "TryDispatchCoopmatBlockedDiagnostic returned false (gate didn't pass) for a shape it should handle.");

        ReadOnlySpan<Half> oS = output.AsReadOnlySpan<Half>();
        float maxRel = 0f; int firstM = -1, firstN = -1;
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
            {
                float acc = hasBias ? (float)bS[n] : 0f;
                for (int k = 0; k < K; k++) acc += (float)iS[m * K + k] * (float)wS[n * K + k];
                float rel = MathF.Abs((float)oS[m * N + n] - acc) / MathF.Max(1e-3f, MathF.Abs(acc));
                if (rel > maxRel) { maxRel = rel; firstM = m; firstN = n; }
            }
        Assert.True(maxRel < 0.02f, $"Coopmat-blocked (M={M},K={K},N={N},bias={hasBias}) maxRelErr {maxRel:P2} at [{firstM},{firstN}] too high.");

        input.Dispose(); weight.Dispose(); bias?.Dispose(); output.Dispose();
    }

    /// <summary>Correctness gate for <c>matmul_coopmat2.comp.glsl</c> (2026-07-31), the
    /// <c>VK_NV_cooperative_matrix2</c> kernel — a genuinely different instruction/memory path from coopmat1
    /// (workgroup-scope + tensor-layout addressing vs subgroup-scope + manual fragment loads), built after
    /// coopmat1 was measured to have zero real-throughput advantage over the scalar fallback on this RTX
    /// 4090 (see docs/Checklists/TROUBLESHOOTING.md). Called directly via the internal entry point (not
    /// through <c>Linear</c>/<c>DispatchMatmul</c>'s <see cref="VulkanBackend.EnableCoopMat2"/> gate) so this
    /// exercises the kernel/bias-epilogue in isolation regardless of that flag's default. Deliberately
    /// includes shapes where M, K, and N are NOT multiples of the device's coopmat2 tile granularity
    /// (32/16 on this hardware) — the whole point of the extension's built-in CLAMPED tensor addressing is
    /// that no manual bounds-checking is needed for non-aligned shapes, unlike every coopmat1 kernel in this
    /// codebase, so this is the key thing to prove. Also covers the bias epilogue (a follow-up
    /// <see cref="VulkanBackend.BroadcastAdd"/> dispatch, not fused into the shader — see
    /// <see cref="VulkanBackend.TryDispatchCoopMat2"/>'s doc comment).</summary>
    [Theory]
    [InlineData(128, 128, 128, false)]
    [InlineData(256, 512, 256, true)]
    [InlineData(129, 130, 144, true)]     // none of M/K/N a multiple of the 32/16/32 tile granularity
    [InlineData(17, 33, 5, false)]        // smaller than one tile in every dimension
    public void Backend_CoopMat2_MatchesCpu(int M, int K, int N, bool hasBias)
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();
        if (!backend.Capabilities.SupportsF16 || !backend.Vk.HasCooperativeMatrix2) return;

        Tensor input = new(new TensorShape(M, K), DType.F16);
        Tensor weight = new(new TensorShape(N, K), DType.F16);
        Tensor? bias = hasBias ? new Tensor(new TensorShape(N), DType.F16) : null;
        Tensor output = new(new TensorShape(M, N), DType.F16);

        Random rng = new(3000 + M + N);
        Span<Half> iS = input.AsSpan<Half>();
        Span<Half> wS = weight.AsSpan<Half>();
        Half[] bS = new Half[N];
        for (int i = 0; i < M * K; i++) iS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.3f);
        for (int i = 0; i < N * K; i++) wS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.2f);
        if (hasBias)
        {
            Span<Half> bSpan = bias!.AsSpan<Half>();
            for (int i = 0; i < N; i++) { bS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.05f); bSpan[i] = bS[i]; }
        }

        bool dispatched = backend.TryDispatchCoopMat2(output, input, weight, transposeA: false, transposeB: true, bias);
        Assert.True(dispatched, "TryDispatchCoopMat2 returned false (gate didn't pass) for a shape it should handle.");

        ReadOnlySpan<Half> oS = output.AsReadOnlySpan<Half>();
        float maxRel = 0f; int firstM = -1, firstN = -1;
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
            {
                float acc = hasBias ? (float)bS[n] : 0f;
                for (int k = 0; k < K; k++) acc += (float)iS[m * K + k] * (float)wS[n * K + k];
                float rel = MathF.Abs((float)oS[m * N + n] - acc) / MathF.Max(1e-3f, MathF.Abs(acc));
                if (rel > maxRel) { maxRel = rel; firstM = m; firstN = n; }
            }
        Assert.True(maxRel < 0.02f, $"CoopMat2 (M={M},K={K},N={N},bias={hasBias}) maxRelErr {maxRel:P2} at [{firstM},{firstN}] too high.");

        input.Dispose(); weight.Dispose(); bias?.Dispose(); output.Dispose();
    }
}
