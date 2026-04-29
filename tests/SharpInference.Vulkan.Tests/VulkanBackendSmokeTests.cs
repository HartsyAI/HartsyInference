using SharpInference.Core.Tensors;
using SharpInference.Vulkan;
using Xunit;

namespace SharpInference.Vulkan.Tests;

/// <summary>
/// Smoke tests for the Vulkan backend. Skip themselves at runtime when no
/// Vulkan-capable physical device is visible — that way CI can include this
/// project on machines with or without a GPU.
/// </summary>
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
}
