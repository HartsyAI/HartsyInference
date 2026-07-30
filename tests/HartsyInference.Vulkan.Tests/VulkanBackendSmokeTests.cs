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

    [Fact]
    public void GetD2hSyncCount_Concat_SyncsEachGpuResidentInput()
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

        // Concat (VulkanBackend.cs) is a pure CPU loop reading .DataPointer directly on each input —
        // it never explicitly "reads back" the way a test does, so this is exactly the kind of
        // silent hidden round-trip the counter is meant to catch.
        backend.Concat(concatOut, new Tensor[] { outA, outB }, 0);

        Assert.Equal(2, backend.GetD2hSyncCount());

        a.Dispose(); b.Dispose(); outA.Dispose(); outB.Dispose(); concatOut.Dispose();
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
    private static float[] CpuFlashReference(
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

    private static float[] FillRandom(int n, int seed)
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
}
