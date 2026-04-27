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
}
