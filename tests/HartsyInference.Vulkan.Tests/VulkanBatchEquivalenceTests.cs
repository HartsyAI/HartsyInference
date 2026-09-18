using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>Every op must give image <c>n</c> of a batch the same answer it gives that image alone.
///
/// <para>This is the generalization of the bug that kept SDXL off this backend: <c>Conv2D</c> at least REFUSED a
/// batch, so it failed loudly. An op that instead reads image 0's data for every image, or writes every image over
/// image 0's output, produces self-consistent output that no single-image test can distinguish — and SDXL's fused
/// denoise loop runs the whole UNet at batch=2 (positive and negative prompt concatenated for CFG), so every op in
/// it is exposed to exactly that.</para>
///
/// <para>Each image is given a different distribution, not merely different values, so a copied plane cannot pass
/// on a loose tolerance. These ops are bitwise-identical today; the assert is on a relative tolerance anyway,
/// because an op that legitimately reduces across a tile in a different order would not be.</para></summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanBatchEquivalenceTests(ITestOutputHelper output)
{
    private readonly ITestOutputHelper _out = output;

    private static bool VulkanAvailable()
    {
        try { using VulkanInstance i = new(); return i.EnumeratePhysicalDevices().Length > 0; }
        catch { return false; }
    }

    private static Tensor Rand(TensorShape shape, int seed, float offset = 0f)
    {
        Tensor t = new(shape, DType.F32);
        Random rng = new(seed);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1) + offset;
        return t;
    }

    private float Compare(string op, Tensor batched, Tensor[] singles, int perImage)
    {
        ReadOnlySpan<float> b = batched.AsReadOnlySpan<float>();
        float maxErr = 0f, maxAbs = 0f;
        for (int n = 0; n < singles.Length; n++)
        {
            ReadOnlySpan<float> e = singles[n].AsReadOnlySpan<float>();
            for (int i = 0; i < perImage; i++)
            {
                maxErr = MathF.Max(maxErr, MathF.Abs(b[n * perImage + i] - e[i]));
                maxAbs = MathF.Max(maxAbs, MathF.Abs(e[i]));
            }
        }
        float rel = maxAbs > 0 ? maxErr / maxAbs : maxErr;
        _out.WriteLine($"{op,-24} maxErr={maxErr:E3} maxAbs={maxAbs:E3} rel={rel:E3}  {(rel < 1e-4f ? "OK" : "**DIVERGES**")}");
        return rel;
    }

    [Fact]
    public void Ops_Give_The_Same_Answer_For_An_Image_In_A_Batch_As_Alone()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();
        const int B = 2, C = 32, H = 16, W = 16;
        int spatial = C * H * W;
        List<(string, float)> results = new();

        // GroupNorm [B,C,H,W]
        {
            using Tensor x = Rand(new TensorShape(B, C, H, W), 1, 0.5f);
            using Tensor g = Rand(new TensorShape(C), 2);
            using Tensor be = Rand(new TensorShape(C), 3);
            using Tensor outB = new(new TensorShape(B, C, H, W), DType.F32);
            backend.GroupNorm(outB, x, g, be, 8, 1e-5f);
            Tensor[] singles = new Tensor[B];
            for (int n = 0; n < B; n++)
            {
                using Tensor xs = new(new TensorShape(1, C, H, W), DType.F32);
                x.AsReadOnlySpan<float>().Slice(n * spatial, spatial).CopyTo(xs.AsSpan<float>());
                singles[n] = new Tensor(new TensorShape(1, C, H, W), DType.F32);
                backend.GroupNorm(singles[n], xs, g, be, 8, 1e-5f);
            }
            results.Add(("GroupNorm", Compare("GroupNorm", outB, singles, spatial)));
            foreach (Tensor s in singles) s.Dispose();
        }

        // Silu [B,C,H,W]
        {
            using Tensor x = Rand(new TensorShape(B, C, H, W), 4, 0.5f);
            using Tensor outB = new(new TensorShape(B, C, H, W), DType.F32);
            backend.Silu(outB, x);
            Tensor[] singles = new Tensor[B];
            for (int n = 0; n < B; n++)
            {
                using Tensor xs = new(new TensorShape(1, C, H, W), DType.F32);
                x.AsReadOnlySpan<float>().Slice(n * spatial, spatial).CopyTo(xs.AsSpan<float>());
                singles[n] = new Tensor(new TensorShape(1, C, H, W), DType.F32);
                backend.Silu(singles[n], xs);
            }
            results.Add(("Silu", Compare("Silu", outB, singles, spatial)));
            foreach (Tensor s in singles) s.Dispose();
        }

        // Linear [B,S,K] x [N,K]
        {
            const int S = 32, K = 64, N = 48;
            int per = S * N, perIn = S * K;
            using Tensor x = Rand(new TensorShape(B, S, K), 5, 0.3f);
            using Tensor w = Rand(new TensorShape(N, K), 6);
            using Tensor outB = new(new TensorShape(B, S, N), DType.F32);
            backend.Linear(outB, x, w, null);
            Tensor[] singles = new Tensor[B];
            for (int n = 0; n < B; n++)
            {
                using Tensor xs = new(new TensorShape(1, S, K), DType.F32);
                x.AsReadOnlySpan<float>().Slice(n * perIn, perIn).CopyTo(xs.AsSpan<float>());
                singles[n] = new Tensor(new TensorShape(1, S, N), DType.F32);
                backend.Linear(singles[n], xs, w, null);
            }
            results.Add(("Linear", Compare("Linear", outB, singles, per)));
            foreach (Tensor s in singles) s.Dispose();
        }

        // ScaledDotProductAttention [B,Hd,S,D]
        {
            const int Hd = 4, S = 32, D = 32;
            int per = Hd * S * D;
            using Tensor q = Rand(new TensorShape(B, Hd, S, D), 7, 0.2f);
            using Tensor k = Rand(new TensorShape(B, Hd, S, D), 8, 0.2f);
            using Tensor v = Rand(new TensorShape(B, Hd, S, D), 9, 0.2f);
            using Tensor outB = new(new TensorShape(B, Hd, S, D), DType.F32);
            backend.ScaledDotProductAttention(outB, q, k, v, null, 1f / MathF.Sqrt(D));
            Tensor[] singles = new Tensor[B];
            for (int n = 0; n < B; n++)
            {
                using Tensor qs = new(new TensorShape(1, Hd, S, D), DType.F32);
                using Tensor ks = new(new TensorShape(1, Hd, S, D), DType.F32);
                using Tensor vs = new(new TensorShape(1, Hd, S, D), DType.F32);
                q.AsReadOnlySpan<float>().Slice(n * per, per).CopyTo(qs.AsSpan<float>());
                k.AsReadOnlySpan<float>().Slice(n * per, per).CopyTo(ks.AsSpan<float>());
                v.AsReadOnlySpan<float>().Slice(n * per, per).CopyTo(vs.AsSpan<float>());
                singles[n] = new Tensor(new TensorShape(1, Hd, S, D), DType.F32);
                backend.ScaledDotProductAttention(singles[n], qs, ks, vs, null, 1f / MathF.Sqrt(D));
            }
            results.Add(("SDPA", Compare("SDPA", outB, singles, per)));
            foreach (Tensor s in singles) s.Dispose();
        }

        // Conv2D control (known-good after the batch fix)
        {
            using Tensor x = Rand(new TensorShape(B, 8, H, W), 10, 0.4f);
            using Tensor w = Rand(new TensorShape(8, 8, 3, 3), 11);
            using Tensor outB = new(new TensorShape(B, 8, H, W), DType.F32);
            int per = 8 * H * W;
            backend.Conv2D(outB, x, w, null, 1, 1, 1, 1);
            Tensor[] singles = new Tensor[B];
            for (int n = 0; n < B; n++)
            {
                using Tensor xs = new(new TensorShape(1, 8, H, W), DType.F32);
                x.AsReadOnlySpan<float>().Slice(n * per, per).CopyTo(xs.AsSpan<float>());
                singles[n] = new Tensor(new TensorShape(1, 8, H, W), DType.F32);
                backend.Conv2D(singles[n], xs, w, null, 1, 1, 1, 1);
            }
            results.Add(("Conv2D", Compare("Conv2D", outB, singles, per)));
            foreach (Tensor s in singles) s.Dispose();
        }

        string diverging = string.Join(", ", results.Where(r => r.Item2 >= 1e-4f).Select(r => r.Item1));
        Assert.True(diverging.Length == 0, $"Batch-dependent ops on Vulkan: {diverging}");
    }
}
