using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Sam3dObjects;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Sam3dObjects;

public sealed class SamEmbedderFuserTests
{
    private const int Fused = 4;
    private static readonly SamFuserEmbedder[] Embedders =
    [
        new(4, [("image", "cropped"), ("rgb_image", "full")]),
        new(2, [("pointmap", "cropped")]),
    ];

    private static Tensor Filled(Random rng, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>())
            v = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    private static Dictionary<string, Tensor> Weights(Random rng)
    {
        int hidden = SamEmbedderFuser.HiddenDim(Fused, 4.0);
        Dictionary<string, Tensor> w = new() { ["idx_emb"] = Filled(rng, 3, Fused) };
        for (int i = 0; i < Embedders.Length; i++)
        {
            int d = Embedders[i].EmbedDim;
            w[$"projection_nets.{i}.0.weight"] = Filled(rng, d);
            w[$"projection_nets.{i}.0.bias"] = Filled(rng, d);
            w[$"projection_nets.{i}.1.w1.weight"] = Filled(rng, hidden, d);
            w[$"projection_nets.{i}.1.w3.weight"] = Filled(rng, hidden, d);
            w[$"projection_nets.{i}.1.w2.weight"] = Filled(rng, Fused, hidden);
        }

        return w;
    }

    // Independent reference in doubles: LayerNorm(eps 1e-5) -> w2(silu(w1 x) * w3 x) -> + idx_emb row.
    private static double[] Reference(Dictionary<string, Tensor> w, int embedder, ReadOnlySpan<float> x, int group, bool drop)
    {
        int d = Embedders[embedder].EmbedDim, hidden = SamEmbedderFuser.HiddenDim(Fused, 4.0);
        string p = $"projection_nets.{embedder}.";
        double mean = 0, var = 0;
        for (int i = 0; i < d; i++) mean += x[i] / d;
        for (int i = 0; i < d; i++) var += (x[i] - mean) * (x[i] - mean) / d;
        double[] n = new double[d];
        for (int i = 0; i < d; i++)
            n[i] = (x[i] - mean) / Math.Sqrt(var + 1e-5) * w[p + "0.weight"].AsReadOnlySpan<float>()[i] + w[p + "0.bias"].AsReadOnlySpan<float>()[i];
        ReadOnlySpan<float> w1 = w[p + "1.w1.weight"].AsReadOnlySpan<float>(), w3 = w[p + "1.w3.weight"].AsReadOnlySpan<float>();
        ReadOnlySpan<float> w2 = w[p + "1.w2.weight"].AsReadOnlySpan<float>();
        double[] h = new double[hidden];
        for (int j = 0; j < hidden; j++)
        {
            double a = 0, b = 0;
            for (int i = 0; i < d; i++)
            {
                a += w1[j * d + i] * n[i];
                b += w3[j * d + i] * n[i];
            }

            h[j] = a / (1 + Math.Exp(-a)) * b;
        }

        double[] result = new double[Fused];
        for (int c = 0; c < Fused; c++)
        {
            double s = 0;
            for (int j = 0; j < hidden; j++) s += w2[c * hidden + j] * h[j];
            result[c] = drop ? 0 : s + w["idx_emb"].AsReadOnlySpan<float>()[group * Fused + c];
        }

        return result;
    }

    private static void DisposeAll(Dictionary<string, Tensor> tensors)
    {
        foreach (Tensor t in tensors.Values) t.Dispose();
    }

    [Fact]
    public void HiddenDim_FollowsTheLlamaRuleAndMatchesTheShippedWidth()
    {
        // 1024-wide fused tokens with multiplier 4: int(2 · 4096 / 3) = 2730, rounded up to a multiple of 256.
        Assert.Equal(2816, SamEmbedderFuser.HiddenDim(1024, 4.0));
        Assert.Equal(256, SamEmbedderFuser.HiddenDim(4, 4.0));
    }

    [Fact]
    public void Fuse_ConcatenatesProjectedCallsInOrderWithGroupEmbeddings()
    {
        Random rng = new(5);
        Dictionary<string, Tensor> w = Weights(rng);
        Dictionary<string, Tensor> tokens = new()
        {
            ["image"] = Filled(rng, 3, 4), ["rgb_image"] = Filled(rng, 2, 4), ["pointmap"] = Filled(rng, 2, 2),
        };
        using CpuBackend backend = new();
        SamEmbedderFuser fuser = new(Embedders, Fused, w);

        using Tensor fused = fuser.Fuse(backend, tokens);

        Assert.Equal(new TensorShape(7, Fused), fused.Shape);
        ReadOnlySpan<float> got = fused.AsReadOnlySpan<float>();
        // Rows 0-2 image (group 0), rows 3-4 rgb_image (group 1), rows 5-6 pointmap (group 0).
        (string Key, int Embedder, int Group, int FirstRow)[] layout = [("image", 0, 0, 0), ("rgb_image", 0, 1, 3), ("pointmap", 1, 0, 5)];
        foreach ((string key, int embedder, int group, int firstRow) in layout)
        {
            int d = Embedders[embedder].EmbedDim;
            for (int r = 0; r < tokens[key].Shape[0]; r++)
            {
                double[] expected = Reference(w, embedder, tokens[key].AsReadOnlySpan<float>().Slice(r * d, d), group, false);
                for (int c = 0; c < Fused; c++)
                    Assert.Equal(expected[c], got[(firstRow + r) * Fused + c], 3);
            }
        }

        DisposeAll(w);
        DisposeAll(tokens);
    }

    [Fact]
    public void Fuse_ForceDropZeroesOnlyTheNamedKeywords()
    {
        Random rng = new(6);
        Dictionary<string, Tensor> w = Weights(rng);
        Dictionary<string, Tensor> tokens = new()
        {
            ["image"] = Filled(rng, 1, 4), ["rgb_image"] = Filled(rng, 1, 4), ["pointmap"] = Filled(rng, 1, 2),
        };
        using CpuBackend backend = new();
        SamEmbedderFuser fuser = new(Embedders, Fused, w);

        using Tensor fused = fuser.Fuse(backend, tokens, new HashSet<string> { "pointmap" });

        ReadOnlySpan<float> got = fused.AsReadOnlySpan<float>();
        Assert.All(got.Slice(2 * Fused, Fused).ToArray(), v => Assert.Equal(0f, v));
        Assert.Contains(got.Slice(0, Fused).ToArray(), v => v != 0f);
        DisposeAll(w);
        DisposeAll(tokens);
    }

    [Fact]
    public void Fuse_RejectsMissingKeywordsWrongWidthsAndBadWeights()
    {
        Random rng = new(7);
        Dictionary<string, Tensor> w = Weights(rng);
        using CpuBackend backend = new();
        SamEmbedderFuser fuser = new(Embedders, Fused, w);
        Dictionary<string, Tensor> tokens = new() { ["image"] = Filled(rng, 1, 4), ["rgb_image"] = Filled(rng, 1, 5), ["pointmap"] = Filled(rng, 1, 2) };

        Assert.Throws<ArgumentException>(() => fuser.Fuse(backend, new Dictionary<string, Tensor> { ["image"] = tokens["image"] }));
        Assert.Throws<ArgumentException>(() => fuser.Fuse(backend, tokens));
        w.Remove("idx_emb");
        Assert.Throws<InvalidOperationException>(() => new SamEmbedderFuser(Embedders, Fused, w));
        DisposeAll(w);
        DisposeAll(tokens);
    }
}
