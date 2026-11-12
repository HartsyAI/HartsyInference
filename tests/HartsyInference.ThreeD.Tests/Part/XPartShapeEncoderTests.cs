using System.Diagnostics;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Hunyuan3D.Part;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Part;

/// <summary>The X-Part ShapeVAE encoder: a unit lane on tiny random weights (shape, query selection, validation) and an integration lane against upstream's own modules on the
/// CPU. Set <c>HARTSY_XPART_VAE_REF</c> (a fixture from <c>tests/python-reference/dump_xpart_shapevae_encoder.py</c>), <c>HARTSY_XPART_SHAPEVAE</c> (<c>shapevae.safetensors</c>) and
/// <c>HARTSY_XPART_CONDITIONER</c> (<c>conditioner.safetensors</c>; its <c>obj_encoder</c> is compared too).</summary>
public sealed class XPartShapeEncoderTests(ITestOutputHelper output)
{
    private static readonly XPartShapeEncoderConfig Tiny = new(Width: 16, Heads: 2, Layers: 2, NumFreqs: 2, PointFeats: 4);

    private static Dictionary<string, Tensor> RandomWeights(XPartShapeEncoderConfig c, int seed)
    {
        Random random = new(seed);
        Dictionary<string, Tensor> w = [];
        void Add(string key, params long[] shape)
        {
            Tensor t = new(new TensorShape(shape), DType.F32);
            foreach (ref float v in t.AsSpan<float>())
                v = (float)(random.NextDouble() - 0.5) * 0.4f;
            w[key] = t;
        }

        void Norm(string key, long width)
        {
            Add($"{key}.weight", width);
            Add($"{key}.bias", width);
            foreach (ref float v in w[$"{key}.weight"].AsSpan<float>())
                v += 1f;
        }

        int width = c.Width, hd = c.HeadDim;
        Add("encoder.input_proj.weight", width, c.InputWidth);
        Add("encoder.input_proj.bias", width);
        string x = "encoder.cross_attn";
        Norm($"{x}.ln_1", width); Norm($"{x}.ln_2", width); Norm($"{x}.ln_3", width);
        Norm($"{x}.attn.attention.q_norm", hd); Norm($"{x}.attn.attention.k_norm", hd);
        Add($"{x}.attn.c_q.weight", width, width); Add($"{x}.attn.c_kv.weight", 2 * width, width);
        Add($"{x}.attn.c_proj.weight", width, width); Add($"{x}.attn.c_proj.bias", width);
        Add($"{x}.mlp.c_fc.weight", 4 * width, width); Add($"{x}.mlp.c_fc.bias", 4 * width);
        Add($"{x}.mlp.c_proj.weight", width, 4 * width); Add($"{x}.mlp.c_proj.bias", width);
        for (int i = 0; i < c.Layers; i++)
        {
            string b = $"encoder.self_attn.resblocks.{i}";
            Norm($"{b}.ln_1", width); Norm($"{b}.ln_2", width);
            Norm($"{b}.attn.attention.q_norm", hd); Norm($"{b}.attn.attention.k_norm", hd);
            Add($"{b}.attn.c_qkv.weight", 3 * width, width);
            Add($"{b}.attn.c_proj.weight", width, width); Add($"{b}.attn.c_proj.bias", width);
            Add($"{b}.mlp.c_fc.weight", 4 * width, width); Add($"{b}.mlp.c_fc.bias", 4 * width);
            Add($"{b}.mlp.c_proj.weight", width, 4 * width); Add($"{b}.mlp.c_proj.bias", width);
        }

        Norm("encoder.ln_post", width);
        return w;
    }

    private static (float[] Points, float[] Feats) Cloud(int count, int seed)
    {
        Random random = new(seed);
        float[] points = new float[3 * count], feats = new float[4 * count];
        for (int i = 0; i < count; i++)
        {
            for (int j = 0; j < 3; j++)
                points[3 * i + j] = (float)(random.NextDouble() - 0.5);
            for (int j = 0; j < 4; j++)
                feats[4 * i + j] = (float)(random.NextDouble() - 0.5);
        }

        return (points, feats);
    }

    [Fact]
    public void Encode_ReturnsOneTokenPerQueryAndSelectsTheGivenPoints()
    {
        using CpuBackend backend = new();
        Dictionary<string, Tensor> weights = RandomWeights(Tiny, 1);
        using XPartShapeEncoder encoder = new(backend, Tiny);
        encoder.LoadWeights(weights);
        (float[] points, float[] feats) = Cloud(64, 2);

        using Tensor a = encoder.Encode(points, feats, [3, 10, 40]);
        using Tensor b = encoder.Encode(points, feats, [3, 10, 40]);
        using Tensor reordered = encoder.Encode(points, feats, [40, 3, 10]);
        Assert.Equal(new TensorShape(1, 3, 16), a.Shape);
        ReadOnlySpan<float> av = a.AsSpan<float>(), bv = b.AsSpan<float>(), rv = reordered.AsSpan<float>();
        Assert.All(av.ToArray(), v => Assert.True(float.IsFinite(v)));
        Assert.Equal(av.ToArray(), bv.ToArray());
        // The tokens are queries over a shared key set: the same query gives the same token wherever it sits.
        for (int d = 0; d < 16; d++)
        {
            Assert.Equal(av[d], rv[16 + d], 4);
            Assert.Equal(av[16 + d], rv[32 + d], 4);
            Assert.Equal(av[32 + d], rv[d], 4);
        }

        foreach (Tensor t in weights.Values)
            t.Dispose();
    }

    [Fact]
    public void Encode_WithAQueryCountTakesTheFarthestPointsFromTheStartIndex()
    {
        using CpuBackend backend = new();
        Dictionary<string, Tensor> weights = RandomWeights(Tiny, 1);
        using XPartShapeEncoder encoder = new(backend, Tiny);
        encoder.LoadWeights(weights);
        (float[] points, float[] feats) = Cloud(64, 5);
        int[] picked = HartsyInference.ThreeD.Geometry.Ops.SurfaceSampler.FarthestPointIndices(points, 6, 3);
        using Tensor byIndex = encoder.Encode(points, feats, picked);
        using Tensor byCount = encoder.Encode(points, feats, 6, 3);
        Assert.Equal(byIndex.AsSpan<float>().ToArray(), byCount.AsSpan<float>().ToArray());
        foreach (Tensor t in weights.Values)
            t.Dispose();
    }

    [Fact]
    public void Dispose_LeavesTheCallersF32WeightsUsable()
    {
        using CpuBackend backend = new();
        Dictionary<string, Tensor> weights = RandomWeights(Tiny, 1);
        (float[] points, float[] feats) = Cloud(16, 6);
        float[] first;
        using (XPartShapeEncoder one = new(backend, Tiny))
        {
            one.LoadWeights(weights);
            using Tensor t = one.Encode(points, feats, [1, 2]);
            first = t.AsSpan<float>().ToArray();
        }

        // The weights are F32, so the encoder uses the caller's tensors as they are: disposing it must not free them, and a second encoder over the same dictionary agrees.
        using XPartShapeEncoder two = new(backend, Tiny);
        two.LoadWeights(weights);
        using Tensor again = two.Encode(points, feats, [1, 2]);
        Assert.Equal(first, again.AsSpan<float>().ToArray());
        foreach (Tensor t in weights.Values)
            t.Dispose();
    }

    [Fact]
    public void Encode_RejectsMalformedInputAndWeights()
    {
        using CpuBackend backend = new();
        Dictionary<string, Tensor> weights = RandomWeights(Tiny, 1);
        using XPartShapeEncoder encoder = new(backend, Tiny);
        (float[] points, float[] feats) = Cloud(8, 3);
        Assert.Throws<InvalidOperationException>(() => encoder.Encode(points, feats, [0]));
        encoder.LoadWeights(weights);
        Assert.Throws<ArgumentException>(() => encoder.Encode(points, feats[..^1], [0]));
        Assert.Throws<ArgumentException>(() => encoder.Encode(points, feats, []));
        Assert.Throws<ArgumentOutOfRangeException>(() => encoder.Encode(points, feats, [8]));
        using Tensor all = encoder.Encode(points, feats, 8);
        Assert.Equal(8, all.Shape[1]);
        Assert.Throws<ArgumentOutOfRangeException>(() => encoder.Encode(points, feats, 9));
        Assert.Throws<ArgumentOutOfRangeException>(() => encoder.Encode(points, feats, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => encoder.Encode(points, feats, 2, 8));
        Assert.Throws<InvalidOperationException>(() => encoder.LoadWeights(weights));

        weights.Remove("encoder.ln_post.bias");
        using XPartShapeEncoder broken = new(backend, Tiny);
        Assert.Throws<KeyNotFoundException>(() => broken.LoadWeights(weights));
        foreach (Tensor t in weights.Values)
            t.Dispose();
    }

    private static unsafe float[] Values(Tensor t)
    {
        float[] values = new float[t.ElementCount];
        new ReadOnlySpan<float>((float*)t.DataPointer, values.Length).CopyTo(values);
        return values;
    }

    private void Compare(string stage, float[] actual, float[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double max = 0, diff2 = 0, ref2 = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double a = actual[i], b = expected[i], d = a - b;
            max = Math.Max(max, Math.Abs(d));
            diff2 += d * d;
            ref2 += b * b;
            sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
        }

        double n = expected.Length, rel = Math.Sqrt(diff2 / ref2);
        double corr = (sab / n - sa / n * (sb / n)) / Math.Sqrt((saa / n - sa / n * (sa / n)) * (sbb / n - sb / n * (sb / n)));
        output.WriteLine($"{stage,-26} n={expected.Length,8}  max|diff|={max:E3}  relRMS={rel:E3}  corr={corr:F6}");
        Assert.True(rel < 1e-3 && corr > 0.99999, $"{stage}: relRMS {rel:E3}, correlation {corr:F6}");
    }

    [RealWeightsFact("HARTSY_XPART_VAE_REF", "HARTSY_XPART_SHAPEVAE", "HARTSY_XPART_CONDITIONER")]
    [Trait("Category", "Integration")]
    public void Encoder_MatchesUpstreamOnTheCpu()
    {
        using SafeTensorsLoader refLoader = new();
        refLoader.Load(Environment.GetEnvironmentVariable("HARTSY_XPART_VAE_REF")!);
        Dictionary<string, Tensor> fixture = refLoader.GetAllTensors();
        float[] points = Values(fixture["points"]), feats = Values(fixture["feats"]);
        int[] queries = Values(fixture["query_idx"]).Select(v => (int)v).ToArray();
        using CpuBackend backend = new();

        foreach ((string name, string path, string prefix) in new[]
        {
            ("vae", Environment.GetEnvironmentVariable("HARTSY_XPART_SHAPEVAE")!, ""),
            ("obj", Environment.GetEnvironmentVariable("HARTSY_XPART_CONDITIONER")!, "obj_encoder"),
            ("local", Environment.GetEnvironmentVariable("HARTSY_XPART_CONDITIONER")!, "geo_encoder.local_encoder"),
        })
        {
            using SafeTensorsLoader loader = new();
            loader.Load(path);
            Dictionary<string, Tensor> weights = loader.GetAllTensors();
            Stopwatch watch = Stopwatch.StartNew();
            using XPartShapeEncoder encoder = new(backend);
            encoder.LoadWeights(weights, prefix);
            output.WriteLine($"{name}: loaded in {watch.Elapsed.TotalSeconds:F1} s");
            watch.Restart();
            using Tensor latents = encoder.Encode(points, feats, queries);
            output.WriteLine($"{name}: {points.Length / 3} points, {queries.Length} queries, encoded in {watch.Elapsed.TotalSeconds:F1} s");
            Compare($"{name} latents", Values(latents), Values(fixture[$"{name}.latents"]));
        }
    }
}
