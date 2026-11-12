using System.Diagnostics;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Hunyuan3D;
using HartsyInference.ThreeD.Models.Hunyuan3D.Part;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Part;

/// <summary>The X-Part DiT: a unit lane on tiny random weights (shapes, part ids, validation) and an integration lane that compares the block kinds, the input embedding and the final layer with
/// upstream's own modules on the CPU. Set <c>HARTSY_XPART_DIT_REF</c> and <c>HARTSY_XPART_DIT_SUBSET</c> (the two files written by <c>tests/python-reference/dump_xpart_dit.py</c>).</summary>
public sealed class XPartDitTests(ITestOutputHelper output)
{
    private static readonly XPartDitConfig Tiny = new(Width: 32, Depth: 3, Heads: 2, LatentChannels: 4, MoeBlocks: 1, Experts: 2, TopK: 1, ValidParts: 5, ContextWidth: 8);

    private static Dictionary<string, Tensor> RandomWeights(XPartDitConfig c, int seed)
    {
        Random random = new(seed);
        Dictionary<string, Tensor> w = [];
        void Add(string key, params long[] shape)
        {
            Tensor t = new(new TensorShape(shape), DType.F32);
            foreach (ref float v in t.AsSpan<float>())
                v = (float)(random.NextDouble() - 0.5) * 0.3f;
            w[key] = t;
        }

        void Norm(string key, long width)
        {
            Add($"{key}.weight", width);
            Add($"{key}.bias", width);
        }

        int width = c.Width, hd = width / c.Heads, ctx = 8;
        Add("x_embedder.weight", width, c.LatentChannels); Add("x_embedder.bias", width);
        Add("t_embedder.mlp.0.weight", 4 * width, width); Add("t_embedder.mlp.0.bias", 4 * width);
        Add("t_embedder.mlp.2.weight", width, 4 * width); Add("t_embedder.mlp.2.bias", width);
        Add("part_embed", c.ValidParts, width);
        Norm("final_layer.norm_final", width);
        Add("final_layer.linear.weight", c.LatentChannels, width); Add("final_layer.linear.bias", c.LatentChannels);
        for (int i = 0; i < c.Depth; i++)
        {
            string b = $"blocks.{i}";
            Norm($"{b}.norm1", width); Norm($"{b}.norm2", width); Norm($"{b}.norm2_2", width); Norm($"{b}.norm3", width);
            foreach (string attn in new[] { "attn1", "attn2", "attn2_2" })
            {
                long kv = attn == "attn1" ? width : ctx;
                Add($"{b}.{attn}.to_q.weight", width, width); Add($"{b}.{attn}.to_k.weight", width, kv); Add($"{b}.{attn}.to_v.weight", width, kv);
                Add($"{b}.{attn}.q_norm.weight", hd); Add($"{b}.{attn}.k_norm.weight", hd);
                Add($"{b}.{attn}.out_proj.weight", width, width); Add($"{b}.{attn}.out_proj.bias", width);
            }

            if (i > c.Depth / 2)
            {
                Add($"{b}.skip_linear.weight", width, 2 * width); Add($"{b}.skip_linear.bias", width);
                Norm($"{b}.skip_norm", width);
            }

            if (c.Depth - i <= c.MoeBlocks)
            {
                Add($"{b}.moe.gate.weight", c.Experts, width);
                Add($"{b}.moe.shared_experts.net.0.proj.weight", 4 * width, width); Add($"{b}.moe.shared_experts.net.0.proj.bias", 4 * width);
                Add($"{b}.moe.shared_experts.net.2.weight", width, 4 * width); Add($"{b}.moe.shared_experts.net.2.bias", width);
                for (int e = 0; e < c.Experts; e++)
                {
                    Add($"{b}.moe.experts.{e}.net.0.proj.weight", 4 * width, width); Add($"{b}.moe.experts.{e}.net.0.proj.bias", 4 * width);
                    Add($"{b}.moe.experts.{e}.net.2.weight", width, 4 * width); Add($"{b}.moe.experts.{e}.net.2.bias", width);
                }
            }
            else
            {
                Add($"{b}.mlp.fc1.weight", 4 * width, width); Add($"{b}.mlp.fc1.bias", 4 * width);
                Add($"{b}.mlp.fc2.weight", width, 4 * width); Add($"{b}.mlp.fc2.bias", width);
            }
        }

        return w;
    }

    private static Tensor Random3(int a, int b, int c, int seed)
    {
        Random random = new(seed);
        Tensor t = new(new TensorShape(a, b, c), DType.F32);
        foreach (ref float v in t.AsSpan<float>())
            v = (float)(random.NextDouble() - 0.5);
        return t;
    }

    [Fact]
    public void Forward_ReturnsOneVelocityPerPartTokenAndDependsOnThePartIds()
    {
        using CpuBackend backend = new();
        Dictionary<string, Tensor> weights = RandomWeights(Tiny, 1);
        using XPartDit dit = new(Tiny);
        dit.LoadWeights(weights);
        using Tensor latents = Random3(3, 6, 4, 2), obj = Random3(3, 5, 8, 3), part = Random3(3, 7, 8, 4);
        // Contexts are 8 wide here (the real ones are 1024): the tiny weights above take kdim 8.
        using Tensor a = dit.Forward(backend, latents, 0.3f, obj, part, [0, 1, 2]);
        using Tensor b = dit.Forward(backend, latents, 0.3f, obj, part, [0, 1, 2]);
        using Tensor swapped = dit.Forward(backend, latents, 0.3f, obj, part, [2, 1, 0]);
        Assert.Equal(new TensorShape(3, 6, 4), a.Shape);
        Assert.All(a.AsSpan<float>().ToArray(), v => Assert.True(float.IsFinite(v)));
        Assert.Equal(a.AsSpan<float>().ToArray(), b.AsSpan<float>().ToArray());
        Assert.NotEqual(a.AsSpan<float>().ToArray(), swapped.AsSpan<float>().ToArray());

        Assert.Throws<ArgumentException>(() => dit.Forward(backend, latents, 0.3f, obj, part, [0, 1]));
        using Tensor wide = Random3(3, 5, 16, 9);
        Assert.Throws<ArgumentException>(() => dit.Forward(backend, latents, 0.3f, wide, part, [0, 1, 2]));
        Assert.Throws<InvalidOperationException>(() => dit.LoadWeights(weights));
        Assert.Throws<ArgumentOutOfRangeException>(() => dit.Forward(backend, latents, 0.3f, obj, part, [0, 1, 5]));
        foreach (Tensor t in weights.Values)
            t.Dispose();
    }

    [Fact]
    public void HalfWeights_KeepTheBf16MatricesAsF16AndAgreeWithTheF32Path()
    {
        using CpuBackend backend = new();
        // BF16 checkpoint: every matrix rounded to BF16 (vectors stay as they are, the released file has BF16 vectors too, which widen exactly).
        Dictionary<string, Tensor> bf16 = RandomWeights(Tiny, 4).ToDictionary(p => p.Key, p => p.Value.Shape.Rank == 2 ? p.Value.CastTo(DType.BF16) : p.Value);
        using XPartDit full = new(Tiny), half = new(Tiny);
        full.LoadWeights(bf16);
        half.LoadWeights(bf16, halfWeights: true);
        using Tensor latents = Random3(2, 6, 4, 2), obj = Random3(2, 5, 8, 3), part = Random3(2, 7, 8, 4);

        using Tensor a = full.Forward(backend, latents, 0.3f, obj, part, [1, 2]);
        using Tensor b = half.Forward(backend, latents, 0.3f, obj, part, [1, 2]);

        double diff = 0, reference = 0;
        for (int i = 0; i < a.ElementCount; i++)
        {
            diff += Math.Pow(a.AsSpan<float>()[i] - b.AsSpan<float>()[i], 2);
            reference += Math.Pow(a.AsSpan<float>()[i], 2);
        }

        Assert.True(Math.Sqrt(diff / reference) < 1e-3, $"relRMS {Math.Sqrt(diff / reference):E2}");
        foreach (Tensor t in bf16.Values)
            t.Dispose();
    }

    [Fact]
    public void LoadWeights_RejectsAMissingSecondCrossAttentionAndAWrongShape()
    {
        Dictionary<string, Tensor> weights = RandomWeights(Tiny, 1);
        using XPartDit wrong = new(Tiny);
        weights["part_embed"] = new Tensor(new TensorShape(4, 32), DType.F32);
        Assert.Throws<InvalidDataException>(() => wrong.LoadEdgeWeights(weights));
        weights["part_embed"] = new Tensor(new TensorShape(5, 32), DType.F32);

        foreach (string key in weights.Keys.Where(k => k.Contains("attn2_2", StringComparison.Ordinal)).ToArray())
            weights.Remove(key);
        using XPartDit missing = new(Tiny);
        Assert.Throws<InvalidDataException>(() => missing.LoadWeights(weights));
    }

    private static unsafe float[] Values(Tensor t)
    {
        float[] values = new float[t.ElementCount];
        new ReadOnlySpan<float>((float*)t.DataPointer, values.Length).CopyTo(values);
        return values;
    }

    private void Compare(string stage, float[] actual, float[] expected, double tolerance = 1e-4)
    {
        Assert.Equal(expected.Length, actual.Length);
        double max = 0, diff2 = 0, ref2 = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = (double)actual[i] - expected[i];
            max = Math.Max(max, Math.Abs(d));
            diff2 += d * d;
            ref2 += (double)expected[i] * expected[i];
        }

        double rel = Math.Sqrt(diff2 / ref2);
        output.WriteLine($"{stage,-34} n={expected.Length,8}  max|diff|={max:E3}  relRMS={rel:E3}");
        Assert.True(rel < tolerance, $"{stage}: relRMS {rel:E3} is above {tolerance:E1}");
    }

    [RealWeightsFact("HARTSY_XPART_DIT_REF", "HARTSY_XPART_DIT_SUBSET")]
    [Trait("Category", "Integration")]
    public void BlocksEmbeddingAndFinalLayer_MatchUpstreamOnTheCpu()
    {
        using SafeTensorsLoader refLoader = new(), subsetLoader = new();
        refLoader.Load(Environment.GetEnvironmentVariable("HARTSY_XPART_DIT_REF")!);
        subsetLoader.Load(Environment.GetEnvironmentVariable("HARTSY_XPART_DIT_SUBSET")!);
        Dictionary<string, Tensor> fixture = refLoader.GetAllTensors(), weights = subsetLoader.GetAllTensors();
        using CpuBackend backend = new();
        XPartDitConfig config = new();

        foreach (int layer in new[] { 0, 1, 11, 15, 16 })
        {
            bool skip = layer > config.Depth / 2, moe = config.Depth - layer <= config.MoeBlocks, global = (layer + 1) % 2 == 0;
            Stopwatch watch = Stopwatch.StartNew();
            Hunyuan3DPlainBlock block = new(config.Width, config.Heads, skip, moe, config.Experts, config.TopK);
            block.LoadWeights(weights, $"blocks.{layer}", halfWeights: false);
            long loaded = watch.ElapsedMilliseconds;
            Tensor x = Copy(fixture[$"block.{layer}.x"]);
            using Tensor ctx1 = fixture[$"block.{layer}.ctx1"], ctx2 = fixture[$"block.{layer}.ctx2"];
            using Tensor? skipValue = skip ? fixture[$"block.{layer}.skip"] : null;
            watch.Restart();
            using Tensor result = block.ForwardParts(backend, x, ctx1, ctx2, skipValue, global);
            output.WriteLine($"block {layer} (global {global}, moe {moe}, skip {skip}): load {loaded} ms, forward {watch.ElapsedMilliseconds} ms");
            Compare($"block {layer}", Values(result), Values(fixture[$"block.{layer}.out"]));
            foreach (Tensor t in block.EnumerateWeights())
                t.Dispose();
        }

        using XPartDit dit = new(config);
        dit.LoadEdgeWeights(weights);
        Tensor latents = fixture["embed.latents"];
        int[] ids = Values(fixture["embed.ids"]).Select(v => (int)v).ToArray();
        using Tensor embedded = dit.Embed(backend, latents, Values(fixture["embed.sigma"])[0], ids);
        Compare("input embedding (time + part ids)", Values(embedded), Values(fixture["embed.out"]));
        using Tensor velocity = dit.Project(backend, fixture["final.x"]);
        Compare("final layer", Values(velocity), Values(fixture["final.out"]));
    }

    private static Tensor Copy(Tensor source)
    {
        Tensor copy = new(source.Shape, DType.F32);
        source.AsSpan<float>().CopyTo(copy.AsSpan<float>());
        return copy;
    }
}
