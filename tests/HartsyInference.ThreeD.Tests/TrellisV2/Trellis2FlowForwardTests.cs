using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

/// <summary>The tiny flows against an independent scalar transcription of the upstream forward.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class Trellis2FlowForwardTests
{
    private const double Tolerance = 2e-4;

    [Fact]
    public void DenseFlow_MatchesTheScalarReference()
    {
        Trellis2FlowConfig config = Trellis2TestData.Tiny(dense: true);
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(config, 11);
        using Trellis2FlowWeights weights = Trellis2FlowWeights.FromTensors(config, tensors);
        using Trellis2SparseStructureFlow flow = new(weights);
        Random rng = new(12);
        int r = config.Resolution, tokens = config.DenseTokens;
        using Tensor latent = Trellis2TestData.Random(rng, 1, config.InChannels, r, r, r);
        using Tensor cond = Trellis2TestData.Random(rng, 1, 5, config.CondChannels);
        using CpuBackend backend = new();

        using Tensor velocity = flow.Forward(backend, latent, 640f, cond);

        Assert.Equal(new TensorShape([1L, config.OutChannels, r, r, r]), velocity.Shape);
        float[] latentValues = Trellis2TestData.ToArray(latent);
        double[][] input = new double[tokens][];
        double[][] positions = new double[tokens][];
        for (int t = 0; t < tokens; t++)
        {
            input[t] = Enumerable.Range(0, config.InChannels).Select(c => (double)latentValues[c * tokens + t]).ToArray();
            positions[t] = [t / (r * r), t / r % r, t % r];
        }

        double[][] expected = new Trellis2Reference(config, tensors).Forward(input, positions, 640.0, ToRows(cond));

        float[] actual = Trellis2TestData.ToArray(velocity);
        for (int t = 0; t < tokens; t++)
        {
            for (int c = 0; c < config.OutChannels; c++)
                Assert.Equal(expected[t][c], actual[c * tokens + t], Tolerance);
        }

        Trellis2TestData.DisposeAll(tensors.Values);
    }

    [Fact]
    public void SparseFlow_MatchesTheScalarReferenceOnScatteredVoxels()
    {
        Trellis2FlowConfig config = Trellis2TestData.Tiny(dense: false);
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(config, 21);
        using Trellis2FlowWeights weights = Trellis2FlowWeights.FromTensors(config, tensors);
        using Trellis2SlatFlow flow = new(weights);
        Random rng = new(22);
        int[] coords = [0, 1, 2, 3, 0, 7, 0, 4, 0, 3, 3, 3, 0, 5, 6, 1, 0, 0, 7, 7];
        int n = coords.Length / 4;
        using Tensor x = Trellis2TestData.Random(rng, 1, n, config.InChannels);
        using Tensor cond = Trellis2TestData.Random(rng, 1, 6, config.CondChannels);
        using CpuBackend backend = new();

        using Tensor velocity = flow.Forward(backend, x, coords, 310f, cond);

        Assert.Equal(new TensorShape(1, n, config.OutChannels), velocity.Shape);
        double[][] input = ToRows(x);
        double[][] positions = Enumerable.Range(0, n).Select(i => new double[] { coords[i * 4 + 1], coords[i * 4 + 2], coords[i * 4 + 3] }).ToArray();
        double[][] expected = new Trellis2Reference(config, tensors).Forward(input, positions, 310.0, ToRows(cond));
        float[] actual = Trellis2TestData.ToArray(velocity);
        for (int t = 0; t < n; t++)
        {
            for (int c = 0; c < config.OutChannels; c++)
                Assert.Equal(expected[t][c], actual[t * config.OutChannels + c], Tolerance);
        }

        Trellis2TestData.DisposeAll(tensors.Values);
    }

    [Fact]
    public void ZeroedGatesAndCrossProjectionLeaveOnlyTheEmbeddingAndOutputLayers()
    {
        // With no modulation, a zero adaLN and a zero cross-attention output projection, every block adds gate · branch = 0.
        Trellis2FlowConfig config = Trellis2TestData.Tiny(dense: false);
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(config, 31);
        foreach (string key in tensors.Keys.Where(k => k.Contains("modulation", StringComparison.Ordinal)
            || k.Contains("cross_attn.to_out", StringComparison.Ordinal)))
            tensors[key].AsSpan<float>().Clear();
        using Trellis2FlowWeights weights = Trellis2FlowWeights.FromTensors(config, tensors);
        using Trellis2SlatFlow flow = new(weights);
        Random rng = new(32);
        int[] coords = [0, 1, 2, 3, 0, 4, 4, 4];
        using Tensor x = Trellis2TestData.Random(rng, 1, 2, config.InChannels);
        using Tensor cond = Trellis2TestData.Random(rng, 1, 3, config.CondChannels);
        using CpuBackend backend = new();

        using Tensor velocity = flow.Forward(backend, x, coords, 500f, cond);

        // The blocks leave h = input_layer(x), so the output is out_layer(layer_norm(h)): a block-free forward in closed form.
        double[][] rows = ToRows(x);
        float[] w = Trellis2TestData.ToArray(tensors["input_layer.weight"]), b = Trellis2TestData.ToArray(tensors["input_layer.bias"]);
        float[] ow = Trellis2TestData.ToArray(tensors["out_layer.weight"]), ob = Trellis2TestData.ToArray(tensors["out_layer.bias"]);
        float[] actual = Trellis2TestData.ToArray(velocity);
        for (int t = 0; t < 2; t++)
        {
            double[] h = new double[16];
            for (int i = 0; i < 16; i++)
                h[i] = b[i] + Enumerable.Range(0, 4).Sum(j => w[i * 4 + j] * rows[t][j]);
            double mean = h.Average(), variance = h.Select(v => (v - mean) * (v - mean)).Average();
            double[] normed = h.Select(v => (v - mean) / Math.Sqrt(variance + 1e-5)).ToArray();
            for (int c = 0; c < 4; c++)
            {
                double expected = ob[c] + Enumerable.Range(0, 16).Sum(j => ow[c * 16 + j] * normed[j]);
                Assert.Equal(expected, actual[t * 4 + c], Tolerance);
            }
        }

        Trellis2TestData.DisposeAll(tensors.Values);
    }

    [Fact]
    public void TimestepChangesTheVelocity()
    {
        Trellis2FlowConfig config = Trellis2TestData.Tiny(dense: false);
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(config, 41);
        using Trellis2FlowWeights weights = Trellis2FlowWeights.FromTensors(config, tensors);
        using Trellis2SlatFlow flow = new(weights);
        Random rng = new(42);
        int[] coords = [0, 1, 2, 3, 0, 4, 4, 4];
        using Tensor x = Trellis2TestData.Random(rng, 1, 2, config.InChannels);
        using Tensor cond = Trellis2TestData.Random(rng, 1, 3, config.CondChannels);
        using CpuBackend backend = new();

        using Tensor a = flow.Forward(backend, x, coords, 100f, cond);
        using Tensor b = flow.Forward(backend, x, coords, 900f, cond);

        Assert.NotEqual(Trellis2TestData.ToArray(a), Trellis2TestData.ToArray(b));
        Trellis2TestData.DisposeAll(tensors.Values);
    }

    [Fact]
    public void WrongInputShapesAreRejected()
    {
        Trellis2FlowConfig dense = Trellis2TestData.Tiny(dense: true);
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(dense, 51);
        using Trellis2FlowWeights weights = Trellis2FlowWeights.FromTensors(dense, tensors);
        using Trellis2SparseStructureFlow flow = new(weights);
        using Tensor wrongGrid = new(new TensorShape([1L, 4, 3, 3, 3]), DType.F32);
        using Tensor cond = new(new TensorShape(1, 2, dense.CondChannels), DType.F32);
        using CpuBackend backend = new();

        Assert.Throws<ArgumentException>(() => flow.Forward(backend, wrongGrid, 1f, cond));
        Trellis2TestData.DisposeAll(tensors.Values);
    }

    [Fact]
    public void FlowsRejectTheOtherArchitecturesWeights()
    {
        Trellis2FlowConfig sparse = Trellis2TestData.Tiny(dense: false);
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(sparse, 61);
        using Trellis2FlowWeights weights = Trellis2FlowWeights.FromTensors(sparse, tensors);

        Assert.Throws<ArgumentException>(() => new Trellis2SparseStructureFlow(weights));
        Trellis2TestData.DisposeAll(tensors.Values);
    }

    private static double[][] ToRows(Tensor t)
    {
        int width = (int)t.Shape[t.Shape.Rank - 1];
        float[] values = Trellis2TestData.ToArray(t);
        return Enumerable.Range(0, values.Length / width)
            .Select(r => Enumerable.Range(0, width).Select(c => (double)values[r * width + c]).ToArray()).ToArray();
    }
}
