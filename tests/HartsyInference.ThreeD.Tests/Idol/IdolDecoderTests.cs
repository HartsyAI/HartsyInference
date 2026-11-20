using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.Idol;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Idol;

public sealed class IdolDecoderTests
{
    private static readonly IBackend Backend = new CpuBackend();

    [Fact]
    public void DecoderConfig_ReadsTheReleasedValuesFromConfigJson()
    {
        IdolDecoderConfig c = IdolDecoderConfig.FromJson("""
            {"decoder": {"args": {"model_channels": 1024, "cond_channels": 16, "num_blocks": 4, "num_heads": 16, "mlp_ratio": 4}},
             "decoder_gs": {"args": {"num_blocks": 16, "representation_config": {"num_gaussians": 32,
               "lr": {"_xyz": 1.0, "_features_dc": 1.0, "_opacity": 1.0, "_scaling": 1.0, "_rotation": 0.1},
               "perturbe_size": 1.5, "3d_filter_kernel_size": 0.0009, "scaling_bias": 0.004, "opacity_bias": 0.1}}},
             "max_voxel_level": 8}
            """);

        Assert.Equal(new IdolDecoderConfig(), c);
        Assert.Equal(480, c.GaussianOutputChannels);
    }

    [Fact]
    public void AnchorFrequencies_AreTwoToALinspaceFromZeroToTen()
    {
        float[] f = IdolAnchorEmbedding.Frequencies(170);

        Assert.Equal(1f, f[0]);
        Assert.Equal(1024f, f[169], 2);
        for (int i = 1; i < f.Length; i++)
            Assert.True(f[i] > f[i - 1]);
        Assert.Equal(Math.Pow(2.0, 10.0 * 85 / 169), f[85], 1);
    }

    [Fact]
    public void SystematicSample_ConservesTheCountAndFollowsTheProbabilities()
    {
        Random random = new(5);
        float[] probs = [0.5f, 0.25f, 0.25f, 0f, 0f, 0f, 0f, 0f];
        int[] counts = new int[8];

        IdolOctreeSampler.SystematicSample(probs, 100, random, counts);

        Assert.Equal(100, counts.Sum());
        Assert.InRange(counts[0], 49, 51);
        Assert.InRange(counts[1], 24, 26);
        Assert.InRange(counts[2], 24, 26);
        Assert.All(counts[3..], c => Assert.Equal(0, c));
    }

    [Fact]
    public void Sampler_ConservesPointsKeepsThemInsideTheirLeafCellsAndFollowsTheLogits()
    {
        // Every cell sends all its points to child 0 (the minimum corner): after four levels each point sits in [0, 1/16)^3 around the origin.
        float[] LogitsToChildZero(ReadOnlySpan<float> centres, int resolution)
        {
            float[] logits = new float[centres.Length / 3 * 8];
            for (int p = 0; p < centres.Length / 3; p++)
                logits[p * 8] = 50f;
            return logits;
        }

        float[] points = IdolOctreeSampler.Sample(LogitsToChildZero, 64, 4, new Random(1));

        Assert.Equal(64 * 3, points.Length);
        Assert.All(points, v => Assert.InRange(v, 0f, 1f / 16f));

        // Uniform logits spread them over the whole box.
        float[] uniform = IdolOctreeSampler.Sample((c, r) => new float[c.Length / 3 * 8], 4096, 3, new Random(2));
        Assert.All(uniform, v => Assert.InRange(v, 0f, 1f));
        Assert.InRange(uniform.Where((_, i) => i % 3 == 0).Average(), 0.4f, 0.6f);
    }

    [Fact]
    public void Sampler_IsReproducibleBySeed()
    {
        float[] a = IdolOctreeSampler.Sample((c, r) => new float[c.Length / 3 * 8], 200, 3, new Random(9));
        float[] b = IdolOctreeSampler.Sample((c, r) => new float[c.Length / 3 * 8], 200, 3, new Random(9));
        float[] c2 = IdolOctreeSampler.Sample((c, r) => new float[c.Length / 3 * 8], 200, 3, new Random(10));

        Assert.Equal(a, b);
        Assert.NotEqual(a, c2);
    }

    [Fact]
    public void TinyDecoders_RunAndTheCloudHasTheDocumentedLayout()
    {
        IdolDecoderConfig tiny = TinyConfig();
        Dictionary<string, Tensor> tensors = TinyTensors(tiny);
        try
        {
            using IdolDecoderWeights weights = IdolDecoderWeights.FromTensors(tiny, tensors);
            using IdolOctreeDecoder octree = new(weights);
            using IdolGaussianDecoder gaussian = new(weights);
            using Tensor latent = Random(new TensorShape(1, 8, tiny.LatentChannels), 31, 1f);

            float[] centres = [0.25f, 0.25f, 0.25f, 0.75f, 0.25f, 0.75f];
            float[] logits = octree.Logits(Backend, centres, 2, latent);
            Assert.Equal(16, logits.Length);
            Assert.All(logits, v => Assert.True(float.IsFinite(v)));
            Assert.NotEqual(logits[..8], logits[8..]);

            float[] anchors = [0.1f, 0.2f, 0.3f, 0.6f, 0.7f, 0.8f, 0.5f, 0.5f, 0.5f];
            using Tensor features = gaussian.Features(Backend, anchors, latent);
            Assert.Equal(new TensorShape(1, 3, tiny.GaussianOutputChannels), features.Shape);

            GaussianSplatCloud cloud = gaussian.ToCloud(anchors, features);
            Assert.Equal(3 * tiny.GaussiansPerAnchor, cloud.Count);
            Assert.Equal(1, cloud.ShCoeffsPerSplat);
            for (int s = 0; s < cloud.Count; s++)
            {
                double norm = 0;
                for (int c = 0; c < 4; c++) norm += cloud.Rotations[s * 4 + c] * (double)cloud.Rotations[s * 4 + c];
                Assert.Equal(1.0, norm, 4);
                Assert.True(float.IsFinite(cloud.Opacities[s]) && Math.Abs(cloud.Opacities[s]) <= 13.82f);
            }
        }
        finally
        {
            foreach (Tensor t in tensors.Values) t.Dispose();
        }
    }

    [Fact]
    public void ToCloud_AndTheDecodersRefuseMalformedPointsAndFeatures()
    {
        IdolDecoderConfig tiny = TinyConfig();
        Dictionary<string, Tensor> tensors = TinyTensors(tiny);
        try
        {
            using IdolDecoderWeights weights = IdolDecoderWeights.FromTensors(tiny, tensors);
            using IdolGaussianDecoder gaussian = new(weights);
            using Tensor oneRow = new(new TensorShape(1, 1, tiny.GaussianOutputChannels), DType.F32);
            using Tensor twoRows = new(new TensorShape(1, 2, tiny.GaussianOutputChannels), DType.F32);
            using Tensor half = new(new TensorShape(1, 1, tiny.GaussianOutputChannels), DType.F16);
            float[] one = [0.25f, 0.5f, 0.75f];

            Assert.Throws<ArgumentException>(() => gaussian.ToCloud(one, twoRows));   // more rows than anchors
            Assert.Throws<ArgumentException>(() => gaussian.ToCloud([0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f], oneRow));   // fewer rows than anchors
            Assert.Throws<ArgumentException>(() => gaussian.ToCloud(one, half));
            Assert.Throws<ArgumentException>(() => gaussian.ToCloud([0.25f, 0.5f], oneRow));
            Assert.Throws<ArgumentException>(() => gaussian.ToCloud([], oneRow));
            Assert.Throws<ArgumentException>(() => IdolAnchorEmbedding.Build([0.25f, 0.5f, 0.75f, 0.1f], 12));
        }
        finally
        {
            foreach (Tensor t in tensors.Values) t.Dispose();
        }
    }

    [Fact]
    public void Config_RefusesAGaussianDecoderWhoseWidthDiffersFromTheOctreesOne()
    {
        const string json = "{\"decoder\":{\"args\":{\"model_channels\":1024,\"num_heads\":16,\"mlp_ratio\":4,\"cond_channels\":16,\"num_blocks\":4}},"
            + "\"decoder_gs\":{\"args\":{\"model_channels\":512,\"num_heads\":16,\"mlp_ratio\":4,\"num_blocks\":16}}}";

        Assert.Throws<FormatException>(() => IdolDecoderConfig.FromJson(json));
    }

    [Fact]
    public void Cloud_ForZeroFeaturesIsTheInitialSplat()
    {
        IdolDecoderConfig tiny = TinyConfig();
        Dictionary<string, Tensor> tensors = TinyTensors(tiny);
        try
        {
            using IdolDecoderWeights weights = IdolDecoderWeights.FromTensors(tiny, tensors);
            // Cleared before the decoder is made, so the test does not depend on ToCloud reading the weights at call time.
            tensors["gs.points_offset_perturbation"].AsSpan<float>().Clear();
            tensors["gs.base_offset_scale"].AsSpan<float>().Clear();
            using IdolGaussianDecoder gaussian = new(weights);
            using Tensor features = new(new TensorShape(1, 1, tiny.GaussianOutputChannels), DType.F32);

            GaussianSplatCloud cloud = gaussian.ToCloud([0.25f, 0.5f, 0.75f], features);

            for (int s = 0; s < cloud.Count; s++)
            {
                // A zero output puts every Gaussian at its anchor (shifted by the box origin) and turns the identity rotation to (sqrt 1/2, sqrt 1/2, 0, 0).
                Assert.Equal(0.25f - 0.5f, cloud.Positions[s * 3], 5);
                Assert.Equal(-(0.75f - 0.5f), cloud.Positions[s * 3 + 1], 5);
                Assert.Equal(0.5f - 0.5f, cloud.Positions[s * 3 + 2], 5);
                Assert.Equal(Math.Sqrt(0.5), cloud.Rotations[s * 4], 5);
                Assert.Equal(Math.Sqrt(0.5), cloud.Rotations[s * 4 + 1], 5);
                Assert.Equal(Math.Log(0.1 / 0.9), cloud.Opacities[s], 4);
                Assert.Equal(0f, cloud.ShCoefficients[s * 3]);
            }
        }
        finally
        {
            foreach (Tensor t in tensors.Values) t.Dispose();
        }
    }

    /// <summary>Two heads of 8, one block each, two Gaussians per anchor.</summary>
    private static IdolDecoderConfig TinyConfig() => new()
    {
        ModelChannels = 16, LatentChannels = 4, Heads = 2, MlpRatio = 2, OctreeBlocks = 1, GaussianBlocks = 1, GaussiansPerAnchor = 2,
    };

    private static Dictionary<string, Tensor> TinyTensors(IdolDecoderConfig c)
    {
        Dictionary<string, Tensor> t = new(StringComparer.Ordinal);
        int seed = 100;
        foreach ((string key, long[] shape) in IdolDecoderWeights.ExpectedShapes(c))
        {
            bool gain = key.EndsWith(".gamma", StringComparison.Ordinal) || key.EndsWith("norm2.weight", StringComparison.Ordinal);
            Tensor w = Random(new TensorShape(shape), seed++, gain ? 0.1f : 0.2f);
            if (gain)
            {
                Span<float> span = w.AsSpan<float>();
                for (int i = 0; i < span.Length; i++) span[i] += 1f;
            }

            t[key] = w;
        }

        return t;
    }

    private static Tensor Random(TensorShape shape, int seed, float scale)
    {
        Tensor t = new(shape, DType.F32);
        Random rng = new(seed);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return t;
    }
}
