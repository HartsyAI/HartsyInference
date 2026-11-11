using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Pixal3D;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Pixal3D;

/// <summary>NAF against upstream's own modules (<c>tests/python-reference/dump_pixal3d_naf.py</c>: the real <c>naf_release.pth</c>, a brute-force
/// transcription of NATTEN's dilated window in place of <c>natten</c>, torch 2.6 float32) on the real weights, and the checkpoint's key check on synthetic ones.
/// The real-weight tests are Integration lane and skip when <c>naf_release.pth</c> is absent.</summary>
public sealed class Pixal3DNafTests
{
    private const string NafFile = "naf_release.pth";

    private readonly ITestOutputHelper _output;

    public Pixal3DNafTests(ITestOutputHelper output) => _output = output;

    private static JsonDocument Fixture()
        => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Pixal3D", "Fixtures", "naf_reference.json")));

    private static float[] Floats(JsonElement array) => array.EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();

    [Trait("Category", "Integration")]
    [Pixal3DRealWeightsFact(NafFile)]
    public void QueryMapAndUpsampledFeatures_MatchUpstreamOnTheRealWeights()
    {
        using Pixal3DNaf naf = Pixal3DNaf.Load(Path.Combine(Pixal3DRealWeightsFactAttribute.FindFolder()!, NafFile));
        using CpuBackend backend = new();
        using JsonDocument fixture = Fixture();
        foreach (JsonElement c in fixture.RootElement.GetProperty("cases").EnumerateArray())
        {
            int size = c.GetProperty("image_size").GetInt32(), output = c.GetProperty("output_size").GetInt32();
            int feature = c.GetProperty("feature_size").GetInt32(), channels = c.GetProperty("channels").GetInt32();
            using Tensor image = new(new TensorShape(1, 3, size, size), DType.F32);
            Floats(c.GetProperty("image")).CopyTo(image.AsSpan<float>());

            using Tensor query = naf.QueryMap(backend, image, output);

            Assert.Equal(new TensorShape(1, Pixal3DNaf.Dim, output, output), query.Shape);
            int[] probe = c.GetProperty("query_channels").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            float[] expectedProbe = Floats(c.GetProperty("query_probe"));
            ReadOnlySpan<float> map = query.AsReadOnlySpan<float>();
            int plane = output * output;
            double worstQuery = 0;
            for (int p = 0; p < probe.Length; p++)
            {
                for (int i = 0; i < plane; i++)
                    worstQuery = Math.Max(worstQuery, Math.Abs(map[probe[p] * plane + i] - expectedProbe[p * plane + i]));
            }

            // Every pixel centre, so the bilinear blend returns the pixel itself.
            float[] uv = new float[plane * 2];
            for (int y = 0; y < output; y++)
            {
                for (int x = 0; x < output; x++)
                {
                    uv[(y * output + x) * 2] = (x + 0.5f) / output * 2f - 1f;
                    uv[(y * output + x) * 2 + 1] = (y + 0.5f) / output * 2f - 1f;
                }
            }

            using Tensor sampled = Pixal3DNaf.SampleUpsampled(map, output, output, Floats(c.GetProperty("feature_rows")), feature, feature, channels, uv);

            float[] expected = Floats(c.GetProperty("expected"));
            ReadOnlySpan<float> actual = sampled.AsReadOnlySpan<float>();
            double worst = 0;
            for (int ch = 0; ch < channels; ch++)
            {
                for (int i = 0; i < plane; i++)
                    worst = Math.Max(worst, Math.Abs(actual[i * channels + ch] - expected[ch * plane + i]));
            }

            _output.WriteLine($"S={size} out={output} features={feature}x{feature}x{channels}: query probe worst {worstQuery:E2}, upsampled worst {worst:E2}.");
            Assert.True(worstQuery < 2e-4, $"query map differs from upstream by {worstQuery:E2}");
            Assert.True(worst < 2e-4, $"upsampled features differ from upstream by {worst:E2}");
        }
    }

    [Fact]
    public void SyntheticWeights_AreCheckedByNameAndShape()
    {
        Dictionary<string, Tensor> good = Weights();
        using (Pixal3DNaf naf = Pixal3DNaf.FromTensors(good))
            Assert.Equal(37, naf.EnumerateWeights().Count());
        Dispose(good);

        Dictionary<string, Tensor> missing = Weights();
        missing["image_encoder.encoder.0.weight"].Dispose();
        missing.Remove("image_encoder.encoder.0.weight");
        Assert.Throws<HartsyInferenceException>(() => Pixal3DNaf.FromTensors(missing));
        Dispose(missing);

        Dictionary<string, Tensor> wrongShape = Weights();
        wrongShape["image_encoder.sem_encoder.1.conv1.weight"].Dispose();
        wrongShape["image_encoder.sem_encoder.1.conv1.weight"] = new Tensor(new TensorShape(128, 128, 1, 1), DType.F32);
        Assert.Throws<HartsyInferenceException>(() => Pixal3DNaf.FromTensors(wrongShape));
        Dispose(wrongShape);

        Dictionary<string, Tensor> extra = Weights();
        extra["image_encoder.shortcut.weight"] = new Tensor(new TensorShape(1), DType.F32);
        Assert.Throws<HartsyInferenceException>(() => Pixal3DNaf.FromTensors(extra));
        Dispose(extra);
    }

    [Fact]
    public void SampleUpsampled_RejectsUnusableGeometry()
    {
        float[] map = new float[Pixal3DNaf.Dim * 16 * 16];
        float[] rows = new float[8 * 8 * 4];
        float[] uv = [0f, 0f];

        // 8 feature cells per axis is below the 9-cell window; 16 is not divisible by 5.
        Assert.Throws<ArgumentException>(() => Pixal3DNaf.SampleUpsampled(map, 16, 16, rows, 8, 8, 4, uv));
        Assert.Throws<ArgumentException>(() => Pixal3DNaf.SampleUpsampled(map, 16, 16, new float[5 * 5 * 4], 5, 5, 4, uv));
    }

    [Fact]
    public void SampleUpsampled_OfConstantValuesReturnsThemWhateverTheQueries()
    {
        // Softmax weights sum to one, so identical value rows come back unchanged for any query and key.
        const int h = 18, w = 18, f = 9, channels = 4;
        Random rng = new(5);
        float[] map = Enumerable.Range(0, Pixal3DNaf.Dim * h * w).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        float[] rows = new float[f * f * channels];
        for (int i = 0; i < f * f; i++)
        {
            for (int c = 0; c < channels; c++)
                rows[i * channels + c] = c + 1.5f;
        }

        using Tensor sampled = Pixal3DNaf.SampleUpsampled(map, h, w, rows, f, f, channels, [-0.9f, 0.3f, 0.2f, -0.7f, 1.5f, 1.5f]);

        Assert.Equal(new TensorShape(1, 3, channels), sampled.Shape);
        ReadOnlySpan<float> v = sampled.AsReadOnlySpan<float>();
        for (int cell = 0; cell < 3; cell++)
        {
            for (int c = 0; c < channels; c++)
                Assert.Equal(c + 1.5f, v[cell * channels + c], 1e-4);
        }
    }

    private static Dictionary<string, Tensor> Weights()
    {
        Dictionary<string, Tensor> weights = [];
        weights["image_encoder.rope.periods"] = new Tensor(new TensorShape(16), DType.F32);
        foreach ((string name, int k) in new[] { ("encoder", 1), ("sem_encoder", 3) })
        {
            string p = $"image_encoder.{name}.";
            weights[p + "0.weight"] = new Tensor(new TensorShape(128, 3, k, k), DType.F32);
            weights[p + "0.bias"] = new Tensor(new TensorShape(128), DType.F32);
            for (int b = 1; b <= 2; b++)
            {
                foreach (string conv in new[] { "conv1", "conv2" })
                {
                    weights[$"{p}{b}.{conv}.weight"] = new Tensor(new TensorShape(128, 128, k, k), DType.F32);
                    weights[$"{p}{b}.{conv}.bias"] = new Tensor(new TensorShape(128), DType.F32);
                }

                foreach (string norm in new[] { "norm1", "norm2" })
                {
                    weights[$"{p}{b}.{norm}.weight"] = new Tensor(new TensorShape(128), DType.F32);
                    weights[$"{p}{b}.{norm}.bias"] = new Tensor(new TensorShape(128), DType.F32);
                }
            }
        }

        return weights;
    }

    private static void Dispose(Dictionary<string, Tensor> tensors)
    {
        foreach (Tensor t in tensors.Values)
            t.Dispose();
    }
}
