using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Kimi;
using HartsyInference.LLM.Kimi.Vision;
using Xunit;

namespace HartsyInference.LLM.Tests.Kimi.Vision;

/// <summary>The MoonViT-V2 encoder end to end on a synthetic checkpoint read through the shard index: token count, finiteness, determinism and the
/// size refusals.</summary>
public sealed class KimiVisionEncoderTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "hartsy-kimi-vision-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }

    [Theory]
    [InlineData(2, 8, 2)]
    [InlineData(2, 16, 4)]
    [InlineData(0, 8, 2)]
    public void Encode_EmitsOneTokenPerMergeBlockThatIsFiniteAndDeterministic(int layers, int side, int tokensPerSide)
    {
        KimiVisionConfig config = KimiVisionSyntheticCheckpoint.Tiny(layers);
        using MoonViTEncoder encoder = OpenEncoder(config, seed: 21);
        using Tensor image = RandomImage(side, seed: 22);
        using IBackend backend = new CpuBackend();

        using Tensor first = encoder.Encode(backend, image);
        using Tensor second = encoder.Encode(backend, image);

        Assert.Equal(1L, first.Shape[0]);
        Assert.Equal((long)tokensPerSide * tokensPerSide, first.Shape[1]);
        Assert.Equal((long)config.TextHidden, first.Shape[2]);
        float[] a = first.AsReadOnlySpan<float>().ToArray();
        Assert.All(a, v => Assert.True(float.IsFinite(v)));
        Assert.Equal(a, second.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void Encode_ReportsTheTokenCountForThePreferredSize()
    {
        KimiVisionConfig config = KimiVisionSyntheticCheckpoint.Tiny(layers: 1);
        using MoonViTEncoder encoder = OpenEncoder(config, seed: 23);

        // Image 8 with patch 2 and merge 2 is a 2x2 grid of merge blocks.
        Assert.Equal(4, encoder.TokensPerImage);
        Assert.Equal(8, encoder.ImageSize);
    }

    [Fact]
    public void Encode_DependsOnTheImage()
    {
        KimiVisionConfig config = KimiVisionSyntheticCheckpoint.Tiny(layers: 2);
        using MoonViTEncoder encoder = OpenEncoder(config, seed: 24);
        using IBackend backend = new CpuBackend();
        using Tensor a = RandomImage(8, seed: 25);
        using Tensor b = RandomImage(8, seed: 26);

        using Tensor outA = encoder.Encode(backend, a);
        using Tensor outB = encoder.Encode(backend, b);

        Assert.NotEqual(outA.AsReadOnlySpan<float>().ToArray(), outB.AsReadOnlySpan<float>().ToArray());
    }

    [Theory]
    [InlineData(10, 8)]
    [InlineData(8, 6)]
    public void Encode_RefusesSidesThatAreNotMergeBlocks(int height, int width)
    {
        KimiVisionConfig config = KimiVisionSyntheticCheckpoint.Tiny(layers: 1);
        using MoonViTEncoder encoder = OpenEncoder(config, seed: 27);
        using IBackend backend = new CpuBackend();
        using Tensor image = Make(new float[3 * height * width], 1, 3, height, width);

        Assert.Throws<ArgumentException>(() => encoder.Encode(backend, image));
    }

    [Fact]
    public void Constructor_RefusesWeightsWithTheWrongBlockCount()
    {
        KimiVisionConfig twoBlocks = KimiVisionSyntheticCheckpoint.Tiny(layers: 2);
        KimiVisionConfig oneBlock = KimiVisionSyntheticCheckpoint.Tiny(layers: 1);
        KimiVisionSyntheticCheckpoint.Write(_dir, twoBlocks, KimiVisionSyntheticCheckpoint.Values(twoBlocks, seed: 28));
        using KimiK31CheckpointWeights source = KimiK31CheckpointWeights.Open(_dir);
        KimiVisionWeights weights = KimiVisionWeightLoader.Load(twoBlocks, source);

        Assert.Throws<ArgumentException>(() => new MoonViTEncoder(oneBlock, weights));
        weights.Dispose();
    }

    private MoonViTEncoder OpenEncoder(KimiVisionConfig config, int seed)
    {
        KimiVisionSyntheticCheckpoint.Write(_dir, config, KimiVisionSyntheticCheckpoint.Values(config, seed));
        using KimiK31CheckpointWeights source = KimiK31CheckpointWeights.Open(_dir);
        KimiVisionWeights weights = KimiVisionWeightLoader.Load(config, source);
        return new MoonViTEncoder(config, weights);
    }

    private static Tensor RandomImage(int side, int seed)
    {
        Random rng = new(seed);
        float[] pixels = new float[3 * side * side];
        for (int i = 0; i < pixels.Length; i++) pixels[i] = (float)(rng.NextDouble() * 2 - 1);
        return Make(pixels, 1, 3, side, side);
    }

    private static Tensor Make(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}
