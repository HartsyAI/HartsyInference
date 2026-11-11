using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.TrellisV2;
using HartsyInference.ThreeD.Pipelines.Stages;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

/// <summary>The texture decoder (guided by the shape decoder's subdivisions), the texture flow and the latent statistics, on tiny
/// synthetic networks. The guided decoder is checked against the predicting one: with the same weights and the subdivisions it chose,
/// it must reproduce its head exactly.</summary>
public sealed class Trellis2TextureNetworkTests
{
    private const int LatentRes = 4, FineRes = 16;

    private static ShapeVaeDecoderConfig TextureConfig() => ShapeVaeTestData.Decoder() with
    {
        OutputChannels = ShapeVaeKeyMap.TextureOutputChannels,
        PredictSubdivision = false,
    };

    private static (SparseTensor Latent, Tensor Feats) Latent(int seed, int n = 10)
    {
        Random rng = new(seed);
        int[] coords = ShapeVaeTestData.RandomCoords(rng, n, LatentRes);
        Tensor feats = ShapeVaeTestData.RandomFeats(rng, n, 4, -1, 1);
        return (ShapeVaeTestData.Sparse(coords, LatentRes, feats), feats);
    }

    [Fact]
    public void ShippedTextureConfig_HasNoSubdivisionHeadsAndASixChannelHead()
    {
        ShapeVaeDecoderConfig texture = ShapeVaeDecoderConfig.Texture;
        IReadOnlyList<ShapeVaeKeyMap.TensorSpec> specs = ShapeVaeKeyMap.Decoder(texture);

        Assert.DoesNotContain(specs, s => s.CheckpointKey.Contains("to_subdiv", StringComparison.Ordinal));
        // The shape decoder's 4 + 8·32 + 8·4 tensors minus the weight and bias of its four subdivision heads.
        Assert.Equal(4 + 8 * 32 + 8 * 4 - 4 * 2, specs.Count);
        Assert.Equal(new long[] { 6, 64 }, specs.Single(s => s.CheckpointKey == "output_layer.weight").Shape);
        Assert.Equal(new long[] { 6 }, specs.Single(s => s.CheckpointKey == "output_layer.bias").Shape);
        Assert.Equal(ShapeVaeConfig.NextDcF16C32.Decoder.ModelChannels, texture.ModelChannels);
        Assert.Equal(ShapeVaeConfig.NextDcF16C32.Decoder.NumBlocks, texture.NumBlocks);
        texture.Validate();
    }

    [Fact]
    public void GuidedDecoder_WithThePredictingWeightsReproducesItsHeadExactly()
    {
        ShapeVaeDecoderConfig predicting = ShapeVaeTestData.Decoder();
        Dictionary<string, Tensor> weights = ShapeVaeTestData.RandomWeights(ShapeVaeKeyMap.Decoder(predicting), seed: 31);
        (SparseTensor latent, Tensor feats) = Latent(32);
        using CpuBackend backend = new();
        using ShapeVaeDecoder shape = new(predicting, weights);
        using ShapeVaeDecoder guided = new(predicting with { PredictSubdivision = false }, weights);
        using ShapeVaeSubdivisions record = new();

        SparseTensor expected = shape.Decode(backend, latent, record);
        SparseTensor actual = guided.DecodeGuided(backend, latent, record);

        Assert.Equal(2, record.Count);
        Assert.Equal(expected.Resolution, actual.Resolution);
        Assert.Equal(expected.Coords, actual.Coords);
        Assert.Equal(Trellis2TestData.ToArray(expected.Feats), Trellis2TestData.ToArray(actual.Feats));
        feats.Dispose();
        expected.Feats.Dispose();
        actual.Feats.Dispose();
        Trellis2TestData.DisposeAll(weights.Values);
    }

    [Fact]
    public void TextureDecoder_FollowsTheShapeVoxelsAndReturnsSixFiniteChannels()
    {
        ShapeVaeDecoderConfig predicting = ShapeVaeTestData.Decoder();
        Dictionary<string, Tensor> shapeWeights = ShapeVaeTestData.RandomWeights(ShapeVaeKeyMap.Decoder(predicting), seed: 41);
        Dictionary<string, Tensor> textureWeights = ShapeVaeTestData.RandomWeights(ShapeVaeKeyMap.Decoder(TextureConfig()), seed: 42);
        (SparseTensor latent, Tensor feats) = Latent(43);
        using CpuBackend backend = new();
        using ShapeVaeDecoder shape = new(predicting, shapeWeights);
        using ShapeVaeDecoder texture = new(TextureConfig(), textureWeights);
        using ShapeVaeSubdivisions record = new();

        SparseTensor head = shape.Decode(backend, latent, record);
        SparseTensor attributes = texture.DecodeGuided(backend, latent, record);

        Assert.Equal(head.Coords, attributes.Coords);
        Assert.Equal(FineRes, attributes.Resolution);
        Assert.Equal(ShapeVaeKeyMap.TextureOutputChannels, attributes.Channels);
        Assert.All(Trellis2TestData.ToArray(attributes.Feats), v => Assert.True(float.IsFinite(v)));
        feats.Dispose();
        head.Feats.Dispose();
        attributes.Feats.Dispose();
        Trellis2TestData.DisposeAll(shapeWeights.Values);
        Trellis2TestData.DisposeAll(textureWeights.Values);
    }

    [Fact]
    public void GuidedDecoder_RejectsAGuideFromOtherVoxels()
    {
        ShapeVaeDecoderConfig predicting = ShapeVaeTestData.Decoder();
        Dictionary<string, Tensor> shapeWeights = ShapeVaeTestData.RandomWeights(ShapeVaeKeyMap.Decoder(predicting), seed: 51);
        Dictionary<string, Tensor> textureWeights = ShapeVaeTestData.RandomWeights(ShapeVaeKeyMap.Decoder(TextureConfig()), seed: 52);
        (SparseTensor shapeLatent, Tensor shapeFeats) = Latent(53, n: 10);
        (SparseTensor otherLatent, Tensor otherFeats) = Latent(54, n: 7);
        using CpuBackend backend = new();
        using ShapeVaeDecoder shape = new(predicting, shapeWeights);
        using ShapeVaeDecoder texture = new(TextureConfig(), textureWeights);
        using ShapeVaeSubdivisions record = new();
        shape.Decode(backend, shapeLatent, record).Feats.Dispose();

        ArgumentException error = Assert.Throws<ArgumentException>(() => texture.DecodeGuided(backend, otherLatent, record));

        Assert.Contains("same coordinates", error.Message, StringComparison.Ordinal);
        shapeFeats.Dispose();
        otherFeats.Dispose();
        Trellis2TestData.DisposeAll(shapeWeights.Values);
        Trellis2TestData.DisposeAll(textureWeights.Values);
    }

    [Fact]
    public void GuidedDecoder_RejectsAGuideRecordedOverOtherVoxelsWithTheSameCounts()
    {
        ShapeVaeDecoderConfig predicting = ShapeVaeTestData.Decoder();
        Dictionary<string, Tensor> shapeWeights = ShapeVaeTestData.RandomWeights(ShapeVaeKeyMap.Decoder(predicting), seed: 55);
        Dictionary<string, Tensor> textureWeights = ShapeVaeTestData.RandomWeights(ShapeVaeKeyMap.Decoder(TextureConfig()), seed: 56);
        (SparseTensor latent, Tensor feats) = Latent(57);
        // The same voxels shifted by one along x: equal count at every level, different coordinates.
        int[] shifted = (int[])latent.Coords.Clone();
        for (int i = 1; i < shifted.Length; i += 4)
            shifted[i] = (shifted[i] + 1) % LatentRes;
        SparseTensor other = ShapeVaeTestData.Sparse(shifted, LatentRes, feats);
        using CpuBackend backend = new();
        using ShapeVaeDecoder shape = new(predicting, shapeWeights);
        using ShapeVaeDecoder texture = new(TextureConfig(), textureWeights);
        using ShapeVaeSubdivisions record = new();
        shape.Decode(backend, latent, record).Feats.Dispose();

        Assert.Throws<ArgumentException>(() => texture.DecodeGuided(backend, other, record));
        texture.DecodeGuided(backend, latent, record).Feats.Dispose();
        feats.Dispose();
        Trellis2TestData.DisposeAll(shapeWeights.Values);
        Trellis2TestData.DisposeAll(textureWeights.Values);
    }

    [Fact]
    public void Decode_RefusesARecordThatAlreadyHoldsLevels()
    {
        Dictionary<string, Tensor> weights = ShapeVaeTestData.RandomWeights(ShapeVaeKeyMap.Decoder(ShapeVaeTestData.Decoder()), seed: 58);
        (SparseTensor latent, Tensor feats) = Latent(59);
        using CpuBackend backend = new();
        using ShapeVaeDecoder shape = new(ShapeVaeTestData.Decoder(), weights);
        using ShapeVaeSubdivisions record = new();
        shape.Decode(backend, latent, record).Feats.Dispose();

        Assert.Throws<ArgumentException>(() => shape.Decode(backend, latent, record));
        feats.Dispose();
        Trellis2TestData.DisposeAll(weights.Values);
    }

    [Fact]
    public void GuidedDecoder_NeedsEveryLevelOfTheGuide()
    {
        Dictionary<string, Tensor> weights = ShapeVaeTestData.RandomWeights(ShapeVaeKeyMap.Decoder(TextureConfig()), seed: 61);
        (SparseTensor latent, Tensor feats) = Latent(62);
        using CpuBackend backend = new();
        using ShapeVaeDecoder texture = new(TextureConfig(), weights);
        using ShapeVaeSubdivisions empty = new();

        Assert.Throws<ArgumentException>(() => texture.DecodeGuided(backend, latent, empty));
        feats.Dispose();
        Trellis2TestData.DisposeAll(weights.Values);
    }

    [Fact]
    public void Decoders_RefuseTheOtherDecodersEntryPoint()
    {
        Dictionary<string, Tensor> shapeWeights = ShapeVaeTestData.RandomWeights(ShapeVaeKeyMap.Decoder(ShapeVaeTestData.Decoder()), seed: 71);
        Dictionary<string, Tensor> textureWeights = ShapeVaeTestData.RandomWeights(ShapeVaeKeyMap.Decoder(TextureConfig()), seed: 72);
        (SparseTensor latent, Tensor feats) = Latent(73);
        using CpuBackend backend = new();
        using ShapeVaeDecoder shape = new(ShapeVaeTestData.Decoder(), shapeWeights);
        using ShapeVaeDecoder texture = new(TextureConfig(), textureWeights);
        using ShapeVaeSubdivisions record = new();

        Assert.Throws<InvalidOperationException>(() => texture.Decode(backend, latent));
        Assert.Throws<InvalidOperationException>(() => texture.Upsample(backend, latent, 1));
        Assert.Throws<InvalidOperationException>(() => shape.DecodeGuided(backend, latent, record));
        feats.Dispose();
        Trellis2TestData.DisposeAll(shapeWeights.Values);
        Trellis2TestData.DisposeAll(textureWeights.Values);
    }

    [Fact]
    public void TextureCheckpointKeys_AreRejectedByTheShapeDecoderAndTheOtherWayRound()
    {
        // A shape checkpoint has the 7-channel head; a texture checkpoint lacks the subdivision heads.
        Dictionary<string, Tensor> shapeWeights = ShapeVaeTestData.RandomWeights(ShapeVaeKeyMap.Decoder(ShapeVaeTestData.Decoder()), seed: 81);
        Dictionary<string, Tensor> textureWeights = ShapeVaeTestData.RandomWeights(ShapeVaeKeyMap.Decoder(TextureConfig()), seed: 82);

        Assert.ThrowsAny<Exception>(() => new ShapeVaeDecoder(TextureConfig(), shapeWeights));
        Assert.ThrowsAny<Exception>(() => new ShapeVaeDecoder(ShapeVaeTestData.Decoder(), textureWeights));
        Trellis2TestData.DisposeAll(shapeWeights.Values);
        Trellis2TestData.DisposeAll(textureWeights.Values);
    }

    [Fact]
    public void Subdivisions_ReportMissingLevelsAndReleaseTheirTensors()
    {
        ShapeVaeSubdivisions record = new();
        Tensor level = new(new TensorShape(8), DType.F32);
        record.Add(level, [0, 1, 2, 3]);

        Assert.Same(level, record.Level(0));
        Assert.True(record.Matches(0, [0, 1, 2, 3]));
        Assert.False(record.Matches(0, [0, 1, 2, 4]));
        Assert.False(record.Matches(1, [0, 1, 2, 3]));
        Assert.Throws<ArgumentOutOfRangeException>(() => record.Level(1));
        record.Dispose();
        Trellis2TestData.AssertDisposed(level);
        Assert.Throws<ObjectDisposedException>(() => record.Level(0));
    }

    [Fact]
    public void TextureDecoderStage_MapsTheHeadFromMinusOneOneToZeroOne()
    {
        ShapeVaeDecoderConfig predicting = ShapeVaeTestData.Decoder();
        Dictionary<string, Tensor> shapeWeights = ShapeVaeTestData.RandomWeights(ShapeVaeKeyMap.Decoder(predicting), seed: 91);
        Dictionary<string, Tensor> textureWeights = ShapeVaeTestData.RandomWeights(ShapeVaeKeyMap.Decoder(TextureConfig()), seed: 92);
        (SparseTensor latent, Tensor feats) = Latent(93);
        using CpuBackend backend = new();
        using ShapeVaeDecoder shape = new(predicting, shapeWeights);
        using ShapeVaeSubdivisions record = new();
        shape.Decode(backend, latent, record).Feats.Dispose();
        using ShapeVaeDecoder raw = new(TextureConfig(), textureWeights);
        SparseTensor head = raw.DecodeGuided(backend, latent, record);
        using Trellis2TextureDecoderStage stage = new(new ShapeVaeDecoder(TextureConfig(), textureWeights));

        SparseTensor attributes = stage.Decode(backend, latent, record);

        float[] expected = Trellis2TestData.ToArray(head.Feats);
        float[] actual = Trellis2TestData.ToArray(attributes.Feats);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i] * 0.5f + 0.5f, actual[i], 6);
        feats.Dispose();
        head.Feats.Dispose();
        attributes.Feats.Dispose();
        Trellis2TestData.DisposeAll(shapeWeights.Values);
        Trellis2TestData.DisposeAll(textureWeights.Values);
    }

    [Fact]
    public void TextureFlow_IsTheSlatFlowOverTheNoiseWithTheShapeLatentAppended()
    {
        Trellis2FlowConfig config = Trellis2TestData.Tiny(dense: false) with { InChannels = 8, OutChannels = 4 };
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(config, 101);
        using Trellis2FlowWeights weights = Trellis2FlowWeights.FromTensors(config, tensors);
        using Trellis2TextureFlow textureFlow = new(weights);
        using Trellis2SlatFlow plain = new(weights);
        Random rng = new(102);
        int[] coords = [0, 1, 2, 3, 0, 7, 0, 4, 0, 3, 3, 3, 0, 5, 6, 1];
        int n = coords.Length / 4;
        using Tensor x = Trellis2TestData.Random(rng, 1, n, 4);
        using Tensor shape = Trellis2TestData.Random(rng, 1, n, 4);
        using Tensor cond = Trellis2TestData.Random(rng, 1, 6, config.CondChannels);
        using CpuBackend backend = new();
        using Tensor joined = new(new TensorShape(1, n, 8), DType.F32);
        for (int row = 0; row < n; row++)
        {
            x.AsReadOnlySpan<float>().Slice(row * 4, 4).CopyTo(joined.AsSpan<float>().Slice(row * 8, 4));
            shape.AsReadOnlySpan<float>().Slice(row * 4, 4).CopyTo(joined.AsSpan<float>().Slice(row * 8 + 4, 4));
        }

        using Tensor actual = textureFlow.Forward(backend, x, shape, coords, 420f, cond);
        using Tensor expected = plain.Forward(backend, joined, coords, 420f, cond);

        Assert.Equal(4, textureFlow.LatentChannels);
        Assert.Equal(4, textureFlow.ConcatChannels);
        Assert.Equal(new TensorShape(1, n, 4), actual.Shape);
        Assert.Equal(Trellis2TestData.ToArray(expected), Trellis2TestData.ToArray(actual));
        Trellis2TestData.DisposeAll(tensors.Values);
    }

    [Fact]
    public void TextureFlow_ChecksTheShapeLatentAndTheArchitecture()
    {
        Trellis2FlowConfig config = Trellis2TestData.Tiny(dense: false) with { InChannels = 8, OutChannels = 4 };
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(config, 111);
        using Trellis2FlowWeights weights = Trellis2FlowWeights.FromTensors(config, tensors);
        using Trellis2TextureFlow flow = new(weights);
        Random rng = new(112);
        int[] coords = [0, 1, 2, 3, 0, 7, 0, 4];
        using Tensor x = Trellis2TestData.Random(rng, 1, 2, 4);
        using Tensor wrongWidth = Trellis2TestData.Random(rng, 1, 2, 3);
        using Tensor wrongRows = Trellis2TestData.Random(rng, 1, 3, 4);
        using Tensor cond = Trellis2TestData.Random(rng, 1, 6, config.CondChannels);
        using CpuBackend backend = new();

        Assert.Throws<ArgumentException>(() => flow.Forward(backend, x, wrongWidth, coords, 100f, cond));
        Assert.Throws<ArgumentException>(() => flow.Forward(backend, x, wrongRows, coords, 100f, cond));

        // A flow with no extra input channels (the shape flow) is not a texture flow.
        Trellis2FlowConfig shapeConfig = Trellis2TestData.Tiny(dense: false);
        Dictionary<string, Tensor> shapeTensors = Trellis2TestData.RandomWeights(shapeConfig, 113);
        using Trellis2FlowWeights shapeWeights = Trellis2FlowWeights.FromTensors(shapeConfig, shapeTensors);
        Assert.Throws<ArgumentException>(() => new Trellis2TextureFlow(shapeWeights));
        Trellis2TestData.DisposeAll(tensors.Values);
        Trellis2TestData.DisposeAll(shapeTensors.Values);
    }

    [Fact]
    public void LatentNormalization_RoundTripsAndChecksRowWidth()
    {
        float[] mean = [1f, -2f, 0.5f];
        float[] std = [2f, 0.5f, 4f];
        float[] original = [1f, 2f, 3f, -1f, 0f, 8f];
        using Tensor latent = new(new TensorShape(1, 2, 3), DType.F32);
        original.CopyTo(latent.AsSpan<float>());

        Trellis2LatentNormalization.Normalize(latent, mean, std);

        Assert.Equal([0f, 8f, 0.625f, -1f, 4f, 1.875f], latent.AsReadOnlySpan<float>().ToArray());
        Trellis2LatentNormalization.Denormalize(latent, mean, std);
        float[] restored = latent.AsReadOnlySpan<float>().ToArray();
        for (int i = 0; i < original.Length; i++)
            Assert.Equal(original[i], restored[i], 5);
        using Tensor ragged = new(new TensorShape(1, 7), DType.F32);
        Assert.Throws<ArgumentException>(() => Trellis2LatentNormalization.Normalize(ragged, mean, std));
    }
}
