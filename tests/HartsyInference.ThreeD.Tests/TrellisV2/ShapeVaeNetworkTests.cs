using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

public sealed class ShapeVaeNetworkTests
{
    private const int LatentRes = 4, FineRes = 16;

    [Fact]
    public void Encoder_ReturnsLatentMeanOnCoarseGrid()
    {
        Random rng = new(11);
        int n = 60;
        int[] coords = ShapeVaeTestData.RandomCoords(rng, n, FineRes);
        using Tensor vertices = ShapeVaeTestData.RandomFeats(rng, n, 3, 0, 1);
        using Tensor intersected = ShapeVaeTestData.RandomFeats(rng, n, 3, 0, 1);
        (Dictionary<string, Tensor> encW, _) = ShapeVaeTestData.RandomPair(ShapeVaeTestData.Encoder(), ShapeVaeTestData.Decoder(), seed: 21);
        using ShapeVaeEncoder encoder = new(ShapeVaeTestData.Encoder(), encW);

        SparseTensor vertexSparse = ShapeVaeTestData.Sparse(coords, FineRes, vertices);
        SparseTensor latent = encoder.Encode(new CpuBackend(), vertexSparse, ShapeVaeTestData.Sparse(coords, FineRes, intersected));

        Assert.Equal(LatentRes, latent.Resolution);
        Assert.Equal(ShapeVaeTestData.Encoder().LatentChannels, latent.Channels);
        Assert.InRange(latent.Count, 1, n);
        Assert.Equal((long)latent.Count * latent.Channels, latent.Feats.ElementCount);
        AssertCoordsInside(latent.Coords, LatentRes);
        AssertDistinct(latent.Coords);
        AssertAllFinite(latent.Feats);
    }

    [Fact]
    public void Decoder_ExpandsCoarseLatentToFineGridWithSevenChannels()
    {
        Random rng = new(12);
        int n = 10;
        int[] coords = ShapeVaeTestData.RandomCoords(rng, n, LatentRes);
        using Tensor latentFeats = ShapeVaeTestData.RandomFeats(rng, n, 4, -1, 1);
        (_, Dictionary<string, Tensor> decW) = ShapeVaeTestData.RandomPair(ShapeVaeTestData.Encoder(), ShapeVaeTestData.Decoder(), seed: 22);
        using ShapeVaeDecoder decoder = new(ShapeVaeTestData.Decoder(), decW);

        SparseTensor head = decoder.Decode(new CpuBackend(), ShapeVaeTestData.Sparse(coords, LatentRes, latentFeats));

        Assert.Equal(FineRes, head.Resolution);
        Assert.Equal(ShapeVaeKeyMap.DecoderOutputChannels, head.Channels);
        // Each of the two upsamples expands a voxel into at most 8 children.
        Assert.InRange(head.Count, 1, n * 64);
        Assert.Equal((long)head.Count * head.Channels, head.Feats.ElementCount);
        AssertCoordsInside(head.Coords, FineRes);
        AssertDistinct(head.Coords);
        AssertAllFinite(head.Feats);
    }

    [Fact]
    public void EncodeThenDecode_RoundTripsShapesThroughBothNetworks()
    {
        Random rng = new(13);
        int n = 48;
        int[] coords = ShapeVaeTestData.RandomCoords(rng, n, FineRes);
        using Tensor vertices = ShapeVaeTestData.RandomFeats(rng, n, 3, 0, 1);
        using Tensor intersected = ShapeVaeTestData.RandomFeats(rng, n, 3, 0, 1);
        (Dictionary<string, Tensor> encW, Dictionary<string, Tensor> decW) =
            ShapeVaeTestData.RandomPair(ShapeVaeTestData.Encoder(), ShapeVaeTestData.Decoder(), seed: 23);
        using ShapeVaeEncoder encoder = new(ShapeVaeTestData.Encoder(), encW);
        using ShapeVaeDecoder decoder = new(ShapeVaeTestData.Decoder(), decW);
        CpuBackend backend = new();

        SparseTensor vertexSparse = ShapeVaeTestData.Sparse(coords, FineRes, vertices);
        SparseTensor latent = encoder.Encode(backend, vertexSparse, ShapeVaeTestData.Sparse(coords, FineRes, intersected));
        SparseTensor head = decoder.Decode(backend, latent);

        Assert.Equal(FineRes, head.Resolution);
        Assert.Equal(ShapeVaeKeyMap.DecoderOutputChannels, head.Channels);
        AssertCoordsInside(head.Coords, FineRes);
        AssertAllFinite(head.Feats);
    }

    [Fact]
    public void Encoder_RejectsResolutionNotMultipleOfDownsampleFactor()
    {
        Random rng = new(14);
        int[] coords = ShapeVaeTestData.RandomCoords(rng, 8, 10);
        using Tensor vertices = ShapeVaeTestData.RandomFeats(rng, 8, 3, 0, 1);
        using Tensor intersected = ShapeVaeTestData.RandomFeats(rng, 8, 3, 0, 1);
        (Dictionary<string, Tensor> encW, _) = ShapeVaeTestData.RandomPair(ShapeVaeTestData.Encoder(), ShapeVaeTestData.Decoder(), seed: 24);
        using ShapeVaeEncoder encoder = new(ShapeVaeTestData.Encoder(), encW);

        // The synthetic encoder downsamples 4x, so 10 is not divisible.
        SparseTensor v = ShapeVaeTestData.Sparse(coords, 10, vertices), e = ShapeVaeTestData.Sparse(coords, 10, intersected);
        Assert.Throws<ArgumentException>(() => encoder.Encode(new CpuBackend(), v, e));
    }

    [Fact]
    public void Encoder_RejectsIntersectedWithDifferentCoords()
    {
        Random rng = new(15);
        int[] coords = ShapeVaeTestData.RandomCoords(rng, 8, FineRes);
        int[] other = ShapeVaeTestData.RandomCoords(rng, 8, FineRes);
        using Tensor vertices = ShapeVaeTestData.RandomFeats(rng, 8, 3, 0, 1);
        using Tensor intersected = ShapeVaeTestData.RandomFeats(rng, 8, 3, 0, 1);
        (Dictionary<string, Tensor> encW, _) = ShapeVaeTestData.RandomPair(ShapeVaeTestData.Encoder(), ShapeVaeTestData.Decoder(), seed: 25);
        using ShapeVaeEncoder encoder = new(ShapeVaeTestData.Encoder(), encW);

        SparseTensor v = ShapeVaeTestData.Sparse(coords, FineRes, vertices), e = ShapeVaeTestData.Sparse(other, FineRes, intersected);
        Assert.Throws<ArgumentException>(() => encoder.Encode(new CpuBackend(), v, e));
    }

    [Fact]
    public void Encoder_RejectsInputWithMoreThanOneBatch()
    {
        Random rng = new(17);
        int[] coords = ShapeVaeTestData.RandomCoords(rng, 8, FineRes);
        coords[4] = 1;
        using Tensor vertices = ShapeVaeTestData.RandomFeats(rng, 8, 3, 0, 1);
        using Tensor intersected = ShapeVaeTestData.RandomFeats(rng, 8, 3, 0, 1);
        (Dictionary<string, Tensor> encW, _) = ShapeVaeTestData.RandomPair(ShapeVaeTestData.Encoder(), ShapeVaeTestData.Decoder(), seed: 27);
        using ShapeVaeEncoder encoder = new(ShapeVaeTestData.Encoder(), encW);

        SparseTensor v = ShapeVaeTestData.Sparse(coords, FineRes, vertices), e = ShapeVaeTestData.Sparse(coords, FineRes, intersected);
        ArgumentException ex = Assert.Throws<ArgumentException>(() => encoder.Encode(new CpuBackend(), v, e));
        Assert.Contains("one batch", ex.Message);
    }

    [Fact]
    public void Decoder_RejectsLatentWithWrongChannelCount()
    {
        Random rng = new(16);
        int[] coords = ShapeVaeTestData.RandomCoords(rng, 4, LatentRes);
        using Tensor latentFeats = ShapeVaeTestData.RandomFeats(rng, 4, 5, -1, 1);
        (_, Dictionary<string, Tensor> decW) = ShapeVaeTestData.RandomPair(ShapeVaeTestData.Encoder(), ShapeVaeTestData.Decoder(), seed: 26);
        using ShapeVaeDecoder decoder = new(ShapeVaeTestData.Decoder(), decW);

        Assert.Throws<ArgumentException>(() => decoder.Decode(new CpuBackend(), ShapeVaeTestData.Sparse(coords, LatentRes, latentFeats)));
    }

    private static void AssertCoordsInside(int[] coords, int resolution)
    {
        for (int i = 0; i < coords.Length; i += 4)
        {
            Assert.True(coords[i] >= 0);
            for (int a = 1; a < 4; a++) Assert.InRange(coords[i + a], 0, resolution - 1);
        }
    }

    private static void AssertDistinct(int[] coords)
    {
        HashSet<(int, int, int, int)> seen = new();
        for (int i = 0; i < coords.Length; i += 4)
            Assert.True(seen.Add((coords[i], coords[i + 1], coords[i + 2], coords[i + 3])));
    }

    private static void AssertAllFinite(Tensor t)
    {
        foreach (float v in ShapeVaeTestData.Read(t)) Assert.True(float.IsFinite(v));
    }
}
