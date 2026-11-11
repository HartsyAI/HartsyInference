using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.TrellisV2;
using HartsyInference.ThreeD.Pipelines.Stages;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

/// <summary>The partial decode the TRELLIS.2 cascade uses: upstream <c>upsample(x, times)</c>.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class ShapeVaeUpsampleTests
{
    private const int LatentRes = 4;

    private static (ShapeVaeDecoder Decoder, SparseTensor Latent) Build(int seed)
    {
        Random rng = new(seed);
        int[] coords = ShapeVaeTestData.RandomCoords(rng, 10, LatentRes);
        Tensor feats = ShapeVaeTestData.RandomFeats(rng, 10, 4, -1, 1);
        (_, Dictionary<string, Tensor> decW) = ShapeVaeTestData.RandomPair(ShapeVaeTestData.Encoder(), ShapeVaeTestData.Decoder(), seed: seed + 10);
        return (new ShapeVaeDecoder(ShapeVaeTestData.Decoder(), decW), ShapeVaeTestData.Sparse(coords, LatentRes, feats));
    }

    [Fact]
    public void UpsampleToTheLastLevelGivesTheCoordinatesTheFullDecodeEmits()
    {
        // The tiny decoder has three levels and no blocks after the last, so Upsample(2) ends where Decode's output head begins.
        (ShapeVaeDecoder decoder, SparseTensor latent) = Build(3);
        using ShapeVaeDecoder owned = decoder;
        CpuBackend backend = new();

        int[] up = decoder.Upsample(backend, latent, 2);
        SparseTensor head = decoder.Decode(backend, latent);

        Assert.Equal(head.Coords, up);
        Assert.Equal(4 * LatentRes, head.Resolution);
        head.Feats.Dispose();
        latent.Feats.Dispose();
    }

    [Fact]
    public void UpsampleOneLevelGivesChildrenOnTheNextFinerGrid()
    {
        (ShapeVaeDecoder decoder, SparseTensor latent) = Build(4);
        using ShapeVaeDecoder owned = decoder;

        int[] up = decoder.Upsample(new CpuBackend(), latent, 1);

        Assert.InRange(up.Length / 4, 1, latent.Count * 8);
        for (int i = 0; i < up.Length; i += 4)
        {
            Assert.Equal(0, up[i]);
            for (int a = 1; a <= 3; a++)
                Assert.InRange(up[i + a], 0, 2 * LatentRes - 1);
        }

        latent.Feats.Dispose();
    }

    [Fact]
    public void UpsampleDoesNotDisturbALaterFullDecode()
    {
        (ShapeVaeDecoder decoder, SparseTensor latent) = Build(5);
        using ShapeVaeDecoder owned = decoder;
        CpuBackend backend = new();

        SparseTensor first = decoder.Decode(backend, latent);
        decoder.Upsample(backend, latent, 1);
        SparseTensor second = decoder.Decode(backend, latent);

        Assert.Equal(first.Coords, second.Coords);
        Assert.Equal(Trellis2TestData.ToArray(first.Feats), Trellis2TestData.ToArray(second.Feats));
        first.Feats.Dispose();
        second.Feats.Dispose();
        latent.Feats.Dispose();
    }

    [Fact]
    public void UpsampleRejectsALevelCountOutsideTheNetwork()
    {
        (ShapeVaeDecoder decoder, SparseTensor latent) = Build(6);
        using ShapeVaeDecoder owned = decoder;
        CpuBackend backend = new();

        Assert.Throws<ArgumentOutOfRangeException>(() => decoder.Upsample(backend, latent, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => decoder.Upsample(backend, latent, 3));
        latent.Feats.Dispose();
    }

    [Fact]
    public void TheDecoderStageServesBothTheDecodeAndTheUpsample()
    {
        (ShapeVaeDecoder decoder, SparseTensor latent) = Build(7);
        using ShapeVaeDecoderStage stage = new(decoder);
        CpuBackend backend = new();

        ISparseShapeUpsampler upsampler = stage;
        int[] coords = upsampler.Upsample(backend, latent, 2);
        HartsyInference.ThreeD.Geometry.DualGridField field = stage.Decode(backend, latent);

        Assert.Equal(coords.Length / 4, field.SplitWeight.Length);
        latent.Feats.Dispose();
    }
}
