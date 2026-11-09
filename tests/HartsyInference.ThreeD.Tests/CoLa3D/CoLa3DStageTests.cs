using System.Numerics;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Models.CoLa3D;
using HartsyInference.ThreeD.Models.TrellisV2;
using HartsyInference.ThreeD.Models.Utonia;
using HartsyInference.ThreeD.Pipelines.Stages;
using HartsyInference.ThreeD.Tests.TrellisV2;
using HartsyInference.ThreeD.Tests.Utonia;
using Xunit;

namespace HartsyInference.ThreeD.Tests.CoLa3D;

/// <summary>Per-stage checks of the CoLa3D scene pipeline on tiny synthetic configs and random weights. No checkpoint is read.</summary>
public sealed class CoLa3DStageTests
{
    private const string CubeObj = "v -1 -1 -1\nv 1 -1 -1\nv 1 1 -1\nv -1 1 -1\nv -1 -1 1\nv 1 -1 1\nv 1 1 1\nv -1 1 1\n"
        + "f 1 3 2\nf 1 4 3\nf 5 6 7\nf 5 7 8\nf 1 2 6\nf 1 6 5\nf 2 3 7\nf 2 7 6\nf 3 4 8\nf 3 8 7\nf 4 1 5\nf 4 5 8\n";

    [Fact]
    public void SceneVoxelizer_PlacesEveryOccupiedVoxelInsideTheGridWithItsDualVertexInside()
    {
        Mesh cube = ObjReader.Parse(CubeObj);
        SceneGrid grid = SceneGrid.Fit(cube, 16);

        DualGridVoxels voxels = new DualGridSceneVoxelizer().Voxelize(cube, grid);

        Assert.True(voxels.Coords.Length > 0);
        Assert.Equal(voxels.Coords.Length, voxels.DualVertices.Length);
        Assert.Equal(voxels.Coords.Length, voxels.Intersected.Length);
        for (int i = 0; i < voxels.Coords.Length; i++)
        {
            Assert.InRange(voxels.Coords[i], 0, grid.Resolution - 1);
            Assert.InRange(voxels.DualVertices[i] - voxels.Coords[i], -1e-4f, 1f + 1e-4f);
        }
    }

    [Fact]
    public void DualGridMesherStage_EmitsTwoTrianglesForAnIntersectedQuad()
    {
        int[] coords = [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0];
        float[] dual = [0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f];
        bool[] intersected = new bool[12];
        intersected[0] = true;
        DualGridField field = new(new DualGridVoxels(coords, dual, intersected), [1f, 1f, 1f, 1f]);
        SceneGrid grid = new() { Resolution = 16, Origin = Vector3.Zero, VoxelSize = 1f };

        Mesh mesh = new DualGridMesherStage().Extract(field, grid);

        Assert.Equal(2, mesh.TriangleCount);
        Assert.Equal(4, mesh.VertexCount);
        Assert.All(mesh.Vertices, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void ObjectDecomposer_ReturnsSingleSampleObjectLatentsOnTheLatentGrid()
    {
        CoLa3DConfig config = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.Fixed);
        using CoLa3DDecomposer decomposer = new(config, CoLa3DTestSupport.RandomWeights(config, 7));
        using CoLa3DObjectDecomposer stage = new(config, decomposer);
        (SparseTensor latent, Tensor utonia) = CoLa3DTestSupport.RandomLatent(config, [8], seed: 2);
        SparseTensor features = new(utonia, latent.Coords, latent.Resolution);
        using CpuBackend backend = new();
        try
        {
            IReadOnlyList<SparseTensor> objects = stage.Decompose(backend, latent, features);

            foreach (SparseTensor part in objects)
            {
                Assert.Equal(config.LatentChannels, part.Channels);
                Assert.Equal(latent.Resolution, part.Resolution);
                Assert.True(part.Count >= CoLa3DObjectSplit.MinObjectVoxels);
                for (int r = 0; r < part.Count; r++)
                    Assert.Equal(0, part.Coords[r * 4]);
                part.Feats.Dispose();
            }
        }
        finally
        {
            latent.Feats.Dispose();
            utonia.Dispose();
        }
    }

    [Fact]
    public void ObjectDecomposer_RejectsFeaturesThatDoNotCoverTheLatentVoxels()
    {
        CoLa3DConfig config = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.Fixed);
        using CoLa3DDecomposer decomposer = new(config, CoLa3DTestSupport.RandomWeights(config, 7));
        using CoLa3DObjectDecomposer stage = new(config, decomposer);
        (SparseTensor latent, Tensor utonia) = CoLa3DTestSupport.RandomLatent(config, [8], seed: 4);
        int[] shifted = (int[])latent.Coords.Clone();
        shifted[^3] = shifted[^3] == 7 ? 6 : shifted[^3] + 1;
        SparseTensor features = new(utonia, shifted, latent.Resolution);
        using CpuBackend backend = new();
        try
        {
            Assert.Throws<HartsyInferenceException>(() => stage.Decompose(backend, latent, features));
        }
        finally
        {
            latent.Feats.Dispose();
            utonia.Dispose();
        }
    }

    [Fact]
    public void UtoniaStage_ReturnsFeaturesOnTheGridReducedByItsTotalStride()
    {
        UtoniaConfig config = UtoniaTestData.TinyConfig(encoderOnly: true);
        using CpuBackend backend = new();
        UtoniaEncoder encoder = new(backend, config, UtoniaTestData.RandomWeights(config, 5));
        using UtoniaPointFeatureStage stage = new(encoder, totalStride: 4);
        int[] coords = ShapeVaeTestData.RandomCoords(new Random(9), 30, 16);

        SparseTensor features = stage.Extract(backend, coords, 16);

        Assert.Equal(4, features.Resolution);
        Assert.Equal(config.EncChannels[^1], features.Channels);
        Assert.Equal(0, features.Coords.Length % 4);
        Assert.True(features.Count > 0);
        features.Feats.Dispose();
        encoder.Dispose();
    }

    [Fact]
    public void ShapeVaeStages_EncodeThenDecodeATinyDualGrid()
    {
        (Dictionary<string, Tensor> encWeights, Dictionary<string, Tensor> decWeights) = ShapeVaeTestData.RandomPair(
            ShapeVaeTestData.Encoder(), ShapeVaeTestData.Decoder(), seed: 3);
        using CpuBackend backend = new();
        using ShapeVaeEncoderStage encoder = new(new ShapeVaeEncoder(ShapeVaeTestData.Encoder(), encWeights));
        using ShapeVaeDecoderStage decoder = new(new ShapeVaeDecoder(ShapeVaeTestData.Decoder(), decWeights));
        int[] coords = ShapeVaeTestData.RandomCoords(new Random(11), 20, 16);
        int n = coords.Length / 4;
        float[] dual = new float[n * 3];
        bool[] intersected = new bool[n * 3];
        int[] xyz = new int[n * 3];
        for (int i = 0; i < n; i++)
        {
            for (int a = 0; a < 3; a++)
            {
                xyz[i * 3 + a] = coords[i * 4 + 1 + a];
                dual[i * 3 + a] = xyz[i * 3 + a] + 0.5f;
                intersected[i * 3 + a] = (i + a) % 2 == 0;
            }
        }

        SparseTensor latent = encoder.Encode(backend, new DualGridVoxels(xyz, dual, intersected), 16);
        DualGridField field = decoder.Decode(backend, latent);
        latent.Feats.Dispose();

        Assert.Equal(4, latent.Channels);
        Assert.Equal(4, latent.Resolution);
        Assert.Equal(latent.Count, latent.Feats.Shape[0]);
        Assert.Equal(field.SplitWeight.Length * 3, field.Voxels.Coords.Length);
        Assert.Equal(field.Voxels.Coords.Length, field.Voxels.Intersected.Length);
        Assert.All(field.Voxels.Coords, c => Assert.InRange(c, 0, 15));
    }

    [Fact]
    public void CheckpointConverter_MapsUtoniaParametersAndDropsComputedBuffers()
    {
        Tensor weight = new(new TensorShape(1), DType.F32);
        Tensor rope = new(new TensorShape(1), DType.F32);
        Tensor mask = new(new TensorShape(1), DType.F32);
        Dictionary<string, Tensor> stateDict = new(StringComparer.Ordinal)
        {
            ["state_dict.enc.enc0.block0.attn.qkv.weight"] = weight,
            ["state_dict.enc.enc0.block0.attn.rope.inv_freq"] = rope,
            ["state_dict.embedding.mask_token"] = mask,
        };

        Dictionary<string, Tensor> named = CoLa3DCheckpointConverter.ToUtoniaNames(stateDict);

        string expected = UtoniaCheckpointKeys.ToCSharpName("enc.enc0.block0.attn.qkv.weight")!;
        Assert.Single(named);
        Assert.Same(weight, named[expected]);
        weight.Dispose();
        rope.Dispose();
        mask.Dispose();
    }
}
