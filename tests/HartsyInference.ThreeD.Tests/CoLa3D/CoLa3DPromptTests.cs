using System.Numerics;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.CoLa3D;
using HartsyInference.ThreeD.Pipelines;
using Xunit;

namespace HartsyInference.ThreeD.Tests.CoLa3D;

/// <summary>Scene prompts: checkpoint selection, the mask tensor, point-to-latent mapping and snapping, and prompt validation.</summary>
public sealed class CoLa3DPromptTests
{
    [Theory]
    [InlineData(CoLa3DConfig.QuerySourceKind.Fixed, "cola3d_free_q20.pt")]
    [InlineData(CoLa3DConfig.QuerySourceKind.PromptMask, "cola3d_mask_q20.pt")]
    [InlineData(CoLa3DConfig.QuerySourceKind.PromptPoint, "cola3d_point_q20.pt")]
    public void QuerySource_SelectsItsCheckpointAndArchitecture(CoLa3DConfig.QuerySourceKind source, string fileName)
    {
        Assert.Equal(fileName, CoLa3DCheckpointConverter.DecomposerFileName(source));
        Assert.Equal(source, CoLa3DConfig.ForQuerySource(source).QuerySource);
    }

    [Fact]
    public void MaskQ20_UsesTheCenterEncodingAndPointQ20_ReadsTheFusedLatent()
    {
        Assert.Equal(CoLa3DConfig.MaskEncoderKind.CnnCenterPe, CoLa3DConfig.MaskQ20.MaskEncoder);
        Assert.Equal(CoLa3DConfig.PointFeatureSourceKind.LatentUtonia, CoLa3DConfig.PointQ20.PointFeatureSource);
        Assert.Equal(20, CoLa3DConfig.PointQ20.MaxParts);
    }

    [Fact]
    public void MaskPrompt_BecomesAnF32TensorOfCoverageFractions()
    {
        byte[] coverage = [0, 255, 128, 0, 255, 0, 0, 255, 255, 0, 0, 0, 255, 0, 255, 0, 0, 255, 255, 255, 0, 0, 0, 255];
        CoLa3DPrompt prompt = new() { Queries = 2, MaskCoverage = coverage, MaskHeight = 3, MaskWidth = 4 };
        SparseTensor latent = Latent(new[] { 0, 2, 2, 2 });
        try
        {
            using CoLa3DPromptInputs inputs = CoLa3DPromptInputs.Create(prompt, OneVoxelGrid(), latent);

            Assert.Equal(1, inputs.Prompts.Batch);
            Assert.Equal(2, inputs.Prompts.Queries);
            Assert.Null(inputs.Prompts.Points);
            Assert.Equal(new TensorShape(1, 2, 3, 4), inputs.Prompts.Masks!.Shape);
            ReadOnlySpan<float> values = inputs.Prompts.Masks.AsReadOnlySpan<float>();
            for (int i = 0; i < coverage.Length; i++)
                Assert.Equal(coverage[i] / 255f, values[i], 6);
        }
        finally
        {
            latent.Feats.Dispose();
        }
    }

    [Fact]
    public void PointPrompt_MapsToTheLatentVoxelThatHoldsIt()
    {
        // Dual voxel 40 on each axis is latent voxel 2 (40 / LatentStride 16).
        CoLa3DPrompt prompt = new() { Queries = 1, Points = [40.5f, 40.5f, 40.5f] };
        SparseTensor latent = Latent(new[] { 0, 2, 2, 2 });
        try
        {
            using CoLa3DPromptInputs inputs = CoLa3DPromptInputs.Create(prompt, OneVoxelGrid(), latent);

            Assert.Equal(new[] { 2, 2, 2 }, inputs.Prompts.Points);
            Assert.Null(inputs.Prompts.Masks);
        }
        finally
        {
            latent.Feats.Dispose();
        }
    }

    [Fact]
    public void PointPrompt_SnapsToTheNearestOccupiedLatentVoxel()
    {
        CoLa3DPrompt prompt = new() { Queries = 1, Points = [40.5f, 40.5f, 40.5f] };
        SparseTensor latent = Latent(new[] { 0, 0, 0, 0, 0, 3, 2, 2 });
        try
        {
            using CoLa3DPromptInputs inputs = CoLa3DPromptInputs.Create(prompt, OneVoxelGrid(), latent);

            Assert.Equal(new[] { 3, 2, 2 }, inputs.Prompts.Points);
        }
        finally
        {
            latent.Feats.Dispose();
        }
    }

    [Fact]
    public void PointPrompt_RejectsAPointOutsideTheSceneGrid()
    {
        CoLa3DPrompt prompt = new() { Queries = 1, Points = [-5f, 0f, 0f] };
        SparseTensor latent = Latent(new[] { 0, 0, 0, 0 });
        try
        {
            HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(
                () => CoLa3DPromptInputs.Create(prompt, OneVoxelGrid(), latent));
            Assert.Contains("outside the scene grid", error.Message, StringComparison.Ordinal);
        }
        finally
        {
            latent.Feats.Dispose();
        }
    }

    [Fact]
    public void Validate_RejectsMasksAndPointsTogetherOrNeither()
    {
        byte[] coverage = new byte[4];
        CoLa3DPrompt both = new()
        {
            Queries = 1,
            MaskCoverage = coverage,
            MaskHeight = 2,
            MaskWidth = 2,
            Points = [0f, 0f, 0f],
        };
        Assert.Throws<ArgumentException>(() => both.Validate());
        Assert.Throws<ArgumentException>(() => new CoLa3DPrompt { Queries = 1 }.Validate());
    }

    [Fact]
    public void Validate_RejectsBuffersThatDoNotMatchTheQueryCount()
    {
        CoLa3DPrompt shortMask = new() { Queries = 2, MaskCoverage = new byte[4], MaskHeight = 2, MaskWidth = 2 };
        Assert.Throws<ArgumentException>(() => shortMask.Validate());
        Assert.Throws<ArgumentException>(() => new CoLa3DPrompt { Queries = 2, Points = [0f, 0f, 0f] }.Validate());
        Assert.Throws<ArgumentException>(() => new CoLa3DPrompt { Queries = 1, Points = [float.NaN, 0f, 0f] }.Validate());
    }

    [Fact]
    public void Validate_RejectsAnEmptyMaskBecauseItHasNoCentroid()
    {
        // Query 1 is all zero; query 0 has coverage, so the check must look per query, not per prompt.
        byte[] coverage = [255, 0, 0, 0, 0, 0, 0, 0];
        CoLa3DPrompt prompt = new() { Queries = 2, MaskCoverage = coverage, MaskHeight = 2, MaskWidth = 2 };

        ArgumentException error = Assert.Throws<ArgumentException>(() => prompt.Validate());
        Assert.Contains("Mask 1 has no coverage", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Validate_NamesTheDimensionThatIsNotPositive()
    {
        ArgumentOutOfRangeException width = Assert.Throws<ArgumentOutOfRangeException>(
            () => new CoLa3DPrompt { Queries = 1, MaskCoverage = new byte[0], MaskHeight = 2, MaskWidth = 0 }.Validate());
        Assert.Equal("MaskWidth", width.ParamName);

        ArgumentOutOfRangeException height = Assert.Throws<ArgumentOutOfRangeException>(
            () => new CoLa3DPrompt { Queries = 1, MaskCoverage = new byte[0], MaskHeight = 0, MaskWidth = 2 }.Validate());
        Assert.Equal("MaskHeight", height.ParamName);
    }

    // A 1024 dual grid over unit voxels from the origin: a point at 40.5 lies in dual voxel 40.
    private static SceneGrid OneVoxelGrid() => new()
    {
        Resolution = CoLa3DPipeline.DefaultResolution,
        Origin = Vector3.Zero,
        VoxelSize = 1f,
    };

    // A 64-grid latent with one row per voxel in <paramref name="coords"/> (batch, x, y, z). The caller disposes Feats.
    private static SparseTensor Latent(int[] coords)
    {
        int n = coords.Length / 4;
        Tensor feats = new(new TensorShape(n, 4), DType.F32);
        return new SparseTensor(feats, coords, CoLa3DConfig.PointQ20.LatentResolution);
    }
}
