using System.Numerics;
using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Geometry;

public sealed class SceneGridTests
{
    [Fact]
    public void Fit_KeepsTheSceneInsideOneVoxelOfMarginOnEveryAxis()
    {
        Mesh cube = ObjReader.Parse("v -1 -1 -1\nv 1 -1 -1\nv 1 1 -1\nv -1 1 -1\nv -1 -1 1\nv 1 -1 1\nv 1 1 1\nv -1 1 1\nf 1 3 2\nf 5 6 7\n");

        SceneGrid grid = SceneGrid.Fit(cube, 16);
        Vector3 low = grid.ToVoxelUnits(new Vector3(-1f));
        Vector3 high = grid.ToVoxelUnits(new Vector3(1f));

        Assert.Equal(16, grid.Resolution);
        Assert.True(grid.VoxelSize > 0f);
        foreach (float v in new[] { low.X, low.Y, low.Z })
            Assert.InRange(v, 1f - 1e-4f, 1f + 1e-4f);
        foreach (float v in new[] { high.X, high.Y, high.Z })
            Assert.InRange(v, 15f - 1e-4f, 15f + 1e-4f);
    }

    [Fact]
    public void Fit_RejectsTooSmallGridAndEmptyScene()
    {
        Mesh cube = ObjReader.Parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
        Mesh empty = new() { Vertices = [0f, 0f, 0f], Indices = [] };

        Assert.Throws<ArgumentOutOfRangeException>(() => SceneGrid.Fit(cube, 8));
        Assert.Throws<HartsyInferenceException>(() => SceneGrid.Fit(empty, 16));
    }
}
