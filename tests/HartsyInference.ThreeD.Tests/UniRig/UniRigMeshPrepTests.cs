using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.UniRig;

namespace HartsyInference.ThreeD.Tests.UniRig;

public sealed class UniRigMeshPrepTests
{
    /// <summary>A 2×4×6 box with duplicated corner vertices per face (24 vertices, 12 triangles), offset from the origin.</summary>
    private static Mesh SplitBox()
    {
        float[][] corners =
        [
            [0, 0, 0], [2, 0, 0], [2, 4, 0], [0, 4, 0], [0, 0, 6], [2, 0, 6], [2, 4, 6], [0, 4, 6],
        ];
        int[][] quads = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]];
        List<float> vertices = [];
        List<int> indices = [];
        foreach (int[] quad in quads)
        {
            int baseIndex = vertices.Count / 3;
            foreach (int corner in quad)
            {
                vertices.AddRange([corners[corner][0] + 10f, corners[corner][1] - 3f, corners[corner][2]]);
            }

            indices.AddRange([baseIndex, baseIndex + 1, baseIndex + 2, baseIndex, baseIndex + 2, baseIndex + 3]);
        }

        return new Mesh { Vertices = [.. vertices], Indices = [.. indices] };
    }

    [Fact]
    public void Prepare_WeldsDuplicatesAndNormalizesByTheLongestSide()
    {
        UniRigPreparedMesh prepared = UniRigMeshPrep.Prepare(SplitBox());
        Assert.Equal(8, prepared.VertexCount);
        Assert.Equal(12, prepared.FaceCount);
        Assert.Equal(3f, prepared.Scale);
        Assert.Equal([11f, -1f, 3f], prepared.Center);

        float[] lo = [float.MaxValue, float.MaxValue, float.MaxValue];
        float[] hi = [float.MinValue, float.MinValue, float.MinValue];
        for (int i = 0; i < prepared.Vertices.Length; i++)
        {
            lo[i % 3] = Math.Min(lo[i % 3], prepared.Vertices[i]);
            hi[i % 3] = Math.Max(hi[i % 3], prepared.Vertices[i]);
        }

        Assert.Equal([-1f / 3f, -2f / 3f, -1f], lo, new FloatComparer());
        Assert.Equal([1f / 3f, 2f / 3f, 1f], hi, new FloatComparer());
        Assert.Equal([10f + 0f, -3f, 0f], prepared.ToSource(new float[] { -1f / 3f, -2f / 3f, -1f }), new FloatComparer());
    }

    [Fact]
    public void Prepare_NormalsAreUnitLengthAndFaceNormalsFollowWinding()
    {
        UniRigPreparedMesh prepared = UniRigMeshPrep.Prepare(SplitBox());
        for (int f = 0; f < prepared.FaceCount; f++)
        {
            float length = MathF.Sqrt(prepared.FaceNormals[3 * f] * prepared.FaceNormals[3 * f] + prepared.FaceNormals[3 * f + 1] * prepared.FaceNormals[3 * f + 1]
                + prepared.FaceNormals[3 * f + 2] * prepared.FaceNormals[3 * f + 2]);
            Assert.Equal(1f, length, 4);
        }

        for (int v = 0; v < prepared.VertexCount; v++)
        {
            float length = MathF.Sqrt(prepared.VertexNormals[3 * v] * prepared.VertexNormals[3 * v] + prepared.VertexNormals[3 * v + 1] * prepared.VertexNormals[3 * v + 1]
                + prepared.VertexNormals[3 * v + 2] * prepared.VertexNormals[3 * v + 2]);
            Assert.Equal(1f, length, 4);
        }

        // Every box corner's angle-weighted normal points outwards, away from the origin of the normalized cube.
        for (int v = 0; v < prepared.VertexCount; v++)
        {
            float dot = prepared.VertexNormals[3 * v] * prepared.Vertices[3 * v] + prepared.VertexNormals[3 * v + 1] * prepared.Vertices[3 * v + 1]
                + prepared.VertexNormals[3 * v + 2] * prepared.Vertices[3 * v + 2];
            Assert.True(dot > 0f, $"vertex {v} normal points inwards (dot {dot})");
        }
    }

    [Fact]
    public void Prepare_RejectsEmptyOrNonFiniteMeshes()
    {
        Assert.Throws<HartsyInferenceException>(() => UniRigMeshPrep.Prepare(new Mesh { Vertices = [0, 0, 0], Indices = [] }));
        Assert.Throws<HartsyInferenceException>(() => UniRigMeshPrep.Prepare(
            new Mesh { Vertices = [float.NaN, 0, 0, 1, 0, 0, 0, 1, 0], Indices = [0, 1, 2] }));
        Assert.Throws<HartsyInferenceException>(() => UniRigMeshPrep.Prepare(
            new Mesh { Vertices = [1, 1, 1, 1, 1, 1, 1, 1, 1], Indices = [0, 1, 2] }));
    }

    [Fact]
    public void Sampler_ReturnsVertexSamplesFirstThenSurfacePointsOnTheMesh()
    {
        UniRigPreparedMesh prepared = UniRigMeshPrep.Prepare(SplitBox());
        UniRigPointSample sample = UniRigSurfaceSampler.Sample(prepared, total: 500, vertexSamples: 5, seed: 3);
        Assert.Equal(500, sample.Count);
        for (int i = 0; i < 5; i++)
        {
            Assert.True(sample.Vertex[i] >= 0 && sample.Face[i] == -1);
        }

        Assert.Equal(5, sample.Vertex.Take(5).Distinct().Count());
        float[] half = [1f / 3f, 2f / 3f, 1f];
        for (int i = 5; i < 500; i++)
        {
            Assert.True(sample.Face[i] >= 0 && sample.Vertex[i] == -1);
            bool onSurface = false;
            for (int axis = 0; axis < 3; axis++)
            {
                onSurface |= Math.Abs(Math.Abs(sample.Points[3 * i + axis]) - half[axis]) < 1e-5f;
                Assert.True(Math.Abs(sample.Points[3 * i + axis]) <= half[axis] + 1e-5f);
            }

            Assert.True(onSurface, $"surface sample {i} is not on the box surface");
            Assert.True(sample.Barycentric[2 * i] + sample.Barycentric[2 * i + 1] <= 1f + 1e-6f);
        }
    }

    [Fact]
    public void Sampler_SameSeedSamePoints_DifferentSeedDifferent()
    {
        UniRigPreparedMesh prepared = UniRigMeshPrep.Prepare(SplitBox());
        UniRigPointSample a = UniRigSurfaceSampler.Sample(prepared, 64, 4, 1);
        UniRigPointSample b = UniRigSurfaceSampler.Sample(prepared, 64, 4, 1);
        UniRigPointSample c = UniRigSurfaceSampler.Sample(prepared, 64, 4, 2);
        Assert.Equal(a.Points, b.Points);
        Assert.NotEqual(a.Points, c.Points);
    }

    [Fact]
    public void Sampler_FacePickingFollowsSquaredCrossLength()
    {
        Mesh mesh = new()
        {
            Vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0, 3, 0, 0, 3, 2, 0, 5, 0, 0],
            Indices = [0, 1, 2, 3, 4, 5],
        };
        // Cross lengths: face 0 = 1, face 1 = 4 (edges (0,2,0) and (2,0,0)); squared 1 : 16 → ~94% on face 1.
        UniRigPreparedMesh prepared = UniRigMeshPrep.Prepare(mesh);
        UniRigPointSample sample = UniRigSurfaceSampler.Sample(prepared, 4000, 0, 11);
        double share = sample.Face.Count(f => f == 1) / 4000.0;
        Assert.InRange(share, 0.90, 0.98);
    }

    [Fact]
    public void PointSelection_DistinctIndicesWithoutReplacement_RepeatsWhenOversubscribed()
    {
        int[] distinct = UniRigPointSelection.Choose(100, 40, 5);
        Assert.Equal(40, distinct.Distinct().Count());
        Assert.All(distinct, i => Assert.InRange(i, 0, 99));
        int[] repeated = UniRigPointSelection.Choose(10, 50, 5);
        Assert.Equal(50, repeated.Length);
        Assert.All(repeated, i => Assert.InRange(i, 0, 9));
    }

    private sealed class FloatComparer : IEqualityComparer<float>
    {
        public bool Equals(float x, float y) => Math.Abs(x - y) < 1e-5f;

        public int GetHashCode(float value) => 0;
    }
}
