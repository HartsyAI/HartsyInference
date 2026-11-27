using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.UniRig;

namespace HartsyInference.ThreeD.Tests.UniRig;

public sealed class UniRigSkinGeometryTests
{
    /// <summary>An axis-aligned cube [-1,1]³ with shared corners (8 vertices, 12 triangles).</summary>
    private static UniRigPreparedMesh Cube() => UniRigMeshPrep.Prepare(new Mesh
    {
        Vertices = [-1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1],
        Indices = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3],
    });

    [Fact]
    public void KdTree_MatchesBruteForceNearestNeighbours()
    {
        Random random = new(4);
        float[] points = new float[3 * 500];
        for (int i = 0; i < points.Length; i++)
        {
            points[i] = (float)random.NextDouble();
        }

        UniRigKdTree tree = new(points);
        int[] idx = new int[7];
        float[] dis = new float[7];
        for (int q = 0; q < 50; q++)
        {
            float x = (float)random.NextDouble(), y = (float)random.NextDouble(), z = (float)random.NextDouble();
            tree.Nearest(x, y, z, 7, idx, dis);
            int[] expected = Enumerable.Range(0, 500)
                .OrderBy(i => (points[3 * i] - x) * (points[3 * i] - x) + (points[3 * i + 1] - y) * (points[3 * i + 1] - y) + (points[3 * i + 2] - z) * (points[3 * i + 2] - z))
                .Take(7).ToArray();
            Assert.Equal(expected, idx);
            Assert.True(dis.Zip(dis.Skip(1)).All(p => p.First <= p.Second));
        }
    }

    [Fact]
    public void KdTree_FewerPointsThanRequested_LeavesTheTailEmpty()
    {
        UniRigKdTree tree = new([0, 0, 0, 1, 1, 1]);
        int[] idx = new int[4];
        float[] dis = new float[4];
        tree.Nearest(0.1f, 0, 0, 4, idx, dis);
        Assert.Equal([0, 1, -1, -1], idx);
    }

    [Fact]
    public void Interior_OfACube_FillsTheGrid()
    {
        UniRigPreparedMesh cube = Cube();
        List<int> interior = UniRigVoxelSkin.Interior(cube.Vertices, cube.Faces, 12);
        Assert.Equal(12 * 12 * 12, interior.Count / 3);
    }

    [Fact]
    public void Interior_OfATriangleSoup_IsEmptyWithoutAnEnclosedVolume()
    {
        float[] vertices = [-1, -1, 0, 1, -1, 0, 0, 1, 0];
        Assert.Empty(UniRigVoxelSkin.Interior(vertices, [0, 1, 2], 8));
    }

    [Fact]
    public void VoxelSkin_RowsAreProbabilitiesAndFavourTheNearerJoint()
    {
        UniRigPreparedMesh cube = Cube();
        float[] joints = [-0.8f, 0f, 0f, 0.8f, 0f, 0f];
        float[] skin = UniRigVoxelSkin.Compute(cube.Vertices, cube.Faces, joints, new UniRigVoxelSkinOptions { Grid = 16 });
        Assert.Equal(8 * 2, skin.Length);
        for (int v = 0; v < 8; v++)
        {
            Assert.True(skin[2 * v] >= 0f && skin[2 * v + 1] >= 0f);
            Assert.Equal(1f, skin[2 * v] + skin[2 * v + 1], 4);
            bool left = cube.Vertices[3 * v] < 0;
            Assert.True(left ? skin[2 * v] > skin[2 * v + 1] : skin[2 * v + 1] > skin[2 * v], $"vertex {v} prefers the far joint");
        }
    }

    [Fact]
    public void Reskin_UniformPredictionsStayUnchanged_AndRowsSumToOne()
    {
        UniRigPreparedMesh cube = Cube();
        float[] samples = new float[3 * 20];
        float[] skin = new float[2 * 20];
        Random random = new(2);
        for (int i = 0; i < samples.Length; i++)
        {
            samples[i] = (float)(random.NextDouble() * 2 - 1);
        }

        for (int i = 0; i < 20; i++)
        {
            skin[2 * i] = 0.7f;
            skin[2 * i + 1] = 0.3f;
        }

        float[] result = UniRigReskin.Apply(samples, skin, cube.Vertices, cube.Faces, [-1, 0]);
        for (int v = 0; v < 8; v++)
        {
            Assert.Equal(0.7f, result[2 * v], 4);
            Assert.Equal(0.3f, result[2 * v + 1], 4);
        }
    }

    [Fact]
    public void Reskin_DropsSmallWeightsWhenALargerOneExists()
    {
        UniRigPreparedMesh cube = Cube();
        float[] samples = new float[3 * 10];
        float[] skin = new float[3 * 10];
        for (int i = 0; i < 10; i++)
        {
            skin[3 * i] = 0.02f;
            skin[3 * i + 1] = 0.58f;
            skin[3 * i + 2] = 0.40f;
        }

        float[] result = UniRigReskin.Apply(samples, skin, cube.Vertices, cube.Faces, [-1, -1, -1]);
        for (int v = 0; v < 8; v++)
        {
            Assert.Equal(0f, result[3 * v]);
            Assert.Equal(1f, result[3 * v + 1] + result[3 * v + 2], 4);
        }
    }

    [Fact]
    public void Limit_KeepsTheLargestWeightsAndRenormalizes()
    {
        UniRigSkinResult skin = new() { Weights = [0.1f, 0.4f, 0.2f, 0.1f, 0.1f, 0.1f, 1f, 0f, 0f, 0f, 0f, 0f], VertexCount = 2, JointCount = 6 };
        (int[] joints, float[] weights) = skin.Limit(2);
        Assert.Equal([1, 2], joints[..2]);
        Assert.Equal(0.4f / 0.6f, weights[0], 5);
        Assert.Equal(0.2f / 0.6f, weights[1], 5);
        Assert.Equal(0, joints[2]);
        Assert.Equal(1f, weights[2]);
        Assert.Equal(0f, weights[3]);
    }

    [Fact]
    public void Reframe_PutsMeshAndJointsInsideTheUnitCube()
    {
        UniRigPreparedMesh cube = Cube();
        (UniRigPreparedMesh framed, float[] joints) = UniRigSkinner.Reframe(cube, [0f, 3f, 0f]);
        Assert.All(framed.Vertices, v => Assert.InRange(v, -1.0001f, 1.0001f));
        Assert.All(joints, v => Assert.InRange(v, -1.0001f, 1.0001f));
        Assert.Equal(1f, MathF.Max(joints.Max(), framed.Vertices.Max()), 4);
    }
}
