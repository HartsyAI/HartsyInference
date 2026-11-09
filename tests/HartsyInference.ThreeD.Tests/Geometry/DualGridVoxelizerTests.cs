using System.Numerics;
using HartsyInference.ThreeD.Geometry;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Geometry;

public sealed class DualGridVoxelizerTests
{
    private const float Tolerance = 1e-4f;

    private static readonly Vector3 UnitVoxel = Vector3.One;
    private static readonly int[] GridMin = [0, 0, 0];
    private static readonly int[] GridMax = [5, 5, 5];

    // Off-center cube so no face sits exactly on a voxel boundary.
    private static readonly Vector3 CubeMin = new(1.25f, 1.25f, 1.25f);
    private static readonly Vector3 CubeMax = new(3.75f, 3.75f, 3.75f);

    [Fact]
    public void AxisAlignedCube_EveryDualVertexLiesInsideItsVoxel()
    {
        DualGridVoxels voxels = VoxelizeCube();
        int count = voxels.Coords.Length / 3;

        Assert.True(count > 0);
        Assert.Equal(count * 3, voxels.DualVertices.Length);
        Assert.Equal(count * 3, voxels.Intersected.Length);
        for (int i = 0; i < count; i++)
        {
            for (int axis = 0; axis < 3; axis++)
            {
                float low = voxels.Coords[i * 3 + axis] * UnitVoxel.X;
                float high = low + UnitVoxel.X;
                float value = voxels.DualVertices[i * 3 + axis];
                Assert.InRange(value, low - Tolerance, high + Tolerance);
            }
        }
    }

    [Fact]
    public void AxisAlignedCube_RoundTripThroughMesherStaysNearSurface()
    {
        DualGridVoxels voxels = VoxelizeCube();
        int count = voxels.Coords.Length / 3;
        float[] offsets = voxels.ToVoxelOffsets(UnitVoxel);

        DualGridMesh mesh = DualGridMesher.Extract(
            voxels.Coords, offsets, voxels.Intersected, null, Vector3.Zero, UnitVoxel);

        int vertexCount = mesh.Vertices.Length / 3;
        Assert.NotEmpty(mesh.Triangles);
        Assert.Equal(0, mesh.Triangles.Length % 3);
        Assert.Equal(count, vertexCount);
        Assert.All(mesh.Triangles, index => Assert.InRange(index, 0, vertexCount - 1));

        for (int v = 0; v < vertexCount; v++)
        {
            Vector3 point = new(mesh.Vertices[v * 3], mesh.Vertices[v * 3 + 1], mesh.Vertices[v * 3 + 2]);
            Assert.True(
                DistanceToCubeSurface(point) <= UnitVoxel.X + Tolerance,
                $"Vertex {v} at {point} is more than one voxel from the cube surface.");
        }
    }

    [Fact]
    public void Voxelize_IsDeterministic()
    {
        DualGridVoxels first = VoxelizeCube();
        DualGridVoxels second = VoxelizeCube();

        Assert.Equal(first.Coords, second.Coords);
        Assert.Equal(first.DualVertices, second.DualVertices);
        Assert.Equal(first.Intersected, second.Intersected);
    }

    [Fact]
    public void EmptyMesh_ReturnsEmptyDualGrid()
    {
        DualGridVoxels voxels = DualGridVoxelizer.Voxelize(
            [], [], UnitVoxel, GridMin, GridMax, faceWeight: 1f, boundaryWeight: 1f, regularizationWeight: 0f);

        Assert.Empty(voxels.Coords);
        Assert.Empty(voxels.DualVertices);
        Assert.Empty(voxels.Intersected);
    }

    [Fact]
    public void NullVertices_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => DualGridVoxelizer.Voxelize(
            null!, CubeFaces(), UnitVoxel, GridMin, GridMax, faceWeight: 1f, boundaryWeight: 1f, regularizationWeight: 0f));
    }

    [Fact]
    public void NullFaces_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => DualGridVoxelizer.Voxelize(
            CubeVertices(), null!, UnitVoxel, GridMin, GridMax, faceWeight: 1f, boundaryWeight: 1f, regularizationWeight: 0f));
    }

    [Fact]
    public void VertexArrayNotMultipleOfThree_Throws()
    {
        Assert.Throws<ArgumentException>(() => VoxelizeWith(vertices: [0f, 0f, 0f, 1f]));
    }

    [Fact]
    public void FaceArrayNotMultipleOfThree_Throws()
    {
        Assert.Throws<ArgumentException>(() => VoxelizeWith(faces: [0, 1, 2, 0]));
    }

    [Fact]
    public void FaceIndexAtVertexCount_Throws()
    {
        // Three vertices only, so index 3 is one past the end.
        Assert.Throws<ArgumentException>(() => VoxelizeWith(vertices: [0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 0f], faces: [0, 1, 3]));
    }

    [Fact]
    public void NegativeFaceIndex_Throws()
    {
        Assert.Throws<ArgumentException>(() => VoxelizeWith(faces: [0, 1, -1]));
    }

    [Fact]
    public void GridMinWrongLength_Throws()
    {
        Assert.Throws<ArgumentException>(() => VoxelizeWith(gridMin: [0, 0]));
    }

    [Fact]
    public void GridMaxNotAboveGridMin_Throws()
    {
        Assert.Throws<ArgumentException>(() => VoxelizeWith(gridMax: [0, 5, 5]));
    }

    [Fact]
    public void NonPositiveVoxelSize_Throws()
    {
        Assert.Throws<ArgumentException>(() => VoxelizeWith(voxelSize: new Vector3(1f, 0f, 1f)));
    }

    [Fact]
    public void NonFiniteVertex_Throws()
    {
        float[] vertices = CubeVertices();
        vertices[4] = float.NaN;
        Assert.Throws<ArgumentException>(() => VoxelizeWith(vertices: vertices));
    }

    private static DualGridVoxels VoxelizeCube() => VoxelizeWith();

    private static DualGridVoxels VoxelizeWith(
        float[]? vertices = null,
        int[]? faces = null,
        Vector3? voxelSize = null,
        int[]? gridMin = null,
        int[]? gridMax = null)
    {
        return DualGridVoxelizer.Voxelize(
            vertices ?? CubeVertices(),
            faces ?? CubeFaces(),
            voxelSize ?? UnitVoxel,
            gridMin ?? GridMin,
            gridMax ?? GridMax,
            faceWeight: 1f,
            boundaryWeight: 1f,
            regularizationWeight: 0f);
    }

    private static float[] CubeVertices() =>
    [
        CubeMin.X, CubeMin.Y, CubeMin.Z,
        CubeMax.X, CubeMin.Y, CubeMin.Z,
        CubeMax.X, CubeMax.Y, CubeMin.Z,
        CubeMin.X, CubeMax.Y, CubeMin.Z,
        CubeMin.X, CubeMin.Y, CubeMax.Z,
        CubeMax.X, CubeMin.Y, CubeMax.Z,
        CubeMax.X, CubeMax.Y, CubeMax.Z,
        CubeMin.X, CubeMax.Y, CubeMax.Z,
    ];

    // Outward-facing triangles, two per cube face.
    private static int[] CubeFaces() =>
    [
        0, 2, 1, 0, 3, 2,
        4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4,
        3, 7, 6, 3, 6, 2,
        0, 4, 7, 0, 7, 3,
        1, 2, 6, 1, 6, 5,
    ];

    private static float DistanceToCubeSurface(Vector3 p)
    {
        bool inside = p.X >= CubeMin.X && p.X <= CubeMax.X
            && p.Y >= CubeMin.Y && p.Y <= CubeMax.Y
            && p.Z >= CubeMin.Z && p.Z <= CubeMax.Z;
        if (!inside)
        {
            Vector3 outside = Vector3.Max(CubeMin - p, Vector3.Zero);
            outside = Vector3.Max(outside, p - CubeMax);
            return outside.Length();
        }

        float dx = MathF.Min(p.X - CubeMin.X, CubeMax.X - p.X);
        float dy = MathF.Min(p.Y - CubeMin.Y, CubeMax.Y - p.Y);
        float dz = MathF.Min(p.Z - CubeMin.Z, CubeMax.Z - p.Z);
        return MathF.Min(dx, MathF.Min(dy, dz));
    }
}
