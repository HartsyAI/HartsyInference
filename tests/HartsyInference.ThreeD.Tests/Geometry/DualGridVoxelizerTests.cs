using System.Numerics;
using HartsyInference.ThreeD.Geometry;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Geometry;

public sealed class DualGridVoxelizerTests
{
    private const float Tolerance = 1e-4f;

    // Looser than Tolerance: boundary and plane quadrics accumulate float error across several solves.
    private const float SolveTolerance = 1e-3f;

    private static readonly Vector3 UnitVoxel = Vector3.One;
    private static readonly Vector3 AnisotropicVoxel = new(0.5f, 1f, 2f);
    private static readonly int[] GridMin = [0, 0, 0];
    private static readonly int[] GridMax = [5, 5, 5];

    // Covers the tilted quad at AnisotropicVoxel: x indices reach 7, y 3, z 2.
    private static readonly int[] AnisotropicGridMax = [10, 5, 5];

    // Off-center cube so no face sits exactly on a voxel boundary.
    private static readonly Vector3 CubeMin = new(1.25f, 1.25f, 1.25f);
    private static readonly Vector3 CubeMax = new(3.75f, 3.75f, 3.75f);

    [Fact]
    public void AxisAlignedCube_EveryDualVertexLiesInsideItsVoxel()
    {
        AssertInsideVoxels(VoxelizeCube(), UnitVoxel);
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
    public void OpenSheet_BoundaryPass_PullsEdgeVoxelsOntoTheirBoundaryLines()
    {
        // Boundary lines are y = 1.25 and y = 3.75 (running along x), plus x = 1.25 and x = 3.75 (along y).
        DualGridVoxels voxels = VoxelizeWith(vertices: SheetVertices(), faces: SheetFaces(), boundaryWeight: 1f);

        AssertInsideVoxels(voxels, UnitVoxel);
        int count = voxels.Coords.Length / 3;
        int onBoundaryLine = 0;
        for (int i = 0; i < count; i++)
        {
            Assert.InRange(voxels.DualVertices[i * 3 + 2], 2.5f - SolveTolerance, 2.5f + SolveTolerance);

            int cy = voxels.Coords[i * 3 + 1];
            float y = voxels.DualVertices[i * 3 + 1];
            if (cy == 1)
            {
                Assert.InRange(y, 1.25f - SolveTolerance, 1.25f + SolveTolerance);
                onBoundaryLine++;
            }
            else if (cy == 3)
            {
                Assert.InRange(y, 3.75f - SolveTolerance, 3.75f + SolveTolerance);
                onBoundaryLine++;
            }
        }

        Assert.True(onBoundaryLine > 0, "No edge voxel was constrained by the boundary pass.");
    }

    [Fact]
    public void TiltedQuad_Regularization_MovesRankDeficientVoxelsAndKeepsThemOnThePlane()
    {
        // Without regularization the face quadrics leave the in-plane position undetermined.
        DualGridVoxels plain = VoxelizeWith(vertices: TiltedVertices(), faces: SheetFaces(), boundaryWeight: 0f);
        DualGridVoxels regularized = VoxelizeWith(
            vertices: TiltedVertices(), faces: SheetFaces(), boundaryWeight: 0f, regularizationWeight: 1f);

        AssertInsideVoxels(regularized, UnitVoxel);
        AssertOnTiltedPlane(regularized);
        Assert.True(
            MaxAbsDifference(plain.DualVertices, regularized.DualVertices) > SolveTolerance,
            "Regularization left every dual vertex unchanged.");
    }

    [Fact]
    public void TiltedQuad_NonUnitVoxels_DualVerticesLieInsideTheirVoxelsAndOnThePlane()
    {
        DualGridVoxels voxels = VoxelizeWith(
            vertices: TiltedVertices(),
            faces: SheetFaces(),
            voxelSize: AnisotropicVoxel,
            gridMax: AnisotropicGridMax,
            boundaryWeight: 0f);

        AssertInsideVoxels(voxels, AnisotropicVoxel);
        AssertOnTiltedPlane(voxels);
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
        int[]? gridMax = null,
        float faceWeight = 1f,
        float boundaryWeight = 1f,
        float regularizationWeight = 0f)
    {
        return DualGridVoxelizer.Voxelize(
            vertices ?? CubeVertices(),
            faces ?? CubeFaces(),
            voxelSize ?? UnitVoxel,
            gridMin ?? GridMin,
            gridMax ?? GridMax,
            faceWeight,
            boundaryWeight,
            regularizationWeight);
    }

    private static void AssertInsideVoxels(DualGridVoxels voxels, Vector3 voxelSize)
    {
        int count = voxels.Coords.Length / 3;

        Assert.True(count > 0);
        Assert.Equal(count * 3, voxels.DualVertices.Length);
        Assert.Equal(count * 3, voxels.Intersected.Length);
        for (int i = 0; i < count; i++)
        {
            for (int axis = 0; axis < 3; axis++)
            {
                float size = AxisValue(voxelSize, axis);
                float low = voxels.Coords[i * 3 + axis] * size;
                Assert.InRange(voxels.DualVertices[i * 3 + axis], low - Tolerance, low + size + Tolerance);
            }
        }
    }

    private static void AssertOnTiltedPlane(DualGridVoxels voxels)
    {
        int count = voxels.DualVertices.Length / 3;
        for (int i = 0; i < count; i++)
        {
            Vector3 p = new(voxels.DualVertices[i * 3], voxels.DualVertices[i * 3 + 1], voxels.DualVertices[i * 3 + 2]);
            float distance = DistanceToTiltedPlane(p);
            Assert.True(distance <= SolveTolerance, $"Dual vertex {i} at {p} is {distance} from the quad plane.");
        }
    }

    private static float MaxAbsDifference(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float max = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            max = MathF.Max(max, MathF.Abs(a[i] - b[i]));
        }

        return max;
    }

    private static float AxisValue(Vector3 v, int axis) => axis switch
    {
        0 => v.X,
        1 => v.Y,
        _ => v.Z,
    };

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

    // Open sheet in the plane z = 2.5 over the cube footprint: two triangles, four boundary edges.
    private static float[] SheetVertices() =>
    [
        1.25f, 1.25f, 2.5f,
        3.75f, 1.25f, 2.5f,
        3.75f, 3.75f, 2.5f,
        1.25f, 3.75f, 2.5f,
    ];

    private static int[] SheetFaces() => [0, 1, 2, 0, 2, 3];

    // Same footprint as the sheet, lifted onto the tilted plane z = 0.5x + 0.25y + 1.75.
    private static float[] TiltedVertices() =>
    [
        1.25f, 1.25f, 2.6875f,
        3.75f, 1.25f, 3.9375f,
        3.75f, 3.75f, 4.5625f,
        1.25f, 3.75f, 3.3125f,
    ];

    private static float DistanceToTiltedPlane(Vector3 p) =>
        MathF.Abs(p.Z - 0.5f * p.X - 0.25f * p.Y - 1.75f) / MathF.Sqrt(1.3125f);

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
