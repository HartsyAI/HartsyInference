using HartsyInference.ThreeD.Models.Sam3dObjects;

namespace HartsyInference.ThreeD.Tests.Sam3dObjects;

public sealed class SamSparseFeatures2MeshTests
{
    private static readonly int[] CubeEdges = [0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6, 2, 0, 3, 1, 7, 5, 6, 4];

    // A stand-in for the real tables (which the test project does not carry): one dual vertex per connected group of inside corners, averaging the edges that leave the group, and no
    // ambiguity resolution. It is the plain dual-contouring reading of a cube; the real tables resolve some of the ambiguous cases differently.
    private static SamFlexiCubesTables ComponentTables()
    {
        const int maxVertices = 4;
        int[] dmc = new int[256 * maxVertices * 7];
        Array.Fill(dmc, -1);
        int[] numVd = new int[256];
        for (int c = 0; c < 256; c++)
        {
            bool[] seen = new bool[8];
            int groups = 0;
            for (int start = 0; start < 8; start++)
            {
                if (((c >> start) & 1) == 0 || seen[start])
                    continue;
                List<int> members = [start];
                seen[start] = true;
                for (int q = 0; q < members.Count; q++)
                {
                    for (int e = 0; e < 12; e++)
                    {
                        int a = CubeEdges[e * 2], b = CubeEdges[e * 2 + 1];
                        int other = a == members[q] ? b : b == members[q] ? a : -1;
                        if (other >= 0 && ((c >> other) & 1) == 1 && !seen[other])
                        {
                            seen[other] = true;
                            members.Add(other);
                        }
                    }
                }

                int slot = 0;
                for (int e = 0; e < 12; e++)
                {
                    bool inA = members.Contains(CubeEdges[e * 2]), inB = members.Contains(CubeEdges[e * 2 + 1]);
                    if (inA != inB)
                        dmc[(c * maxVertices + groups) * 7 + slot++] = e;
                }

                Assert.True(slot <= 7);
                if (slot > 0)
                    groups++;
            }

            numVd[c] = groups;
        }

        int[] check = new int[256 * 5];
        return new SamFlexiCubesTables(dmc, maxVertices, numVd, check);
    }

    // The head output for every voxel of a shell around a sphere: the signed distance at each cube corner (the 1/res bias is added by the extraction), zero deformation, zero weights, zero colour.
    private static (int[] Coords, float[] Features) SphereFeatures(int res, double radius)
    {
        List<int> coords = [];
        List<float> features = [];
        double Sdf(int x, int y, int z) => Math.Sqrt(Math.Pow((double)x / res - 0.5, 2) + Math.Pow((double)y / res - 0.5, 2) + Math.Pow((double)z / res - 0.5, 2)) - radius;
        int[][] offsets = SamMeshCornerGrid.CornerOffsets;
        for (int x = 0; x < res; x++)
        {
            for (int y = 0; y < res; y++)
            {
                for (int z = 0; z < res; z++)
                {
                    double min = double.MaxValue;
                    float[] row = new float[101];
                    for (int j = 0; j < 8; j++)
                    {
                        double d = Sdf(x + offsets[j][0], y + offsets[j][1], z + offsets[j][2]);
                        row[j] = (float)d;
                        min = Math.Min(min, d);
                    }

                    if (min > 3.0 / res)
                        continue;
                    coords.AddRange([0, x, y, z]);
                    features.AddRange(row);
                }
            }
        }

        return ([.. coords], [.. features]);
    }

    [Fact]
    public void Extract_ASphereGivesAClosedManifoldMeshOnTheSphere()
    {
        const int res = 16;
        (int[] coords, float[] features) = SphereFeatures(res, 0.3);

        SamMeshResult mesh = SamSparseFeatures2Mesh.Extract(coords, features, ComponentTables(), res);

        Assert.True(mesh.FaceCount > 100);
        // Every undirected edge borders exactly two triangles, and no triangle is degenerate.
        Dictionary<(int, int), int> edgeUses = [];
        for (int f = 0; f < mesh.FaceCount; f++)
        {
            int[] v = [mesh.Faces[f * 3], mesh.Faces[f * 3 + 1], mesh.Faces[f * 3 + 2]];
            Assert.Equal(3, v.Distinct().Count());
            for (int k = 0; k < 3; k++)
            {
                (int, int) edge = (Math.Min(v[k], v[(k + 1) % 3]), Math.Max(v[k], v[(k + 1) % 3]));
                edgeUses[edge] = edgeUses.GetValueOrDefault(edge) + 1;
            }
        }

        Assert.All(edgeUses, pair => Assert.Equal(2, pair.Value));
        // A sphere: Euler characteristic 2, vertices at the radius (the bias of -1/res moves the surface out by one voxel at most).
        Assert.Equal(2, mesh.VertexCount - edgeUses.Count + mesh.FaceCount);
        for (int v = 0; v < mesh.VertexCount; v++)
        {
            double r = Math.Sqrt(mesh.Vertices[v * 3] * mesh.Vertices[v * 3] + mesh.Vertices[v * 3 + 1] * mesh.Vertices[v * 3 + 1] + mesh.Vertices[v * 3 + 2] * mesh.Vertices[v * 3 + 2]);
            Assert.InRange(r, 0.3 - 2.0 / res, 0.3 + 2.0 / res);
        }

        // Outward-facing triangles: the normal of every face points away from the centre.
        for (int f = 0; f < mesh.FaceCount; f++)
        {
            double[][] p = new double[3][];
            for (int k = 0; k < 3; k++)
                p[k] = [mesh.Vertices[mesh.Faces[f * 3 + k] * 3], mesh.Vertices[mesh.Faces[f * 3 + k] * 3 + 1], mesh.Vertices[mesh.Faces[f * 3 + k] * 3 + 2]];
            double[] a = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]], b = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
            double[] n = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
            double dot = n[0] * (p[0][0] + p[1][0] + p[2][0]) + n[1] * (p[0][1] + p[1][1] + p[2][1]) + n[2] * (p[0][2] + p[1][2] + p[2][2]);
            Assert.True(dot > 0, $"face {f} points inwards");
        }
    }

    [Fact]
    public void Extract_ColoursAreTheSigmoidOfTheCornerColoursAndComeBackPerVertex()
    {
        const int res = 8;
        (int[] coords, float[] features) = SphereFeatures(res, 0.3);
        for (int i = 0; i < coords.Length / 4; i++)
        {
            for (int k = 0; k < 48; k++)
                features[i * 101 + 53 + k] = 0f;
        }

        SamMeshResult mesh = SamSparseFeatures2Mesh.Extract(coords, features, ComponentTables(), res);

        Assert.Equal(mesh.VertexCount * 6, mesh.Colors.Length);
        Assert.All(mesh.Colors, c => Assert.Equal(0.5f, c, 4));
    }

    [Fact]
    public void Extract_ANoSurfaceGivesAnEmptyMeshAndMismatchedArraysAreRejected()
    {
        SamFlexiCubesTables tables = ComponentTables();
        float[] outside = new float[101];
        Array.Fill(outside, 1f, 0, 8);

        SamMeshResult empty = SamSparseFeatures2Mesh.Extract([0, 3, 3, 3], outside, tables, 16);

        Assert.Equal(0, empty.VertexCount);
        Assert.Equal(0, empty.FaceCount);
        Assert.Throws<ArgumentException>(() => SamSparseFeatures2Mesh.Extract([0, 3, 3, 3], new float[100], tables, 16));
        Assert.Throws<ArgumentException>(() => SamSparseFeatures2Mesh.Extract([0, 16, 3, 3], outside, tables, 16));
    }

    [Fact]
    public void Tables_RejectTheWrongShapes()
    {
        Assert.Throws<ArgumentException>(() => new SamFlexiCubesTables(new int[10], 4, new int[256], new int[1280]));
        Assert.Throws<ArgumentException>(() => new SamFlexiCubesTables(new int[256 * 4 * 7], 4, new int[255], new int[1280]));
        Assert.Throws<ArgumentException>(() => new SamFlexiCubesTables(new int[256 * 4 * 7], 4, Enumerable.Repeat(5, 256).ToArray(), new int[1280]));
    }

    [Fact]
    public void Tables_RejectValuesOutOfRange()
    {
        int[] Valid() => [.. ComponentTables().Dmc];
        int[] numVd = [.. ComponentTables().NumVd];
        int[] check = new int[1280];
        int[] bad = Valid();
        bad[2] = 12;
        Assert.Throws<ArgumentException>(() => new SamFlexiCubesTables(bad, 4, numVd, check));
        int[] flag = new int[1280];
        flag[5 * 3] = 2;
        Assert.Throws<ArgumentException>(() => new SamFlexiCubesTables(Valid(), 4, numVd, flag));
        int[] replacement = new int[1280];
        replacement[7 * 5 + 4] = 256;
        Assert.Throws<ArgumentException>(() => new SamFlexiCubesTables(Valid(), 4, numVd, replacement));
        // A case that emits a vertex but lists no edge for it.
        int[] empty = Valid();
        Array.Fill(empty, -1, 1 * 4 * 7, 7);
        Assert.Throws<ArgumentException>(() => new SamFlexiCubesTables(empty, 4, numVd, check));
    }

    [Fact]
    public void ResolveAmbiguousCases_SwapsCasesAgainstTheCasesAsTheyWereAndLeavesLoneOrOutOfRangeOnesAlone()
    {
        // Case 1 is flagged, looks at the cube at +x and becomes case 10; case 2 is flagged, looks at -x and becomes case 20.
        int[] check = new int[1280];
        check[1 * 5] = 1;
        check[1 * 5 + 1] = 1;
        check[1 * 5 + 4] = 10;
        check[2 * 5] = 1;
        check[2 * 5 + 1] = -1;
        check[2 * 5 + 4] = 20;
        SamFlexiCubesTables tables = new(new int[256 * 4 * 7], 4, new int[256], check);
        // A (2,2,2) case 1 and B (3,2,2) case 2 look at each other; C (5,5,5) case 1 looks at (6,5,5), which is not a cube here; D (0,0,0) case 2 looks at x = -1, outside the grid.
        long[] keys = [SamMeshCornerGrid.Pack(2, 2, 2), SamMeshCornerGrid.Pack(3, 2, 2), SamMeshCornerGrid.Pack(5, 5, 5), SamMeshCornerGrid.Pack(0, 0, 0)];
        int[] cases = [1, 2, 1, 2];

        SamSparseFeatures2Mesh.ResolveAmbiguousCases(keys, [0, 1, 2, 3], cases, tables, 16);

        // Both of the pair are replaced (B is checked against A as A was, not as A became); the others are not.
        Assert.Equal([10, 20, 1, 2], cases);
    }

    [Fact]
    public void Extract_InterpolatesTheColoursAlongTheEdgesItCrosses()
    {
        const int res = 8;
        (int[] coords, float[] features) = SphereFeatures(res, 0.3);
        for (int i = 0; i < coords.Length / 4; i++)
        {
            // Colour channel 0 of every corner grows with x, so the sigmoid colour of a vertex grows with its x.
            for (int j = 0; j < 8; j++)
                features[i * 101 + 53 + j * 6] = 4f * ((coords[i * 4 + 1] + SamMeshCornerGrid.CornerOffsets[j][0]) / (float)res - 0.5f) * 2f;
        }

        SamMeshResult mesh = SamSparseFeatures2Mesh.Extract(coords, features, ComponentTables(), res);

        for (int v = 0; v < mesh.VertexCount; v++)
        {
            double expectedLogit = 4.0 * (mesh.Vertices[v * 3] + 0.5 - 0.5) * 2.0;
            double colour = mesh.Colors[v * 6];
            // The mix is of sigmoids of corner values, so it lies between the sigmoids at the voxel's x extremes and grows with x.
            Assert.InRange(colour, 0.0, 1.0);
            Assert.True(Math.Abs(colour - 1.0 / (1.0 + Math.Exp(-expectedLogit))) < 0.2, $"vertex {v}: colour {colour}");
            Assert.Equal(0.5f, mesh.Colors[v * 6 + 1], 4);
        }

        Assert.True(Enumerable.Range(0, mesh.VertexCount).Select(v => mesh.Colors[v * 6]).Distinct().Count() > 10);
    }

    [Fact]
    public void Extract_RejectsAVoxelOfAnotherBatchAndTheLatticeRejectsNegativeCoordinates()
    {
        float[] outside = new float[101];
        Array.Fill(outside, 1f, 0, 8);
        Assert.Throws<ArgumentException>(() => SamSparseFeatures2Mesh.Extract([1, 3, 3, 3], outside, ComponentTables(), 16));
        Assert.Throws<ArgumentException>(() => SamMeshCornerGrid.Build([0, -1, 0], new float[8], 1));
    }
}
