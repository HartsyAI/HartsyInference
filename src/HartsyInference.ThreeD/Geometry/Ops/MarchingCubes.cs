using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>Classic Marching Cubes (Lorensen &amp; Cline / Paul Bourke tables): extracts a triangle
/// <see cref="Mesh"/> from a <see cref="ScalarField3D"/> at a given iso level. Pure managed C# — mesh
/// extraction is a post-decode step, not a GPU hot path. The reusable mesh extractor for every
/// occupancy/SDF-producing 3D model (Hunyuan3D today; TripoSR/TRELLIS later).
/// <para>Convention: a corner is "inside" when its value is <b>below</b> <c>isoLevel</c>. For an SDF
/// pass <c>isoLevel = 0</c>; for an occupancy/logit field pass the model's threshold. Output winding is
/// CCW for outward-facing normals when inside &lt; outside.</para></summary>
public static class MarchingCubes
{
    /// <summary>Extracts a mesh from <paramref name="field"/> at <paramref name="isoLevel"/>. Vertices are
    /// de-duplicated per shared cube edge within a Z-slab so the result is watertight (no cracks).</summary>
    public static Mesh Extract(ScalarField3D field, float isoLevel = 0f)
    {
        ArgumentNullException.ThrowIfNull(field);
        int nx = field.ResX, ny = field.ResY, nz = field.ResZ;

        List<float> verts = new();
        List<int> tris = new();
        // Edge-vertex cache: map a canonical edge key -> emitted vertex index, so adjacent cells
        // sharing an edge reuse the same vertex (watertight).
        Dictionary<long, int> edgeCache = new();

        Span<float> val = stackalloc float[8];
        Span<int> cornerLin = stackalloc int[8];
        Span<int> edgeVert = stackalloc int[12]; // reused per cell; fully overwritten each iteration
        // Corner offsets (Bourke convention), indices 0..7.
        ReadOnlySpan<int> cox = [0, 1, 1, 0, 0, 1, 1, 0];
        ReadOnlySpan<int> coy = [0, 0, 1, 1, 0, 0, 1, 1];
        ReadOnlySpan<int> coz = [0, 0, 0, 0, 1, 1, 1, 1];

        for (int z = 0; z < nz - 1; z++)
        for (int y = 0; y < ny - 1; y++)
        for (int x = 0; x < nx - 1; x++)
        {
            int cubeIndex = 0;
            for (int c = 0; c < 8; c++)
            {
                int cx = x + cox[c], cy = y + coy[c], cz = z + coz[c];
                cornerLin[c] = field.Index(cx, cy, cz);
                float v = field.Values[cornerLin[c]];
                val[c] = v;
                if (v < isoLevel) cubeIndex |= 1 << c;
            }

            int edges = EdgeTable[cubeIndex];
            if (edges == 0) continue;

            for (int e = 0; e < 12; e++)
            {
                if ((edges & (1 << e)) == 0) { edgeVert[e] = -1; continue; }
                int a = EdgeCorners[e * 2], b = EdgeCorners[e * 2 + 1];
                long key = EdgeKey(x, y, z, cox, coy, coz, a, b, nx, ny);
                if (!edgeCache.TryGetValue(key, out int vi))
                {
                    (float vx, float vy, float vz) = InterpEdge(field, x, y, z, cox, coy, coz, a, b, val[a], val[b], isoLevel);
                    vi = verts.Count / 3;
                    verts.Add(vx); verts.Add(vy); verts.Add(vz);
                    edgeCache[key] = vi;
                }
                edgeVert[e] = vi;
            }

            // Emit with reversed winding: the canonical Bourke table yields inward-facing triangles
            // for the "corner inside when value < iso" convention; reversing makes normals point
            // outward (toward increasing field value, i.e. out of an SDF/occupancy surface).
            for (int t = 0; TriTable[cubeIndex][t] != -1; t += 3)
            {
                tris.Add(edgeVert[TriTable[cubeIndex][t]]);
                tris.Add(edgeVert[TriTable[cubeIndex][t + 2]]);
                tris.Add(edgeVert[TriTable[cubeIndex][t + 1]]);
            }
        }

        return new Mesh { Vertices = [.. verts], Indices = [.. tris] };
    }

    /// <summary>Linearly interpolate the iso-crossing position on the edge between corners a and b.</summary>
    private static (float, float, float) InterpEdge(
        ScalarField3D f, int x, int y, int z,
        ReadOnlySpan<int> cox, ReadOnlySpan<int> coy, ReadOnlySpan<int> coz,
        int a, int b, float va, float vb, float iso)
    {
        (float ax, float ay, float az) = f.WorldOf(x + cox[a], y + coy[a], z + coz[a]);
        (float bx, float by, float bz) = f.WorldOf(x + cox[b], y + coy[b], z + coz[b]);
        float denom = vb - va;
        float t = MathF.Abs(denom) < 1e-12f ? 0.5f : (iso - va) / denom;
        t = Math.Clamp(t, 0f, 1f);
        return (ax + t * (bx - ax), ay + t * (by - ay), az + t * (bz - az));
    }

    /// <summary>Canonical key for the grid edge between two cube corners, independent of which cell
    /// references it — so neighbouring cells share the vertex. Encodes the lower endpoint's grid coord
    /// plus the edge axis.</summary>
    private static long EdgeKey(int x, int y, int z,
        ReadOnlySpan<int> cox, ReadOnlySpan<int> coy, ReadOnlySpan<int> coz,
        int a, int b, int nx, int ny)
    {
        int ax = x + cox[a], ay = y + coy[a], az = z + coz[a];
        int bx = x + cox[b], by = y + coy[b], bz = z + coz[b];
        // Lower endpoint = smaller (z,y,x); axis = which coordinate differs.
        int axis = ax != bx ? 0 : (ay != by ? 1 : 2);
        int lx = Math.Min(ax, bx), ly = Math.Min(ay, by), lz = Math.Min(az, bz);
        long cell = (long)lx + (long)nx * (ly + (long)ny * lz);
        return cell * 3 + axis;
    }

    // Edge e connects corner EdgeCorners[2e] -> EdgeCorners[2e+1] (Bourke convention).
    private static readonly int[] EdgeCorners =
    [
        0, 1,  1, 2,  2, 3,  3, 0,
        4, 5,  5, 6,  6, 7,  7, 4,
        0, 4,  1, 5,  2, 6,  3, 7,
    ];

    private static readonly int[] EdgeTable =
    [
        0x0,0x109,0x203,0x30a,0x406,0x50f,0x605,0x70c,0x80c,0x905,0xa0f,0xb06,0xc0a,0xd03,0xe09,0xf00,
        0x190,0x99,0x393,0x29a,0x596,0x49f,0x795,0x69c,0x99c,0x895,0xb9f,0xa96,0xd9a,0xc93,0xf99,0xe90,
        0x230,0x339,0x33,0x13a,0x636,0x73f,0x435,0x53c,0xa3c,0xb35,0x83f,0x936,0xe3a,0xf33,0xc39,0xd30,
        0x3a0,0x2a9,0x1a3,0xaa,0x7a6,0x6af,0x5a5,0x4ac,0xbac,0xaa5,0x9af,0x8a6,0xfaa,0xea3,0xda9,0xca0,
        0x460,0x569,0x663,0x76a,0x66,0x16f,0x265,0x36c,0xc6c,0xd65,0xe6f,0xf66,0x86a,0x963,0xa69,0xb60,
        0x5f0,0x4f9,0x7f3,0x6fa,0x1f6,0xff,0x3f5,0x2fc,0xdfc,0xcf5,0xfff,0xef6,0x9fa,0x8f3,0xbf9,0xaf0,
        0x650,0x759,0x453,0x55a,0x256,0x35f,0x55,0x15c,0xe5c,0xf55,0xc5f,0xd56,0xa5a,0xb53,0x859,0x950,
        0x7c0,0x6c9,0x5c3,0x4ca,0x3c6,0x2cf,0x1c5,0xcc,0xfcc,0xec5,0xdcf,0xcc6,0xbca,0xac3,0x9c9,0x8c0,
        0x8c0,0x9c9,0xac3,0xbca,0xcc6,0xdcf,0xec5,0xfcc,0xcc,0x1c5,0x2cf,0x3c6,0x4ca,0x5c3,0x6c9,0x7c0,
        0x950,0x859,0xb53,0xa5a,0xd56,0xc5f,0xf55,0xe5c,0x15c,0x55,0x35f,0x256,0x55a,0x453,0x759,0x650,
        0xaf0,0xbf9,0x8f3,0x9fa,0xef6,0xfff,0xcf5,0xdfc,0x2fc,0x3f5,0xff,0x1f6,0x6fa,0x7f3,0x4f9,0x5f0,
        0xb60,0xa69,0x963,0x86a,0xf66,0xe6f,0xd65,0xc6c,0x36c,0x265,0x16f,0x66,0x76a,0x663,0x569,0x460,
        0xca0,0xda9,0xea3,0xfaa,0x8a6,0x9af,0xaa5,0xbac,0x4ac,0x5a5,0x6af,0x7a6,0xaa,0x1a3,0x2a9,0x3a0,
        0xd30,0xc39,0xf33,0xe3a,0x936,0x83f,0xb35,0xa3c,0x53c,0x435,0x73f,0x636,0x13a,0x33,0x339,0x230,
        0xe90,0xf99,0xc93,0xd9a,0xa96,0xb9f,0x895,0x99c,0x69c,0x795,0x49f,0x596,0x29a,0x393,0x99,0x190,
        0xf00,0xe09,0xd03,0xc0a,0xb06,0xa0f,0x905,0x80c,0x70c,0x605,0x50f,0x406,0x30a,0x203,0x109,0x0,
    ];

    // Assigned in the static ctor (not a field initializer) so it runs AFTER TriRows is initialized.
    private static readonly int[][] TriTable;

    static MarchingCubes() => TriTable = BuildTriTable();

    // The 256-row triangle table, terminated by -1 per row. Generated once at startup from the
    // canonical Bourke listing embedded below (flat, -1-separated rows) to keep the source readable.
    private static int[][] BuildTriTable()
    {
        int[][] table = new int[256][];
        int idx = 0;
        foreach (string row in TriRows)
        {
            if (row.Length == 0) { table[idx++] = [-1]; continue; }
            string[] parts = row.Split(',');
            int[] r = new int[parts.Length + 1];
            for (int i = 0; i < parts.Length; i++) r[i] = int.Parse(parts[i]);
            r[parts.Length] = -1;
            table[idx++] = r;
        }
        return table;
    }

    // Canonical triangle table (Paul Bourke). Each line = one cube case (0..255); empty = no triangles.
    private static readonly string[] TriRows =
    [
        "", "0,8,3", "0,1,9", "1,8,3,9,8,1", "1,2,10", "0,8,3,1,2,10", "9,2,10,0,2,9", "2,8,3,2,10,8,10,9,8",
        "3,11,2", "0,11,2,8,11,0", "1,9,0,2,3,11", "1,11,2,1,9,11,9,8,11", "3,10,1,11,10,3", "0,10,1,0,8,10,8,11,10", "3,9,0,3,11,9,11,10,9", "9,8,10,10,8,11",
        "4,7,8", "4,3,0,7,3,4", "0,1,9,8,4,7", "4,1,9,4,7,1,7,3,1", "1,2,10,8,4,7", "3,4,7,3,0,4,1,2,10", "9,2,10,9,0,2,8,4,7", "2,10,9,2,9,7,2,7,3,7,9,4",
        "8,4,7,3,11,2", "11,4,7,11,2,4,2,0,4", "9,0,1,8,4,7,2,3,11", "4,7,11,9,4,11,9,11,2,9,2,1", "3,10,1,3,11,10,7,8,4", "1,11,10,1,4,11,1,0,4,7,11,4", "4,7,8,9,0,11,9,11,10,11,0,3", "4,7,11,4,11,9,9,11,10",
        "9,5,4", "9,5,4,0,8,3", "0,5,4,1,5,0", "8,5,4,8,3,5,3,1,5", "1,2,10,9,5,4", "3,0,8,1,2,10,4,9,5", "5,2,10,5,4,2,4,0,2", "2,10,5,3,2,5,3,5,4,3,4,8",
        "9,5,4,2,3,11", "0,11,2,0,8,11,4,9,5", "0,5,4,0,1,5,2,3,11", "2,1,5,2,5,8,2,8,11,4,8,5", "10,3,11,10,1,3,9,5,4", "4,9,5,0,8,1,8,10,1,8,11,10", "5,4,0,5,0,11,5,11,10,11,0,3", "5,4,8,5,8,10,10,8,11",
        "9,7,8,5,7,9", "9,3,0,9,5,3,5,7,3", "0,7,8,0,1,7,1,5,7", "1,5,3,3,5,7", "9,7,8,9,5,7,10,1,2", "10,1,2,9,5,0,5,3,0,5,7,3", "8,0,2,8,2,5,8,5,7,10,5,2", "2,10,5,2,5,3,3,5,7",
        "7,9,5,7,8,9,3,11,2", "9,5,7,9,7,2,9,2,0,2,7,11", "2,3,11,0,1,8,1,7,8,1,5,7", "11,2,1,11,1,7,7,1,5", "9,5,8,8,5,7,10,1,3,10,3,11", "5,7,0,5,0,9,7,11,0,1,0,10,11,10,0", "11,10,0,11,0,3,10,5,0,8,0,7,5,7,0", "11,10,5,7,11,5",
        "10,6,5", "0,8,3,5,10,6", "9,0,1,5,10,6", "1,8,3,1,9,8,5,10,6", "1,6,5,2,6,1", "1,6,5,1,2,6,3,0,8", "9,6,5,9,0,6,0,2,6", "5,9,8,5,8,2,5,2,6,3,2,8",
        "2,3,11,10,6,5", "11,0,8,11,2,0,10,6,5", "0,1,9,2,3,11,5,10,6", "5,10,6,1,9,2,9,11,2,9,8,11", "6,3,11,6,5,3,5,1,3", "0,8,11,0,11,5,0,5,1,5,11,6", "3,11,6,0,3,6,0,6,5,0,5,9", "6,5,9,6,9,11,11,9,8",
        "5,10,6,4,7,8", "4,3,0,4,7,3,6,5,10", "1,9,0,5,10,6,8,4,7", "10,6,5,1,9,7,1,7,3,7,9,4", "6,1,2,6,5,1,4,7,8", "1,2,5,5,2,6,3,0,4,3,4,7", "8,4,7,9,0,5,0,6,5,0,2,6", "7,3,9,7,9,4,3,2,9,5,9,6,2,6,9",
        "3,11,2,7,8,4,10,6,5", "5,10,6,4,7,2,4,2,0,2,7,11", "0,1,9,4,7,8,2,3,11,5,10,6", "9,2,1,9,11,2,9,4,11,7,11,4,5,10,6", "8,4,7,3,11,5,3,5,1,5,11,6", "5,1,11,5,11,6,1,0,11,7,11,4,0,4,11", "0,5,9,0,6,5,0,3,6,11,6,3,8,4,7", "6,5,9,6,9,11,4,7,9,7,11,9",
        "10,4,9,6,4,10", "4,10,6,4,9,10,0,8,3", "10,0,1,10,6,0,6,4,0", "8,3,1,8,1,6,8,6,4,6,1,10", "1,4,9,1,2,4,2,6,4", "3,0,8,1,2,9,2,4,9,2,6,4", "0,2,4,4,2,6", "8,3,2,8,2,4,4,2,6",
        "10,4,9,10,6,4,11,2,3", "0,8,2,2,8,11,4,9,10,4,10,6", "3,11,2,0,1,6,0,6,4,6,1,10", "6,4,1,6,1,10,4,8,1,2,1,11,8,11,1", "9,6,4,9,3,6,9,1,3,11,6,3", "8,11,1,8,1,0,11,6,1,9,1,4,6,4,1", "3,11,6,3,6,0,0,6,4", "6,4,8,11,6,8",
        "7,10,6,7,8,10,8,9,10", "0,7,3,0,10,7,0,9,10,6,7,10", "10,6,7,1,10,7,1,7,8,1,8,0", "10,6,7,10,7,1,1,7,3", "1,2,6,1,6,8,1,8,9,8,6,7", "2,6,9,2,9,1,6,7,9,0,9,3,7,3,9", "7,8,0,7,0,6,6,0,2", "7,3,2,6,7,2",
        "2,3,11,10,6,8,10,8,9,8,6,7", "2,0,7,2,7,11,0,9,7,6,7,10,9,10,7", "1,8,0,1,7,8,1,10,7,6,7,10,2,3,11", "11,2,1,11,1,7,10,6,1,6,7,1", "8,9,6,8,6,7,9,1,6,11,6,3,1,3,6", "0,9,1,11,6,7", "7,8,0,7,0,6,3,11,0,11,6,0", "7,11,6",
        "7,6,11", "3,0,8,11,7,6", "0,1,9,11,7,6", "8,1,9,8,3,1,11,7,6", "10,1,2,6,11,7", "1,2,10,3,0,8,6,11,7", "2,9,0,2,10,9,6,11,7", "6,11,7,2,10,3,10,8,3,10,9,8",
        "7,2,3,6,2,7", "7,0,8,7,6,0,6,2,0", "2,7,6,2,3,7,0,1,9", "1,6,2,1,8,6,1,9,8,8,7,6", "10,7,6,10,1,7,1,3,7", "10,7,6,1,7,10,1,8,7,1,0,8", "0,3,7,0,7,10,0,10,9,6,10,7", "7,6,10,7,10,8,8,10,9",
        "6,8,4,11,8,6", "3,6,11,3,0,6,0,4,6", "8,6,11,8,4,6,9,0,1", "9,4,6,9,6,3,9,3,1,11,3,6", "6,8,4,6,11,8,2,10,1", "1,2,10,3,0,11,0,6,11,0,4,6", "4,11,8,4,6,11,0,2,9,2,10,9", "10,9,3,10,3,2,9,4,3,11,3,6,4,6,3",
        "8,2,3,8,4,2,4,6,2", "0,4,2,4,6,2", "1,9,0,2,3,4,2,4,6,4,3,8", "1,9,4,1,4,2,2,4,6", "8,1,3,8,6,1,8,4,6,6,10,1", "10,1,0,10,0,6,6,0,4", "4,6,3,4,3,8,6,10,3,0,3,9,10,9,3", "10,9,4,6,10,4",
        "4,9,5,7,6,11", "0,8,3,4,9,5,11,7,6", "5,0,1,5,4,0,7,6,11", "11,7,6,8,3,4,3,5,4,3,1,5", "9,5,4,10,1,2,7,6,11", "6,11,7,1,2,10,0,8,3,4,9,5", "7,6,11,5,4,10,4,2,10,4,0,2", "3,4,8,3,5,4,3,2,5,10,5,2,11,7,6",
        "7,2,3,7,6,2,5,4,9", "9,5,4,0,8,6,0,6,2,6,8,7", "3,6,2,3,7,6,1,5,0,5,4,0", "6,2,8,6,8,7,2,1,8,4,8,5,1,5,8", "9,5,4,10,1,6,1,7,6,1,3,7", "1,6,10,1,7,6,1,0,7,8,7,0,9,5,4", "4,0,10,4,10,5,0,3,10,6,10,7,3,7,10", "7,6,10,7,10,8,5,4,10,4,8,10",
        "6,9,5,6,11,9,11,8,9", "3,6,11,0,6,3,0,5,6,0,9,5", "0,11,8,0,5,11,0,1,5,5,6,11", "6,11,3,6,3,5,5,3,1", "1,2,10,9,5,11,9,11,8,11,5,6", "0,11,3,0,6,11,0,9,6,5,6,9,1,2,10", "11,8,5,11,5,6,8,0,5,10,5,2,0,2,5", "6,11,3,6,3,5,2,10,3,10,5,3",
        "5,8,9,5,2,8,5,6,2,3,8,2", "9,5,6,9,6,0,0,6,2", "1,5,8,1,8,0,5,6,8,3,8,2,6,2,8", "1,5,6,2,1,6", "1,3,6,1,6,10,3,8,6,5,6,9,8,9,6", "10,1,0,10,0,6,9,5,0,5,6,0", "0,3,8,5,6,10", "10,5,6",
        "11,5,10,7,5,11", "11,5,10,11,7,5,8,3,0", "5,11,7,5,10,11,1,9,0", "10,7,5,10,11,7,9,8,1,8,3,1", "11,1,2,11,7,1,7,5,1", "0,8,3,1,2,7,1,7,5,7,2,11", "9,7,5,9,2,7,9,0,2,2,11,7", "7,5,2,7,2,11,5,9,2,3,2,8,9,8,2",
        "2,5,10,2,3,5,3,7,5", "8,2,0,8,5,2,8,7,5,10,2,5", "9,0,1,5,10,3,5,3,7,3,10,2", "9,8,2,9,2,1,8,7,2,10,2,5,7,5,2", "1,3,5,3,7,5", "0,8,7,0,7,1,1,7,5", "9,0,3,9,3,5,5,3,7", "9,8,7,5,9,7",
        "5,8,4,5,10,8,10,11,8", "5,0,4,5,11,0,5,10,11,11,3,0", "0,1,9,8,4,10,8,10,11,10,4,5", "10,11,4,10,4,5,11,3,4,9,4,1,3,1,4", "2,5,1,2,8,5,2,11,8,4,5,8", "0,4,11,0,11,3,4,5,11,2,11,1,5,1,11", "0,2,5,0,5,9,2,11,5,4,5,8,11,8,5", "9,4,5,2,11,3",
        "2,5,10,3,5,2,3,4,5,3,8,4", "5,10,2,5,2,4,4,2,0", "3,10,2,3,5,10,3,8,5,4,5,8,0,1,9", "5,10,2,5,2,4,1,9,2,9,4,2", "8,4,5,8,5,3,3,5,1", "0,4,5,1,0,5", "8,4,5,8,5,3,9,0,5,0,3,5", "9,4,5",
        "4,11,7,4,9,11,9,10,11", "0,8,3,4,9,7,9,11,7,9,10,11", "1,10,11,1,11,4,1,4,0,7,4,11", "3,1,4,3,4,8,1,10,4,7,4,11,10,11,4", "4,11,7,9,11,4,9,2,11,9,1,2", "9,7,4,9,11,7,9,1,11,2,11,1,0,8,3", "11,7,4,11,4,2,2,4,0", "11,7,4,11,4,2,8,3,4,3,2,4",
        "2,9,10,2,7,9,2,3,7,7,4,9", "9,10,7,9,7,4,10,2,7,8,7,0,2,0,7", "3,7,10,3,10,2,7,4,10,1,10,0,4,0,10", "1,10,2,8,7,4", "4,9,1,4,1,7,7,1,3", "4,9,1,4,1,7,0,8,1,8,7,1", "4,0,3,7,4,3", "4,8,7",
        "9,10,8,10,11,8", "3,0,9,3,9,11,11,9,10", "0,1,10,0,10,8,8,10,11", "3,1,10,11,3,10", "1,2,11,1,11,9,9,11,8", "3,0,9,3,9,11,1,2,9,2,11,9", "0,2,11,8,0,11", "3,2,11",
        "2,3,8,2,8,10,10,8,9", "9,10,2,0,9,2", "2,3,8,2,8,10,0,1,8,1,10,8", "1,10,2", "1,3,8,9,1,8", "0,9,1", "0,3,8", "",
    ];
}
