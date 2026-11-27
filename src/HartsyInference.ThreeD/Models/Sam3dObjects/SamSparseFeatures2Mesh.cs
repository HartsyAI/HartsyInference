namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>A mesh from the FlexiCubes extraction: vertices <c>[V, 3]</c> in the unit cube centred on the origin, triangles <c>[F, 3]</c>, and 6 colour channels per vertex (RGB and the normal map).</summary>
public sealed record SamMeshResult(float[] Vertices, int[] Faces, float[] Colors)
{
    /// <summary>Number of vertices.</summary>
    public int VertexCount => Vertices.Length / 3;

    /// <summary>Number of triangles.</summary>
    public int FaceCount => Faces.Length / 3;
}

/// <summary>
/// The extraction of SAM 3D's mesh decoder (upstream's <c>SparseFeatures2Mesh</c> with <c>FlexiCubes</c> in inference mode) on the sparse features of <see cref="SlatMeshDecoder"/>. Upstream
/// builds dense <c>(res + 1)^3</c> grids; this keeps only what they hold that matters: the corner lattice of the active voxels, and the cubes around its inside corners (a cube with no active
/// corner is all outside and emits nothing). The same cubes, vertices and triangles come out in the same order, so a mesh can be compared with upstream's one for one.
/// </summary>
public static class SamSparseFeatures2Mesh
{
    private const int Res = 256;

    // cube_edges of FlexiCubes: the corner pair of each of the 12 edges.
    private static readonly int[] CubeEdges = [0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6, 2, 0, 3, 1, 7, 5, 6, 4];
    private static readonly int[] QuadSplit1 = [0, 1, 2, 0, 2, 3];
    private static readonly int[] QuadSplit2 = [0, 1, 3, 3, 1, 2];

    /// <summary>
    /// Extracts the mesh. <paramref name="coords"/> are the voxels <c>(batch, x, y, z)</c> on the <paramref name="resolution"/>-grid (batch 0), <paramref name="features"/> the
    /// <c>[M, 101]</c> head output (8 SDF, 24 deformation, 21 weights, 48 colour per voxel).
    /// </summary>
    /// <exception cref="ArgumentException">The arrays disagree in length, or a voxel lies outside the grid.</exception>
    public static SamMeshResult Extract(int[] coords, float[] features, SamFlexiCubesTables tables, int resolution = Res)
    {
        ArgumentNullException.ThrowIfNull(coords);
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(tables);
        const int channels = SlatMeshDecoder.OutChannels;
        if (coords.Length % 4 != 0 || features.Length != (long)coords.Length / 4 * channels)
            throw new ArgumentException($"{coords.Length / 4} voxels carry {(long)coords.Length / 4 * channels} features; got {features.Length}.", nameof(features));
        int m = coords.Length / 4;
        float sdfBias = -1f / resolution;

        // Per cube corner: SDF (with the bias), deformation, colour; averaged where cubes share a corner.
        int[] cubes = new int[m * 3];
        float[] cornerValues = new float[(long)m * 8 * 10];
        for (int i = 0; i < m; i++)
        {
            if (coords[i * 4] != 0)
                throw new ArgumentException($"Voxel {i} is in batch {coords[i * 4]}; the extraction takes one mesh at a time (batch 0).", nameof(coords));
            cubes[i * 3] = coords[i * 4 + 1];
            cubes[i * 3 + 1] = coords[i * 4 + 2];
            cubes[i * 3 + 2] = coords[i * 4 + 3];
            if (cubes[i * 3] < 0 || cubes[i * 3] >= resolution || cubes[i * 3 + 1] < 0 || cubes[i * 3 + 1] >= resolution || cubes[i * 3 + 2] < 0 || cubes[i * 3 + 2] >= resolution)
                throw new ArgumentException($"Voxel {i} lies outside the {resolution}-grid.", nameof(coords));
            long source = (long)i * channels;
            for (int j = 0; j < 8; j++)
            {
                long target = ((long)i * 8 + j) * 10;
                cornerValues[target] = features[source + j] + sdfBias;
                for (int k = 0; k < 3; k++)
                    cornerValues[target + 1 + k] = features[source + 8 + j * 3 + k];
                for (int k = 0; k < 6; k++)
                    cornerValues[target + 4 + k] = features[source + 53 + j * 6 + k];
            }
        }

        SamMeshCornerGrid grid = SamMeshCornerGrid.Build(cubes, cornerValues, 10);

        // Lattice point -> vertex, for the active points; the points of cubes that touch them but carry nothing are outside (SDF 1, no deformation, colour 0) and appended on demand.
        Dictionary<long, int> vertexOf = new(grid.PointCount);
        List<long> keys = new(grid.PointCount);
        List<float> sdf = new(grid.PointCount);
        List<float[]> position = new(grid.PointCount), colour = new(grid.PointCount);
        for (int p = 0; p < grid.PointCount; p++)
        {
            long key = SamMeshCornerGrid.Pack(grid.Points[p * 3], grid.Points[p * 3 + 1], grid.Points[p * 3 + 2]);
            vertexOf[key] = p;
            keys.Add(key);
            float[] a = grid.Attributes[(p * 10)..(p * 10 + 10)];
            sdf.Add(a[0]);
            position.Add(Deformed(grid.Points[p * 3], grid.Points[p * 3 + 1], grid.Points[p * 3 + 2], a[1], a[2], a[3], resolution));
            colour.Add(a[4..10]);
        }

        int Vertex(int x, int y, int z)
        {
            long key = SamMeshCornerGrid.Pack(x, y, z);
            if (vertexOf.TryGetValue(key, out int existing))
                return existing;
            int index = keys.Count;
            vertexOf[key] = index;
            keys.Add(key);
            sdf.Add(1f);
            position.Add(Deformed(x, y, z, 0, 0, 0, resolution));
            colour.Add(new float[6]);
            return index;
        }

        // The cubes with an inside corner, in the order of upstream's dense grid (x, then y, then z), with their weights.
        HashSet<long> candidates = [];
        for (int p = 0; p < grid.PointCount; p++)
        {
            if (sdf[p] >= 0)
                continue;
            for (int j = 0; j < 8; j++)
            {
                int cx = grid.Points[p * 3] - SamMeshCornerGrid.CornerOffsets[j][0], cy = grid.Points[p * 3 + 1] - SamMeshCornerGrid.CornerOffsets[j][1], cz = grid.Points[p * 3 + 2] - SamMeshCornerGrid.CornerOffsets[j][2];
                if (cx >= 0 && cx < resolution && cy >= 0 && cy < resolution && cz >= 0 && cz < resolution)
                    candidates.Add(SamMeshCornerGrid.Pack(cx, cy, cz));
            }
        }

        Dictionary<long, int> voxelOf = new(m);
        for (int i = 0; i < m; i++)
            voxelOf[SamMeshCornerGrid.Pack(cubes[i * 3], cubes[i * 3 + 1], cubes[i * 3 + 2])] = i;

        long[] cubeKeys = [.. candidates.Order()];
        int n = cubeKeys.Length;
        int[] corner = new int[n * 8];
        int[] cubeCase = new int[n];
        bool[] surface = new bool[n];
        for (int c = 0; c < n; c++)
        {
            SamMeshCornerGrid.Unpack(cubeKeys[c], out int cx, out int cy, out int cz);
            int caseId = 0, inside = 0;
            for (int j = 0; j < 8; j++)
            {
                int v = Vertex(cx + SamMeshCornerGrid.CornerOffsets[j][0], cy + SamMeshCornerGrid.CornerOffsets[j][1], cz + SamMeshCornerGrid.CornerOffsets[j][2]);
                corner[c * 8 + j] = v;
                if (sdf[v] < 0)
                {
                    caseId |= 1 << j;
                    inside++;
                }
            }

            cubeCase[c] = caseId;
            surface[c] = inside > 0 && inside < 8;
        }

        int[] surfaceCubes = [.. Enumerable.Range(0, n).Where(c => surface[c])];
        if (surfaceCubes.Length == 0)
            return new SamMeshResult([], [], []);

        ResolveAmbiguousCases(cubeKeys, surfaceCubes, cubeCase, tables, resolution);
        return Triangulate(cubeKeys, corner, surfaceCubes, cubeCase, voxelOf, features, keys, vertexOf, sdf, position, colour, tables);
    }

    // The cases whose check flag is set are resolved against the cube they point at, as upstream's _get_case_id does, from the cases as they were before any was replaced.
    internal static void ResolveAmbiguousCases(long[] cubeKeys, int[] surfaceCubes, int[] cubeCase, SamFlexiCubesTables tables, int resolution)
    {
        Dictionary<long, int> flagged = [];
        foreach (int c in surfaceCubes)
        {
            if (tables.Check[cubeCase[c] * 5] == 1)
                flagged[cubeKeys[c]] = cubeCase[c];
        }

        List<(int Cube, int Replacement)> replacements = [];
        foreach (int c in surfaceCubes)
        {
            int config = cubeCase[c];
            if (tables.Check[config * 5] != 1)
                continue;
            SamMeshCornerGrid.Unpack(cubeKeys[c], out int cx, out int cy, out int cz);
            int nx = cx + tables.Check[config * 5 + 1], ny = cy + tables.Check[config * 5 + 2], nz = cz + tables.Check[config * 5 + 3];
            if (nx < 0 || nx >= resolution || ny < 0 || ny >= resolution || nz < 0 || nz >= resolution)
                continue;
            if (flagged.ContainsKey(SamMeshCornerGrid.Pack(nx, ny, nz)))
                replacements.Add((c, tables.Check[config * 5 + 4]));
        }

        foreach ((int cube, int replacement) in replacements)
            cubeCase[cube] = replacement;
    }

    private static SamMeshResult Triangulate(
        long[] cubeKeys, int[] corner, int[] surfaceCubes, int[] cubeCase, Dictionary<long, int> voxelOf, float[] features,
        List<long> keys, Dictionary<long, int> indexOfKey, List<float> sdf, List<float[]> position, List<float[]> colour, SamFlexiCubesTables tables)
    {
        int s = surfaceCubes.Length;

        // Each surface cube's 12 edges as ordered vertex pairs; the distinct pairs in lexicographic order of their (packed) end points; the ones whose ends differ in sign are the surface edges.
        List<(long A, long B)> all = new(s * 12);
        for (int i = 0; i < s; i++)
        {
            for (int e = 0; e < 12; e++)
                all.Add((keys[corner[surfaceCubes[i] * 8 + CubeEdges[e * 2]]], keys[corner[surfaceCubes[i] * 8 + CubeEdges[e * 2 + 1]]]));
        }

        (long A, long B)[] distinct = [.. all.Distinct().Order()];
        Dictionary<(long, long), int> distinctId = new(distinct.Length);
        for (int d = 0; d < distinct.Length; d++)
            distinctId[distinct[d]] = d;
        int[] counts = new int[distinct.Length];
        foreach ((long, long) pair in all)
            counts[distinctId[pair]]++;
        int[] surfaceEdgeOf = new int[distinct.Length];
        List<int> surfaceEdgeA = [], surfaceEdgeB = [];
        for (int d = 0; d < distinct.Length; d++)
        {
            int a = indexOfKey[distinct[d].A], b = indexOfKey[distinct[d].B];
            if ((sdf[a] < 0) != (sdf[b] < 0))
            {
                surfaceEdgeOf[d] = surfaceEdgeA.Count;
                surfaceEdgeA.Add(a);
                surfaceEdgeB.Add(b);
            }
            else
            {
                surfaceEdgeOf[d] = -1;
            }
        }

        // Weights of every surface cube, normalised as upstream does (a cube without a voxel has zeros: beta and alpha 1, gamma 0.5).
        float[] beta = new float[s * 12], alpha = new float[s * 8], gamma = new float[s];
        for (int i = 0; i < s; i++)
        {
            SamMeshCornerGrid.Unpack(cubeKeys[surfaceCubes[i]], out int cx, out int cy, out int cz);
            long weights = voxelOf.TryGetValue(SamMeshCornerGrid.Pack(cx, cy, cz), out int voxel) ? (long)voxel * SlatMeshDecoder.OutChannels + 32 : -1;
            for (int e = 0; e < 12; e++)
                beta[i * 12 + e] = (float)(Math.Tanh(weights < 0 ? 0 : features[weights + e]) * 0.99 + 1);
            for (int j = 0; j < 8; j++)
                alpha[i * 8 + j] = (float)(Math.Tanh(weights < 0 ? 0 : features[weights + 12 + j]) * 0.99 + 1);
            double g = weights < 0 ? 0 : features[weights + 20];
            gamma[i] = (float)(1.0 / (1.0 + Math.Exp(-g)) * 0.99 + 0.005);
        }

        // Dual vertices, batched by the number of vertices a case emits, in ascending order of that number (as upstream's unique(num_vd) loop).
        int[] numVd = new int[s];
        for (int i = 0; i < s; i++)
            numVd[i] = tables.NumVd[cubeCase[surfaceCubes[i]]];
        List<float[]> vd = [], vdColour = [];
        List<float> vdGamma = [];
        int[] vdOfCubeEdge = new int[s * 12];
        Array.Fill(vdOfCubeEdge, -1);
        foreach (int num in numVd.Distinct().Order())
        {
            for (int i = 0; i < s; i++)
            {
                if (numVd[i] != num)
                    continue;
                for (int g = 0; g < num; g++)
                {
                    float[] sum = new float[3], colourSum = new float[6];
                    float betaSum = 0;
                    int created = vd.Count;
                    for (int t = 0; t < SamFlexiCubesTables.EdgesPerVertex; t++)
                    {
                        int e = tables.Dmc[(cubeCase[surfaceCubes[i]] * tables.MaxVertices + g) * SamFlexiCubesTables.EdgesPerVertex + t];
                        if (e == -1)
                            continue;
                        int cornerA = CubeEdges[e * 2], cornerB = CubeEdges[e * 2 + 1];
                        int va = corner[surfaceCubes[i] * 8 + cornerA], vb = corner[surfaceCubes[i] * 8 + cornerB];
                        float sa = sdf[va] * alpha[i * 8 + cornerA], sb = sdf[vb] * alpha[i * 8 + cornerB];
                        // linear_interp: the zero of the line through (x_a, s_a) and (x_b, s_b), weights swapped and negated.
                        float wa = sb, wb = -sa, denominator = wa + wb;
                        float b = beta[i * 12 + e];
                        for (int k = 0; k < 3; k++)
                            sum[k] += (position[va][k] * wa + position[vb][k] * wb) / denominator * b;
                        for (int k = 0; k < 6; k++)
                            colourSum[k] += (Sigmoid(colour[va][k]) * wa + Sigmoid(colour[vb][k]) * wb) / denominator * b;
                        betaSum += b;
                        vdOfCubeEdge[i * 12 + e] = created;
                    }

                    for (int k = 0; k < 3; k++)
                        sum[k] /= betaSum;
                    for (int k = 0; k < 6; k++)
                        colourSum[k] /= betaSum;
                    vd.Add(sum);
                    vdColour.Add(colourSum);
                    vdGamma.Add(gamma[i]);
                }
            }
        }

        // Quads: every surface edge shared by four surface cubes joins the four dual vertices of those cubes; sorted by edge, stably in cube order.
        List<(int Edge, int Vertex)> incidences = [];
        for (int i = 0; i < s; i++)
        {
            for (int e = 0; e < 12; e++)
            {
                int d = distinctId[(keys[corner[surfaceCubes[i] * 8 + CubeEdges[e * 2]]], keys[corner[surfaceCubes[i] * 8 + CubeEdges[e * 2 + 1]]])];
                if (surfaceEdgeOf[d] >= 0 && counts[d] == 4)
                    incidences.Add((surfaceEdgeOf[d], vdOfCubeEdge[i * 12 + e]));
            }
        }

        // OrderBy is stable: incidences of one edge stay in cube order.
        (int Edge, int Vertex)[] ordered = [.. incidences.OrderBy(x => x.Edge)];
        List<int[]> flipped = [], unflipped = [];
        for (int q = 0; q + 3 < ordered.Length; q += 4)
        {
            int edge = ordered[q].Edge;
            int[] quad = [ordered[q].Vertex, ordered[q + 1].Vertex, ordered[q + 2].Vertex, ordered[q + 3].Vertex];
            if (sdf[surfaceEdgeA[edge]] > 0)
                flipped.Add([quad[0], quad[1], quad[3], quad[2]]);
            else
                unflipped.Add([quad[2], quad[3], quad[1], quad[0]]);
        }

        List<int> faces = [];
        foreach (int[] quad in flipped.Concat(unflipped))
        {
            float gamma02 = vdGamma[quad[0]] * vdGamma[quad[2]], gamma13 = vdGamma[quad[1]] * vdGamma[quad[3]];
            foreach (int index in gamma02 > gamma13 ? QuadSplit1 : QuadSplit2)
                faces.Add(quad[index]);
        }

        float[] vertices = new float[vd.Count * 3], colours = new float[vd.Count * 6];
        for (int v = 0; v < vd.Count; v++)
        {
            Array.Copy(vd[v], 0, vertices, v * 3, 3);
            Array.Copy(vdColour[v], 0, colours, v * 6, 6);
        }

        return new SamMeshResult(vertices, [.. faces], colours);
    }

    private static float[] Deformed(int x, int y, int z, float dx, float dy, float dz, int resolution)
    {
        double scale = (1 - 1e-8) / (resolution * 2);
        return
        [
            (float)((double)x / resolution - 0.5 + scale * Math.Tanh(dx)),
            (float)((double)y / resolution - 0.5 + scale * Math.Tanh(dy)),
            (float)((double)z / resolution - 0.5 + scale * Math.Tanh(dz)),
        ];
    }

    private static float Sigmoid(float v) => (float)(1.0 / (1.0 + Math.Exp(-v)));
}
