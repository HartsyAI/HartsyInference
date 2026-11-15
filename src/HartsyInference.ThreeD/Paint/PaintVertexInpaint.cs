namespace HartsyInference.ThreeD.Paint;

/// <summary>Upstream's mesh-vertex inpaint (<c>mesh_inpaint_processor.cpp</c>, <c>meshVerticeInpaint</c>, method <c>smooth</c>), which <c>MeshRender.uv_inpaint</c> runs before its 2D
/// inpaint. A vertex whose texel the bake painted keeps that colour; every other vertex takes the inverse-squared-distance weighted mean of its coloured neighbours on the mesh, repeated
/// until a pass colours nothing new twice in a row. The colours are written to the vertices' texels, which then seed the 2D inpaint. Without it, a surface no view sees (a forehead
/// under a hat brim, the inside of an ear) is filled only from the neighbouring texels of its UV chart, which smears them into streaks; with it, it takes the colour of the surface
/// around it on the mesh.
/// Differences from upstream: vertices at the same position are one vertex (upstream gets this from separate position and UV indices), edges are followed both ways, and a texel
/// the bake painted is never overwritten.</summary>
public static class PaintVertexInpaint
{
    /// <summary>Writes propagated vertex colours into the unpainted texels at the mesh's vertices and marks them known.</summary>
    /// <param name="texture">Row-major texture, <paramref name="channels"/> floats per texel; texel (x, y) samples UV ((x + 0.5) / width, (y + 0.5) / height).</param>
    /// <param name="known">Per texel, whether the bake painted it; updated.</param>
    /// <param name="positions">Vertex positions, three per vertex.</param>
    /// <param name="uvs">Vertex UVs, two per vertex.</param>
    /// <param name="indices">Triangle indices.</param>
    /// <returns>How many texels were seeded.</returns>
    public static int Seed(float[] texture, bool[] known, int width, int height, int channels, float[] positions, float[] uvs, int[] indices)
    {
        ArgumentNullException.ThrowIfNull(texture);
        ArgumentNullException.ThrowIfNull(known);
        ArgumentNullException.ThrowIfNull(positions);
        ArgumentNullException.ThrowIfNull(uvs);
        ArgumentNullException.ThrowIfNull(indices);
        int vertexCount = positions.Length / 3;
        if (uvs.Length != vertexCount * 2 || known.Length != width * height || texture.Length != known.Length * channels)
            throw new ArgumentException("The texture, mask, positions and UVs do not match.");

        // One id per distinct position, so a seam's split vertices are one vertex of the graph.
        int[] canonical = new int[vertexCount];
        Dictionary<(float, float, float), int> byPosition = new(vertexCount);
        List<int> first = [];
        for (int v = 0; v < vertexCount; v++)
        {
            (float, float, float) key = (positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]);
            if (!byPosition.TryGetValue(key, out int id))
            {
                id = first.Count;
                byPosition[key] = id;
                first.Add(v);
            }
            canonical[v] = id;
        }
        int points = first.Count;
        int[] texel = new int[vertexCount];
        for (int v = 0; v < vertexCount; v++)
        {
            int x = Math.Clamp((int)Math.Floor(uvs[v * 2] * width), 0, width - 1);
            int y = Math.Clamp((int)Math.Floor(uvs[v * 2 + 1] * height), 0, height - 1);
            texel[v] = y * width + x;
        }

        bool[] coloured = new bool[points];
        float[] colour = new float[points * channels];
        for (int v = 0; v < vertexCount; v++)
        {
            if (!known[texel[v]])
                continue;
            int id = canonical[v];
            coloured[id] = true;
            for (int c = 0; c < channels; c++)
                colour[id * channels + c] = texture[texel[v] * channels + c];
        }

        HashSet<int>[] neighbours = new HashSet<int>[points];
        for (int i = 0; i < points; i++)
            neighbours[i] = [];
        for (int t = 0; t + 2 < indices.Length; t += 3)
            for (int k = 0; k < 3; k++)
            {
                int a = canonical[indices[t + k]], b = canonical[indices[t + (k + 1) % 3]];
                if (a == b)
                    continue;
                neighbours[a].Add(b);
                neighbours[b].Add(a);
            }

        List<int> uncoloured = [];
        for (int i = 0; i < points; i++)
            if (!coloured[i])
                uncoloured.Add(i);
        // Upstream's loop: sweep the uncoloured vertices in place (a vertex coloured early in a sweep already counts for later ones), and stop once the count of vertices left
        // uncoloured has stayed the same for two sweeps.
        float[] sum = new float[channels];
        int patience = 2, lastLeft = 0;
        while (patience > 0)
        {
            int left = 0;
            foreach (int id in uncoloured)
            {
                Array.Clear(sum);
                double total = 0;
                int p = first[id] * 3;
                foreach (int other in neighbours[id])
                {
                    if (!coloured[other])
                        continue;
                    int q = first[other] * 3;
                    double dx = positions[p] - positions[q], dy = positions[p + 1] - positions[q + 1], dz = positions[p + 2] - positions[q + 2];
                    double weight = 1.0 / Math.Max(Math.Sqrt(dx * dx + dy * dy + dz * dz), 1e-4);
                    weight *= weight;
                    for (int c = 0; c < channels; c++)
                        sum[c] += (float)(colour[other * channels + c] * weight);
                    total += weight;
                }
                if (total > 0)
                {
                    for (int c = 0; c < channels; c++)
                        colour[id * channels + c] = (float)(sum[c] / total);
                    coloured[id] = true;
                }
                else
                    left++;
            }
            if (left == lastLeft)
                patience--;
            else
                patience++;
            lastLeft = left;
        }

        int seeded = 0;
        for (int v = 0; v < vertexCount; v++)
        {
            int id = canonical[v];
            if (!coloured[id] || known[texel[v]])
                continue;
            for (int c = 0; c < channels; c++)
                texture[texel[v] * channels + c] = colour[id * channels + c];
            known[texel[v]] = true;
            seeded++;
        }
        return seeded;
    }
}
