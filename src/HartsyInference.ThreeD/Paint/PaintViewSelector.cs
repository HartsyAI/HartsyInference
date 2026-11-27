namespace HartsyInference.ThreeD.Paint;

/// <summary>Greedy view selection of <c>ViewProcessor.bake_view_selection</c>: keep the six fixed views, then repeatedly add the candidate that
/// reveals the most not yet visible surface area, stopping when the best gain drops to 1 percent of the total area or less.</summary>
public static class PaintViewSelector
{
    /// <summary>Minimum fraction of the total area a further view must reveal.</summary>
    public const double MinimumGain = 0.01;

    /// <summary>Candidate views are rasterized at this size for visibility.</summary>
    public const int VisibilityResolution = 1024;

    /// <summary>Chooses at most <paramref name="maxViews"/> views (the six fixed ones at least) from <paramref name="candidates"/>.</summary>
    public static IReadOnlyList<PaintViewCandidate> Select(PaintViewRenderer renderer, IReadOnlyList<PaintViewCandidate> candidates, int maxViews,
        int resolution = VisibilityResolution, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(renderer);
        ArgumentNullException.ThrowIfNull(candidates);
        int fixedCount = PaintViewCandidate.Fixed.Count;
        if (candidates.Count < fixedCount)
            throw new ArgumentException("The candidate list must start with the six fixed views.", nameof(candidates));
        if (maxViews < fixedCount)
            throw new ArgumentOutOfRangeException(nameof(maxViews), $"At least {fixedCount} views are required.");

        PaintMeshFrame frame = renderer.Frame;
        double[] ratio = AreaRatios(frame);
        bool[][] visible = new bool[candidates.Count][];
        for (int i = 0; i < candidates.Count; i++)
        {
            ViewRaster raster = renderer.Rasterize(candidates[i].Elevation, candidates[i].Azimuth, resolution, cancellationToken);
            visible[i] = new bool[frame.TriangleCount];
            foreach (int id in raster.FaceIds)
                if (id > 0)
                    visible[i][id - 1] = true;
        }

        bool[] covered = new bool[frame.TriangleCount];
        bool[] selected = new bool[candidates.Count];
        List<PaintViewCandidate> result = [];
        for (int i = 0; i < fixedCount; i++)
        {
            selected[i] = true;
            result.Add(candidates[i]);
            Merge(covered, visible[i]);
        }
        while (result.Count < maxViews)
        {
            double best = 0;
            int bestIndex = -1;
            for (int i = 0; i < candidates.Count; i++)
            {
                if (selected[i])
                    continue;
                double gain = 0;
                for (int f = 0; f < covered.Length; f++)
                    if (visible[i][f] && !covered[f])
                        gain += ratio[f];
                if (gain > best)
                {
                    best = gain;
                    bestIndex = i;
                }
            }
            if (!(best > MinimumGain))
                break;
            selected[bestIndex] = true;
            result.Add(candidates[bestIndex]);
            Merge(covered, visible[bestIndex]);
        }
        return result;
    }

    private static void Merge(bool[] covered, bool[] view)
    {
        for (int f = 0; f < covered.Length; f++)
            covered[f] |= view[f];
    }

    private static double[] AreaRatios(PaintMeshFrame frame)
    {
        double[] area = new double[frame.TriangleCount];
        double total = 0;
        for (int t = 0; t < area.Length; t++)
        {
            int a = frame.Indices[t * 3] * 3, b = frame.Indices[t * 3 + 1] * 3, c = frame.Indices[t * 3 + 2] * 3;
            double e1x = frame.Vertices[b] - frame.Vertices[a], e1y = frame.Vertices[b + 1] - frame.Vertices[a + 1], e1z = frame.Vertices[b + 2] - frame.Vertices[a + 2];
            double e2x = frame.Vertices[c] - frame.Vertices[a], e2y = frame.Vertices[c + 1] - frame.Vertices[a + 1], e2z = frame.Vertices[c + 2] - frame.Vertices[a + 2];
            double nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
            area[t] = 0.5 * Math.Sqrt(nx * nx + ny * ny + nz * nz);
            total += area[t];
        }
        for (int t = 0; t < area.Length; t++)
            area[t] = total > 0 ? area[t] / total : 0;
        return area;
    }
}
