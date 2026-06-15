using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>Point-set sampling used by VecSet-style 3D encoders and as a generic geometry utility.
/// Farthest-point sampling (FPS) gives an even, coverage-maximizing subset — the standard input
/// reduction for point-conditioned shape encoders.</summary>
public static class SurfaceSampler
{
    /// <summary>Farthest-point sampling: returns <paramref name="count"/> indices into
    /// <paramref name="points"/> (length 3*N) whose selected points are maximally spread. Deterministic:
    /// seeds from <paramref name="startIndex"/>. If <paramref name="count"/> ≥ N, returns all indices.</summary>
    public static int[] FarthestPointIndices(ReadOnlySpan<float> points, int count, int startIndex = 0)
    {
        int n = points.Length / 3;
        if (count >= n) { int[] all = new int[n]; for (int i = 0; i < n; i++) all[i] = i; return all; }

        int[] picked = new int[count];
        float[] minDist = new float[n];
        Array.Fill(minDist, float.MaxValue);

        int cur = Math.Clamp(startIndex, 0, n - 1);
        for (int s = 0; s < count; s++)
        {
            picked[s] = cur;
            float px = points[cur * 3], py = points[cur * 3 + 1], pz = points[cur * 3 + 2];
            float best = -1f; int bestIdx = cur;
            for (int i = 0; i < n; i++)
            {
                float dx = points[i * 3] - px, dy = points[i * 3 + 1] - py, dz = points[i * 3 + 2] - pz;
                float d = dx * dx + dy * dy + dz * dz;
                if (d < minDist[i]) minDist[i] = d;
                if (minDist[i] > best) { best = minDist[i]; bestIdx = i; }
            }
            cur = bestIdx;
        }
        return picked;
    }

    /// <summary>Gathers the points at <paramref name="indices"/> into a new <see cref="PointCloud"/>.</summary>
    public static PointCloud Gather(ReadOnlySpan<float> points, int[] indices)
    {
        float[] outP = new float[indices.Length * 3];
        for (int i = 0; i < indices.Length; i++)
        {
            int s = indices[i] * 3;
            outP[i * 3] = points[s]; outP[i * 3 + 1] = points[s + 1]; outP[i * 3 + 2] = points[s + 2];
        }
        return new PointCloud { Positions = outP };
    }
}
