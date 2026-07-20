namespace HartsyInference.World.Memory;

using HartsyInference.World.Camera;

/// <summary>Camera-frustum overlap memory retrieval — the CPU port of Matrix-Game-3
/// <c>utils/cam_utils.py:select_memory_idx_fov</c>. Samples points uniformly inside a sphere around the current
/// camera, computes each candidate frame's visible-overlap fraction with the current view, and picks the top-k.
/// Purely geometric — no learned retrieval. A GPU kernel is a documented perf follow-up (the reference shows FOV
/// retrieval on CPU costs ~33 FPS of the 40 FPS budget).</summary>
public sealed class FrustumOverlapSelector
{
    private readonly float[] _points;   // [n, 3] world offsets around the current camera
    private readonly float _fx, _fy, _cx, _cy;
    private readonly int _width, _height;
    private readonly float _near, _far;

    /// <summary>Creates a selector with a fixed deterministic point cloud (seeded — replayable sessions select the
    /// same memory). <paramref name="radius"/> is the world-space sampling radius around the current camera.</summary>
    public FrustumOverlapSelector(float fx, float fy, float cx, float cy, int width, int height,
        int numPoints = 2048, float radius = 10f, float near = 0.01f, float far = 100f, int seed = 7)
    {
        _fx = fx; _fy = fy; _cx = cx; _cy = cy;
        _width = width; _height = height;
        _near = near; _far = far;
        _points = GeneratePointsInSphere(numPoints, radius, seed);
    }

    /// <summary>Selects the indices (into <paramref name="candidatePoses"/>) of the <paramref name="k"/> candidates
    /// whose frustum shares the most visible points with <paramref name="currentPose"/> (both camera-to-world).
    /// Zero-overlap slots fall back to the most recent unselected candidates, mirroring the reference.</summary>
    public int[] Select(IReadOnlyList<float[]> candidatePoses, ReadOnlySpan<float> currentPose, int k)
    {
        if (candidatePoses.Count < k)
            throw new ArgumentException($"need at least {k} candidates; got {candidatePoses.Count}.", nameof(candidatePoses));

        int n = _points.Length / 3;
        Span<float> world = new float[n * 3];
        Span<float> currentW2C = stackalloc float[16];
        Se3Math.Invert(currentPose, currentW2C);

        // Sphere centered on the current camera.
        float ox = currentPose[3], oy = currentPose[7], oz = currentPose[11];
        for (int i = 0; i < n; i++)
        {
            world[i * 3 + 0] = _points[i * 3 + 0] + ox;
            world[i * 3 + 1] = _points[i * 3 + 1] + oy;
            world[i * 3 + 2] = _points[i * 3 + 2] + oz;
        }

        bool[] inCurrent = new bool[n];
        int currentVisible = 0;
        for (int i = 0; i < n; i++)
            if (inCurrent[i] = IsVisible(currentW2C, world.Slice(i * 3, 3)))
                currentVisible++;

        double[] ratios = new double[candidatePoses.Count];
        Span<float> w2c = stackalloc float[16];
        for (int c = 0; c < candidatePoses.Count; c++)
        {
            Se3Math.Invert(candidatePoses[c], w2c);
            int shared = 0;
            for (int i = 0; i < n; i++)
                if (inCurrent[i] && IsVisible(w2c, world.Slice(i * 3, 3)))
                    shared++;
            ratios[c] = currentVisible > 0 ? shared / (double)currentVisible : 0;
        }

        // Top-k by overlap; zero-overlap slots fall back to the most recent unselected candidates.
        bool[] taken = new bool[candidatePoses.Count];
        int[] selected = new int[k];
        for (int s = 0; s < k; s++)
        {
            int best = -1;
            double bestRatio = 0;
            for (int c = 0; c < candidatePoses.Count; c++)
                if (!taken[c] && ratios[c] > bestRatio) { bestRatio = ratios[c]; best = c; }
            if (best < 0)
            {
                for (int c = candidatePoses.Count - 1; c >= 0; c--)
                    if (!taken[c]) { best = c; break; }
            }
            taken[best] = true;
            selected[s] = best;
        }
        Array.Sort(selected);   // historical order
        return selected;
    }

    private bool IsVisible(ReadOnlySpan<float> w2c, ReadOnlySpan<float> point)
    {
        float xc = w2c[0] * point[0] + w2c[1] * point[1] + w2c[2] * point[2] + w2c[3];
        float yc = w2c[4] * point[0] + w2c[5] * point[1] + w2c[6] * point[2] + w2c[7];
        float zc = w2c[8] * point[0] + w2c[9] * point[1] + w2c[10] * point[2] + w2c[11];
        if (zc < _near || zc > _far) return false;
        float u = _fx * (xc / zc) + _cx;
        float v = _fy * (yc / zc) + _cy;
        return u >= 0 && u < _width && v >= 0 && v < _height;
    }

    /// <summary>Uniform-in-sphere sampling (radius <c>r = R·u^(1/3)</c>), deterministic for a given seed.</summary>
    public static float[] GeneratePointsInSphere(int numPoints, float radius, int seed)
    {
        Random rng = new(seed);
        float[] points = new float[numPoints * 3];
        for (int i = 0; i < numPoints; i++)
        {
            double phi = 2 * Math.PI * rng.NextDouble();
            double cosTheta = 2 * rng.NextDouble() - 1;
            double sinTheta = Math.Sqrt(1 - cosTheta * cosTheta);
            double r = radius * Math.Cbrt(rng.NextDouble());
            points[i * 3 + 0] = (float)(r * sinTheta * Math.Cos(phi));
            points[i * 3 + 1] = (float)(r * sinTheta * Math.Sin(phi));
            points[i * 3 + 2] = (float)(r * cosTheta);
        }
        return points;
    }
}
