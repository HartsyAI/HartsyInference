namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>A static 3D k-d tree over a point set, for exact k-nearest-neighbour queries (the role of scipy's <c>cKDTree</c> upstream).</summary>
public sealed class UniRigKdTree
{
    private readonly float[] _points;
    private readonly int[] _order;
    private readonly byte[] _axis;

    /// <summary>Builds the tree over <paramref name="points"/> (length <c>3 * N</c>); the array is referenced, not copied.</summary>
    public UniRigKdTree(float[] points)
    {
        ArgumentNullException.ThrowIfNull(points);
        if (points.Length % 3 != 0)
        {
            throw new ArgumentException("Points need three floats each.", nameof(points));
        }

        _points = points;
        int count = points.Length / 3;
        _order = new int[count];
        for (int i = 0; i < count; i++)
        {
            _order[i] = i;
        }

        _axis = new byte[count];
        Build(0, count);
    }

    /// <summary>Number of points.</summary>
    public int Count => _order.Length;

    /// <summary>
    /// Writes the <paramref name="k"/> nearest points to <paramref name="query"/> into <paramref name="indices"/> and their Euclidean
    /// distances into <paramref name="distances"/>, nearest first. Fewer than <paramref name="k"/> points leaves the tail at <c>-1</c> and infinity.
    /// </summary>
    public void Nearest(float qx, float qy, float qz, int k, Span<int> indices, Span<float> distances)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(k, 1);
        for (int i = 0; i < k; i++)
        {
            indices[i] = -1;
            distances[i] = float.PositiveInfinity;
        }

        Search(0, _order.Length, qx, qy, qz, k, indices, distances);
        for (int i = 0; i < k; i++)
        {
            if (indices[i] >= 0)
            {
                distances[i] = MathF.Sqrt(distances[i]);
            }
        }
    }

    /// <summary>Index and distance of the single nearest point.</summary>
    public (int Index, float Distance) Nearest(float qx, float qy, float qz)
    {
        Span<int> index = stackalloc int[1];
        Span<float> distance = stackalloc float[1];
        Nearest(qx, qy, qz, 1, index, distance);
        return (index[0], distance[0]);
    }

    private void Build(int lo, int hi)
    {
        if (hi - lo <= 1)
        {
            return;
        }

        float[] min = [float.MaxValue, float.MaxValue, float.MaxValue];
        float[] max = [float.MinValue, float.MinValue, float.MinValue];
        for (int i = lo; i < hi; i++)
        {
            for (int a = 0; a < 3; a++)
            {
                float v = _points[3 * _order[i] + a];
                min[a] = Math.Min(min[a], v);
                max[a] = Math.Max(max[a], v);
            }
        }

        int axis = 0;
        for (int a = 1; a < 3; a++)
        {
            if (max[a] - min[a] > max[axis] - min[axis])
            {
                axis = a;
            }
        }

        int mid = (lo + hi) / 2;
        float[] keys = new float[hi - lo];
        int[] slice = new int[hi - lo];
        for (int i = lo; i < hi; i++)
        {
            keys[i - lo] = _points[3 * _order[i] + axis];
            slice[i - lo] = _order[i];
        }

        Array.Sort(keys, slice);
        Array.Copy(slice, 0, _order, lo, slice.Length);
        _axis[mid] = (byte)axis;
        Build(lo, mid);
        Build(mid + 1, hi);
    }

    private void Search(int lo, int hi, float qx, float qy, float qz, int k, Span<int> indices, Span<float> squared)
    {
        if (lo >= hi)
        {
            return;
        }

        int mid = (lo + hi) / 2;
        int point = _order[mid];
        float dx = _points[3 * point] - qx;
        float dy = _points[3 * point + 1] - qy;
        float dz = _points[3 * point + 2] - qz;
        Insert(point, dx * dx + dy * dy + dz * dz, k, indices, squared);

        if (hi - lo == 1)
        {
            return;
        }

        int axis = _axis[mid];
        float delta = axis == 0 ? qx - _points[3 * point] : axis == 1 ? qy - _points[3 * point + 1] : qz - _points[3 * point + 2];
        (int nearLo, int nearHi, int farLo, int farHi) = delta < 0 ? (lo, mid, mid + 1, hi) : (mid + 1, hi, lo, mid);
        Search(nearLo, nearHi, qx, qy, qz, k, indices, squared);
        if (delta * delta < squared[k - 1])
        {
            Search(farLo, farHi, qx, qy, qz, k, indices, squared);
        }
    }

    private static void Insert(int point, float distance, int k, Span<int> indices, Span<float> squared)
    {
        if (distance >= squared[k - 1])
        {
            return;
        }

        int at = k - 1;
        while (at > 0 && squared[at - 1] > distance)
        {
            squared[at] = squared[at - 1];
            indices[at] = indices[at - 1];
            at--;
        }

        squared[at] = distance;
        indices[at] = point;
    }
}
