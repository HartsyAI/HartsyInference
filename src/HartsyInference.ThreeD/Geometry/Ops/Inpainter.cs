namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>
/// Fills the unknown texels of a float image (texture gutters) with the fast-marching method of Telea (2004), the algorithm behind
/// OpenCV's <c>INPAINT_TELEA</c>: unknown texels are visited in order of distance from the known region and each takes a weighted,
/// gradient-corrected average of known texels within <c>radius</c>. Deterministic and single-threaded; not bit-identical to OpenCV.
/// </summary>
public static class Inpainter
{
    private const byte Known = 0;
    private const byte Band = 1;
    private const byte Inside = 2;
    private const float Infinity = 1e6f;

    /// <summary>Returns a copy of <paramref name="data"/> (<c>width * height * channels</c>, texel-major) with every texel where
    /// <paramref name="known"/> is false filled in. Known texels are unchanged; with <paramref name="clampToUnit"/> every filled value is held to [0, 1], as OpenCV does by working in 8 bits (the gradient term otherwise runs away across large gaps); an image with no known texel is returned as is. With <paramref name="boundByNeighbours"/> every filled value is also held, per channel, to the range of the texels that contributed to it:
    /// the gradient correction can then no longer push a value past its neighbours, which is what turns the border of a large gap into saturated red, green and blue streaks (on a Paint 2.1 atlas a quarter of the filled
    /// texels came out saturated, against a third of a percent for OpenCV's Telea).</summary>
    public static float[] Inpaint(float[] data, int width, int height, int channels, bool[] known, int radius, bool clampToUnit = false, bool boundByNeighbours = false)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(known);
        if (width <= 0 || height <= 0 || channels <= 0 || radius <= 0)
            throw new ArgumentOutOfRangeException(nameof(radius), "Dimensions, channels and radius must be positive.");
        if (boundByNeighbours && channels > 16)
            throw new ArgumentOutOfRangeException(nameof(channels), "Bounding by the neighbours takes at most 16 channels.");
        int n = width * height;
        if (data.Length != n * channels || known.Length != n)
            throw new ArgumentException("Data and mask do not match the image shape.");

        float[] result = (float[])data.Clone();
        byte[] flag = new byte[n];
        int unknown = 0;
        for (int i = 0; i < n; i++)
        {
            if (known[i]) continue;
            flag[i] = Inside;
            unknown++;
        }
        if (unknown == 0 || unknown == n)
            return result;

        float[] signed = new float[n];
        ComputeOutsideDistance(flag, width, height, radius + 2, signed);

        float[] t = new float[n];
        PriorityQueue<int, (float, int)> heap = new();
        for (int i = 0; i < n; i++)
        {
            if (flag[i] != Inside || !HasNeighborWithFlag(flag, i, width, height, Known))
                continue;
            t[i] = Solve(t, flag, i, width, height);
            signed[i] = t[i];
            Estimate(result, flag, signed, i, width, height, channels, radius, clampToUnit, boundByNeighbours);
            flag[i] = Band;
            heap.Enqueue(i, (t[i], i));
        }

        Span<int> nb = stackalloc int[4];
        while (heap.TryDequeue(out int p, out (float T, int Index) key))
        {
            if (flag[p] == Known || key.T > t[p])
                continue;
            flag[p] = Known;
            int count = Neighbors(p, width, height, nb);
            for (int k = 0; k < count; k++)
            {
                int q = nb[k];
                if (flag[q] == Known)
                    continue;
                float cand = Solve(t, flag, q, width, height);
                if (flag[q] == Inside)
                {
                    t[q] = cand;
                    signed[q] = cand;
                    Estimate(result, flag, signed, q, width, height, channels, radius, clampToUnit, boundByNeighbours);
                    flag[q] = Band;
                    heap.Enqueue(q, (cand, q));
                }
                else if (cand < t[q])
                {
                    t[q] = cand;
                    signed[q] = cand;
                    heap.Enqueue(q, (cand, q));
                }
            }
        }
        return result;
    }

    private static void ComputeOutsideDistance(byte[] flag, int width, int height, int range, float[] signed)
    {
        int n = width * height;
        float[] d = new float[n];
        Array.Fill(d, Infinity);
        bool[] done = new bool[n];
        PriorityQueue<int, (float, int)> heap = new();
        for (int i = 0; i < n; i++)
        {
            if (flag[i] != Known || !HasNeighborWithFlag(flag, i, width, height, Inside))
                continue;
            d[i] = 0f;
            heap.Enqueue(i, (0f, i));
        }
        Span<int> nb = stackalloc int[4];
        while (heap.TryDequeue(out int p, out (float T, int Index) key))
        {
            if (done[p] || key.T > d[p])
                continue;
            done[p] = true;
            int count = Neighbors(p, width, height, nb);
            for (int k = 0; k < count; k++)
            {
                int q = nb[k];
                if (done[q] || flag[q] != Known)
                    continue;
                float cand = SolveDistance(d, done, q, width, height);
                if (cand < d[q] && cand <= range)
                {
                    d[q] = cand;
                    heap.Enqueue(q, (cand, q));
                }
            }
        }
        for (int i = 0; i < n; i++)
            signed[i] = flag[i] == Known ? -(done[i] ? d[i] : range + 1f) : 0f;
    }

    private static float Solve(float[] t, byte[] flag, int p, int width, int height)
    {
        int x = p % width, y = p / width;
        float a = Infinity, b = Infinity;
        if (x > 0 && flag[p - 1] == Known) a = MathF.Min(a, t[p - 1]);
        if (x < width - 1 && flag[p + 1] == Known) a = MathF.Min(a, t[p + 1]);
        if (y > 0 && flag[p - width] == Known) b = MathF.Min(b, t[p - width]);
        if (y < height - 1 && flag[p + width] == Known) b = MathF.Min(b, t[p + width]);
        return Eikonal(a, b);
    }

    private static float SolveDistance(float[] d, bool[] done, int p, int width, int height)
    {
        int x = p % width, y = p / width;
        float a = Infinity, b = Infinity;
        if (x > 0 && done[p - 1]) a = MathF.Min(a, d[p - 1]);
        if (x < width - 1 && done[p + 1]) a = MathF.Min(a, d[p + 1]);
        if (y > 0 && done[p - width]) b = MathF.Min(b, d[p - width]);
        if (y < height - 1 && done[p + width]) b = MathF.Min(b, d[p + width]);
        return Eikonal(a, b);
    }

    private static float Eikonal(float a, float b)
    {
        float lo = MathF.Min(a, b), hi = MathF.Max(a, b);
        if (lo >= Infinity) return Infinity;
        if (hi >= Infinity || hi - lo >= 1f) return lo + 1f;
        return (lo + hi + MathF.Sqrt(2f - (hi - lo) * (hi - lo))) * 0.5f;
    }

    private static void Estimate(float[] img, byte[] flag, float[] signed, int p, int width, int height, int channels, int radius, bool clampToUnit, bool boundByNeighbours)
    {
        int px = p % width, py = p / width;
        float tp = signed[p];
        float gtx = Gradient(signed, p, 1, px > 0 && flag[p - 1] != Inside, px < width - 1 && flag[p + 1] != Inside, tp);
        float gty = Gradient(signed, p, width, py > 0 && flag[p - width] != Inside, py < height - 1 && flag[p + width] != Inside, tp);
        Span<double> acc = stackalloc double[Math.Min(channels, 16)];
        double[]? big = channels > 16 ? new double[channels] : null;
        Span<double> sum = big ?? acc;
        sum[..channels].Clear();
        bool bound = boundByNeighbours;
        Span<float> low = stackalloc float[16], high = stackalloc float[16];
        if (bound)
        {
            low.Fill(float.MaxValue);
            high.Fill(float.MinValue);
        }

        double wsum = 0;
        int r2 = radius * radius;
        for (int dy = -radius; dy <= radius; dy++)
        {
            int qy = py + dy;
            if ((uint)qy >= (uint)height) continue;
            for (int dx = -radius; dx <= radius; dx++)
            {
                int d2 = dx * dx + dy * dy;
                int qx = px + dx;
                if (d2 == 0 || d2 > r2 || (uint)qx >= (uint)width) continue;
                int q = qy * width + qx;
                if (flag[q] == Inside) continue;
                double rx = -dx, ry = -dy, len = Math.Sqrt(d2);
                double dir = Math.Abs(rx * gtx + ry * gty) / len;
                if (dir < 0.01) dir = 1e-6;
                double w = dir / d2 / (1.0 + Math.Abs(signed[q] - tp));
                for (int c = 0; c < channels; c++)
                {
                    double gx = Gradient(img, q * channels + c, channels, qx > 0 && flag[q - 1] != Inside,
                        qx < width - 1 && flag[q + 1] != Inside, img[q * channels + c]);
                    double gy = Gradient(img, q * channels + c, width * channels, qy > 0 && flag[q - width] != Inside,
                        qy < height - 1 && flag[q + width] != Inside, img[q * channels + c]);
                    sum[c] += w * (img[q * channels + c] + gx * rx + gy * ry);
                    if (bound)
                    {
                        low[c] = MathF.Min(low[c], img[q * channels + c]);
                        high[c] = MathF.Max(high[c], img[q * channels + c]);
                    }
                }
                wsum += w;
            }
        }
        for (int c = 0; c < channels; c++)
        {
            float value = wsum > 0 ? (float)(sum[c] / wsum) : 0f;
            if (wsum > 0 && bound)
                value = Math.Clamp(value, low[c], high[c]);
            img[p * channels + c] = clampToUnit ? Math.Clamp(value, 0f, 1f) : value;
        }
    }

    // Central difference when both sides are valid, one-sided when one is, zero otherwise.
    private static float Gradient(float[] values, int at, int step, bool hasMinus, bool hasPlus, float center)
    {
        if (hasPlus)
            return hasMinus ? (values[at + step] - values[at - step]) * 0.5f : values[at + step] - center;
        return hasMinus ? center - values[at - step] : 0f;
    }

    private static bool HasNeighborWithFlag(byte[] flag, int p, int width, int height, byte value)
    {
        int x = p % width, y = p / width;
        return (x > 0 && flag[p - 1] == value) || (x < width - 1 && flag[p + 1] == value)
            || (y > 0 && flag[p - width] == value) || (y < height - 1 && flag[p + width] == value);
    }

    private static int Neighbors(int p, int width, int height, Span<int> dst)
    {
        int x = p % width, y = p / width, count = 0;
        if (x > 0) dst[count++] = p - 1;
        if (x < width - 1) dst[count++] = p + 1;
        if (y > 0) dst[count++] = p - width;
        if (y < height - 1) dst[count++] = p + width;
        return count;
    }
}
