namespace HartsyInference.ThreeD.Paint;

/// <summary>Binary image filters the bake needs: OpenCV's Canny edge detector and square dilation.</summary>
public static class PaintImageFilters
{
    private const int CannyShift = 15, Tg22 = 13573; // tan(22.5 degrees) in 15-bit fixed point, as in OpenCV

    /// <summary><c>cv2.Canny(gray, low, high)</c> with its defaults (3x3 Sobel, L1 gradient magnitude) on an 8-bit image, as OpenCV's <c>canny.cpp</c> computes it: Sobel with a replicated border, the magnitude <c>|dx| + |dy|</c>,
    /// non-maximum suppression along the gradient direction chosen with the fixed-point tangent of 22.5 and 67.5 degrees (candidates need a magnitude above <paramref name="low"/>, zero outside the image), then hysteresis: a candidate is an
    /// edge when an 8-connected chain of candidates joins it to one above <paramref name="high"/>.</summary>
    public static bool[] Edges(byte[] gray, int width, int height, int low, int high)
    {
        if (gray.Length != width * height)
            throw new ArgumentException("Image size does not match the dimensions.", nameof(gray));
        int[] dx = new int[gray.Length], dy = new int[gray.Length], magnitude = new int[gray.Length];
        Parallel.For(0, height, y =>
        {
            int up = Math.Max(y - 1, 0) * width, mid = y * width, down = Math.Min(y + 1, height - 1) * width;
            for (int x = 0; x < width; x++)
            {
                int left = Math.Max(x - 1, 0), right = Math.Min(x + 1, width - 1);
                int gx = gray[up + right] + 2 * gray[mid + right] + gray[down + right] - gray[up + left] - 2 * gray[mid + left] - gray[down + left];
                int gy = gray[down + left] + 2 * gray[down + x] + gray[down + right] - gray[up + left] - 2 * gray[up + x] - gray[up + right];
                int i = mid + x;
                dx[i] = gx;
                dy[i] = gy;
                magnitude[i] = Math.Abs(gx) + Math.Abs(gy);
            }
        });

        int Mag(int x, int y) => (uint)x >= (uint)width || (uint)y >= (uint)height ? 0 : magnitude[y * width + x];
        bool[] candidate = new bool[gray.Length];
        Parallel.For(0, height, y =>
        {
            for (int x = 0; x < width; x++)
            {
                int i = y * width + x, m = magnitude[i];
                if (m <= low)
                    continue;
                int xs = dx[i], ys = dy[i], ax = Math.Abs(xs), ay = Math.Abs(ys) << CannyShift, tg22x = ax * Tg22;
                if (ay < tg22x)
                    candidate[i] = m > Mag(x - 1, y) && m >= Mag(x + 1, y);
                else if (ay > tg22x + (ax << (CannyShift + 1)))
                    candidate[i] = m > Mag(x, y - 1) && m >= Mag(x, y + 1);
                else
                {
                    int s = (xs ^ ys) < 0 ? -1 : 1;
                    candidate[i] = m > Mag(x - s, y - 1) && m > Mag(x + s, y + 1);
                }
            }
        });

        bool[] edge = new bool[gray.Length];
        Stack<int> pending = new();
        for (int i = 0; i < candidate.Length; i++)
            if (candidate[i] && magnitude[i] > high)
            {
                edge[i] = true;
                pending.Push(i);
            }
        while (pending.Count > 0)
        {
            int i = pending.Pop();
            int x = i % width, y = i / width;
            for (int ny = Math.Max(y - 1, 0); ny <= Math.Min(y + 1, height - 1); ny++)
                for (int nx = Math.Max(x - 1, 0); nx <= Math.Min(x + 1, width - 1); nx++)
                {
                    int j = ny * width + nx;
                    if (candidate[j] && !edge[j])
                    {
                        edge[j] = true;
                        pending.Push(j);
                    }
                }
        }
        return edge;
    }

    /// <summary>Grows true pixels by <paramref name="radius"/> in both axes (a square structuring element); pixels outside the image are false.</summary>
    public static bool[] Dilate(bool[] mask, int width, int height, int radius)
    {
        if (radius <= 0)
            return (bool[])mask.Clone();
        bool[] horizontal = new bool[mask.Length];
        Parallel.For(0, height, y =>
        {
            int run = int.MaxValue / 2;
            int[] nearest = new int[width];
            for (int x = 0; x < width; x++)
            {
                run = mask[y * width + x] ? 0 : run + 1;
                nearest[x] = run;
            }
            run = int.MaxValue / 2;
            for (int x = width - 1; x >= 0; x--)
            {
                run = mask[y * width + x] ? 0 : run + 1;
                horizontal[y * width + x] = Math.Min(run, nearest[x]) <= radius;
            }
        });
        bool[] result = new bool[mask.Length];
        Parallel.For(0, width, x =>
        {
            int run = int.MaxValue / 2;
            int[] nearest = new int[height];
            for (int y = 0; y < height; y++)
            {
                run = horizontal[y * width + x] ? 0 : run + 1;
                nearest[y] = run;
            }
            run = int.MaxValue / 2;
            for (int y = height - 1; y >= 0; y--)
            {
                run = horizontal[y * width + x] ? 0 : run + 1;
                result[y * width + x] = Math.Min(run, nearest[y]) <= radius;
            }
        });
        return result;
    }
}
