namespace HartsyInference.ThreeD.Paint;

/// <summary>Binary image filters the bake needs: a Canny-style edge detector and square dilation.</summary>
public static class PaintImageFilters
{
    /// <summary>Sobel L1 gradient magnitude with hysteresis thresholds (<c>cv2.Canny</c>'s default norm and 3x3 aperture) on an 8-bit image.
    /// Non-maximum suppression is omitted, so edges are up to a pixel thicker; the caller dilates them anyway.</summary>
    public static bool[] Edges(byte[] gray, int width, int height, int low, int high)
    {
        if (gray.Length != width * height)
            throw new ArgumentException("Image size does not match the dimensions.", nameof(gray));
        int[] magnitude = new int[gray.Length];
        Parallel.For(1, height - 1, y =>
        {
            for (int x = 1; x < width - 1; x++)
            {
                int i = y * width + x;
                int gx = gray[i - width + 1] + 2 * gray[i + 1] + gray[i + width + 1] - gray[i - width - 1] - 2 * gray[i - 1] - gray[i + width - 1];
                int gy = gray[i + width - 1] + 2 * gray[i + width] + gray[i + width + 1] - gray[i - width - 1] - 2 * gray[i - width] - gray[i - width + 1];
                magnitude[i] = Math.Abs(gx) + Math.Abs(gy);
            }
        });
        bool[] edge = new bool[gray.Length];
        Stack<int> pending = new();
        for (int i = 0; i < magnitude.Length; i++)
            if (magnitude[i] > high)
            {
                edge[i] = true;
                pending.Push(i);
            }
        while (pending.Count > 0)
        {
            int i = pending.Pop();
            int x = i % width, y = i / width;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    int nx = x + dx, ny = y + dy;
                    if ((uint)nx >= (uint)width || (uint)ny >= (uint)height)
                        continue;
                    int j = ny * width + nx;
                    if (!edge[j] && magnitude[j] > low)
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
