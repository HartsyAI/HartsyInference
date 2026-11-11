namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>Orients, scales and packs charts into the unit square at uniform texel density (UV area proportional to surface area).
/// Small chart counts use MaxRects best-short-side-fit; large counts use a height-sorted shelf packer.</summary>
internal static class UvAtlasPacker
{
    private const int RotationSteps = 24;
    private const int MaxRectsLimit = 256;
    private const int SearchIterations = 18;

    /// <summary>Rewrites every chart's <see cref="UvChart.Uv"/> to final atlas coordinates inside [0,1].</summary>
    public static void Pack(IReadOnlyList<UvChart> charts, double paddingUv)
    {
        int n = charts.Count;
        double[] w = new double[n], h = new double[n];
        double areaSum = 0;
        for (int i = 0; i < n; i++)
        {
            Normalize(charts[i]);
            w[i] = charts[i].Width;
            h[i] = charts[i].Height;
            areaSum += w[i] * h[i];
        }
        int[] order = SortedOrder(w, h);

        double pad = paddingUv;
        double[] ox = new double[n], oy = new double[n];
        double scale = 0;
        while (true)
        {
            double hi = 1.0 / Math.Sqrt(Math.Max(areaSum, 1e-300));
            double lo = 0;
            for (int it = 0; it < SearchIterations; it++)
            {
                double mid = 0.5 * (lo + hi);
                if (TryPack(w, h, order, mid, pad, ox, oy)) lo = mid; else hi = mid;
            }
            bool acceptable = lo >= 0.2 * (1.0 / Math.Sqrt(Math.Max(areaSum, 1e-300))) || pad <= 0;
            if (acceptable)
            {
                scale = lo;
                break;
            }
            pad *= 0.5;
            if (pad < 1e-6) pad = 0;
        }
        if (scale <= 0)
            scale = 1e-6;
        if (!TryPack(w, h, order, scale, pad, ox, oy))
            throw new InvalidOperationException("Atlas packing failed.");

        for (int i = 0; i < n; i++)
        {
            double[] uv = charts[i].Uv;
            for (int k = 0; k < uv.Length; k += 2)
            {
                uv[k] = ox[i] + pad + scale * uv[k];
                uv[k + 1] = oy[i] + pad + scale * uv[k + 1];
            }
        }
    }

    private static void Normalize(UvChart chart)
    {
        double[] uv = chart.Uv;
        double bestArea = double.MaxValue, bestAngle = 0;
        for (int s = 0; s < RotationSteps; s++)
        {
            double angle = s * (Math.PI / 2) / RotationSteps;
            double cos = Math.Cos(angle), sin = Math.Sin(angle);
            double minX = double.MaxValue, minY = double.MaxValue, maxX = double.MinValue, maxY = double.MinValue;
            for (int k = 0; k < uv.Length; k += 2)
            {
                double x = uv[k] * cos - uv[k + 1] * sin, y = uv[k] * sin + uv[k + 1] * cos;
                minX = Math.Min(minX, x); maxX = Math.Max(maxX, x);
                minY = Math.Min(minY, y); maxY = Math.Max(maxY, y);
            }
            double area = (maxX - minX) * (maxY - minY);
            if (area < bestArea - 1e-12 * bestArea) { bestArea = area; bestAngle = angle; }
        }

        double c = Math.Cos(bestAngle), sn = Math.Sin(bestAngle);
        double scale = chart.Area2d > 1e-18 ? Math.Sqrt(chart.Area3d / chart.Area2d) : 1.0;
        double x0 = double.MaxValue, y0 = double.MaxValue, x1 = double.MinValue, y1 = double.MinValue;
        for (int k = 0; k < uv.Length; k += 2)
        {
            double x = (uv[k] * c - uv[k + 1] * sn) * scale, y = (uv[k] * sn + uv[k + 1] * c) * scale;
            uv[k] = x; uv[k + 1] = y;
            x0 = Math.Min(x0, x); x1 = Math.Max(x1, x);
            y0 = Math.Min(y0, y); y1 = Math.Max(y1, y);
        }
        for (int k = 0; k < uv.Length; k += 2)
        {
            uv[k] -= x0; uv[k + 1] -= y0;
        }
        chart.Width = x1 - x0;
        chart.Height = y1 - y0;
    }

    private static int[] SortedOrder(double[] w, double[] h)
    {
        int n = w.Length;
        int[] order = new int[n];
        double[] keys = new double[n];
        for (int i = 0; i < n; i++)
        {
            order[i] = i;
            keys[i] = -Math.Max(w[i], h[i]);
        }
        Array.Sort(keys, order);
        return order;
    }

    private static bool TryPack(double[] w, double[] h, int[] order, double scale, double pad, double[] ox, double[] oy) =>
        order.Length <= MaxRectsLimit ? TryMaxRects(w, h, order, scale, pad, ox, oy) : TryShelf(w, h, order, scale, pad, ox, oy);

    private static bool TryShelf(double[] w, double[] h, int[] order, double scale, double pad, double[] ox, double[] oy)
    {
        double x = 0, y = 0, shelfHeight = 0;
        foreach (int i in order)
        {
            double rw = scale * w[i] + 2 * pad, rh = scale * h[i] + 2 * pad;
            if (rw > 1.0 + 1e-12) return false;
            if (x + rw > 1.0 + 1e-12)
            {
                y += shelfHeight;
                x = 0;
                shelfHeight = 0;
            }
            if (y + rh > 1.0 + 1e-12) return false;
            ox[i] = x; oy[i] = y;
            x += rw;
            shelfHeight = Math.Max(shelfHeight, rh);
        }
        return true;
    }

    private static bool TryMaxRects(double[] w, double[] h, int[] order, double scale, double pad, double[] ox, double[] oy)
    {
        List<Rect> free = [new Rect(0, 0, 1, 1)];
        List<Rect> next = [];
        foreach (int i in order)
        {
            double rw = scale * w[i] + 2 * pad, rh = scale * h[i] + 2 * pad;
            int best = -1;
            double bestShort = double.MaxValue, bestLong = double.MaxValue;
            for (int f = 0; f < free.Count; f++)
            {
                Rect r = free[f];
                if (r.W < rw - 1e-12 || r.H < rh - 1e-12) continue;
                double dw = r.W - rw, dh = r.H - rh;
                double shortSide = Math.Min(dw, dh), longSide = Math.Max(dw, dh);
                bool better = shortSide < bestShort - 1e-12
                    || (Math.Abs(shortSide - bestShort) <= 1e-12 && longSide < bestLong - 1e-12);
                if (better) { best = f; bestShort = shortSide; bestLong = longSide; }
            }
            if (best < 0) return false;
            Rect target = free[best];
            ox[i] = target.X; oy[i] = target.Y;
            Rect placed = new(target.X, target.Y, rw, rh);

            next.Clear();
            foreach (Rect r in free)
            {
                if (!Intersects(r, placed))
                {
                    next.Add(r);
                    continue;
                }
                if (placed.X > r.X) next.Add(new Rect(r.X, r.Y, placed.X - r.X, r.H));
                if (placed.X + placed.W < r.X + r.W) next.Add(new Rect(placed.X + placed.W, r.Y, r.X + r.W - placed.X - placed.W, r.H));
                if (placed.Y > r.Y) next.Add(new Rect(r.X, r.Y, r.W, placed.Y - r.Y));
                if (placed.Y + placed.H < r.Y + r.H) next.Add(new Rect(r.X, placed.Y + placed.H, r.W, r.Y + r.H - placed.Y - placed.H));
            }
            free.Clear();
            for (int a = 0; a < next.Count; a++)
            {
                bool contained = false;
                for (int b = 0; b < next.Count && !contained; b++)
                    if (a != b && Contains(next[b], next[a]) && (!Contains(next[a], next[b]) || b < a)) contained = true;
                if (!contained && next[a].W > 1e-12 && next[a].H > 1e-12) free.Add(next[a]);
            }
        }
        return true;
    }

    private static bool Intersects(Rect a, Rect b) =>
        a.X < b.X + b.W - 1e-12 && b.X < a.X + a.W - 1e-12 && a.Y < b.Y + b.H - 1e-12 && b.Y < a.Y + a.H - 1e-12;

    private static bool Contains(Rect outer, Rect inner) =>
        inner.X >= outer.X - 1e-12 && inner.Y >= outer.Y - 1e-12 && inner.X + inner.W <= outer.X + outer.W + 1e-12
        && inner.Y + inner.H <= outer.Y + outer.H + 1e-12;

    private readonly record struct Rect(double X, double Y, double W, double H);
}
