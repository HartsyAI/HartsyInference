namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>Orients, scales and packs charts into the unit square at uniform texel density (UV area proportional to surface area).
/// Small chart counts use MaxRects best-short-side-fit; large counts use a height-sorted shelf packer.</summary>
internal static class UvAtlasPacker
{
    private const int RotationSteps = 24;
    private const int MaxRectsLimit = 256;
    private const int SearchIterations = 18;

    /// <summary>A chart whose longest side is under this many texels at the packed scale keeps only the minimum gutter: its few texels are
    /// inpainted from the neighbours anyway, and a full gutter around every sliver would cost more atlas than the slivers themselves.</summary>
    private const double SmallChartTexels = 8.0;

    /// <summary>Rewrites every chart's <see cref="UvChart.Uv"/> to final atlas coordinates inside [0,1] and returns the gutter kept, in texels.
    /// The gutter is halved while the charts do not fit at a useful scale, down to <paramref name="minPaddingTexels"/>; below that the scale
    /// shrinks, so charts stay a full gutter apart.</summary>
    public static double Pack(IReadOnlyList<UvChart> charts, int paddingTexels, int minPaddingTexels, int textureSize)
    {
        double paddingUv = (double)paddingTexels / textureSize;
        double minPad = (double)minPaddingTexels / textureSize;
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
        double smallSide = SmallChartTexels / textureSize;
        double[] ox = new double[n], oy = new double[n];
        double scale = 0;
        while (true)
        {
            double hi = 1.0 / Math.Sqrt(Math.Max(areaSum, 1e-300));
            double lo = 0;
            for (int it = 0; it < SearchIterations; it++)
            {
                double mid = 0.5 * (lo + hi);
                if (TryPack(w, h, order, mid, new Gutters(pad, minPad, smallSide), ox, oy)) lo = mid; else hi = mid;
            }
            bool acceptable = lo >= 0.2 * (1.0 / Math.Sqrt(Math.Max(areaSum, 1e-300))) || pad <= minPad;
            if (acceptable)
            {
                scale = lo;
                break;
            }
            pad = Math.Max(pad * 0.5, minPad);
            if (pad < 1e-9) pad = 0;
        }
        if (scale <= 0)
            scale = 1e-6;
        Gutters gutters = new(pad, minPad, smallSide);
        if (!TryPack(w, h, order, scale, gutters, ox, oy))
            throw new InvalidOperationException("Atlas packing failed.");

        for (int i = 0; i < n; i++)
        {
            double[] uv = charts[i].Uv;
            double gutter = gutters.Of(scale, w[i], h[i]);
            for (int k = 0; k < uv.Length; k += 2)
            {
                uv[k] = ox[i] + gutter + scale * uv[k];
                uv[k + 1] = oy[i] + gutter + scale * uv[k + 1];
            }
        }

        return pad * textureSize;
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
        for (int i = 0; i < n; i++)
            order[i] = i;
        // Longest side first, lowest chart index on ties: Array.Sort alone would leave equal keys in an unspecified order.
        Array.Sort(order, (a, b) =>
        {
            int byLongSide = Math.Max(w[b], h[b]).CompareTo(Math.Max(w[a], h[a]));
            return byLongSide != 0 ? byLongSide : a.CompareTo(b);
        });
        return order;
    }

    private readonly record struct Gutters(double Pad, double MinPad, double SmallSide)
    {
        public double Of(double scale, double w, double h) => scale * Math.Max(w, h) < SmallSide ? MinPad : Pad;
    }

    private static bool TryPack(double[] w, double[] h, int[] order, double scale, Gutters g, double[] ox, double[] oy) =>
        order.Length <= MaxRectsLimit ? TryMaxRects(w, h, order, scale, g, ox, oy) : TrySkyline(w, h, order, scale, g, ox, oy);

    // Bottom-left skyline: each rectangle goes where its bottom edge ends lowest (leftmost on ties), so a row of small charts fills the gaps
    // beside tall ones that a shelf packer leaves empty. The skyline stays a few hundred segments long, so thousands of charts pack quickly.
    private static bool TrySkyline(double[] w, double[] h, int[] order, double scale, Gutters g, double[] ox, double[] oy)
    {
        List<(double X, double Width, double Y)> line = [(0, 1, 0)];
        foreach (int i in order)
        {
            double pad = g.Of(scale, w[i], h[i]);
            double rw = scale * w[i] + 2 * pad, rh = scale * h[i] + 2 * pad;
            if (rw > 1.0 + 1e-12) return false;
            int bestStart = -1;
            double bestX = 0, bestY = double.MaxValue;
            for (int start = 0; start < line.Count; start++)
            {
                double x = line[start].X;
                if (x + rw > 1.0 + 1e-12) break;
                double y = 0, covered = 0;
                for (int s = start; s < line.Count && covered < rw - 1e-12; s++)
                {
                    y = Math.Max(y, line[s].Y);
                    covered += line[s].Width;
                }
                if (y + rh > 1.0 + 1e-12 || y >= bestY - 1e-12) continue;
                bestStart = start; bestX = x; bestY = y;
            }
            if (bestStart < 0) return false;
            ox[i] = bestX; oy[i] = bestY;

            List<(double X, double Width, double Y)> updated = new(line.Count + 2);
            double right = bestX + rw;
            bool inserted = false;
            foreach ((double X, double Width, double Y) seg in line)
            {
                double segRight = seg.X + seg.Width;
                if (segRight <= bestX + 1e-12 || seg.X >= right - 1e-12)
                {
                    if (!inserted && seg.X >= right - 1e-12)
                    {
                        updated.Add((bestX, rw, bestY + rh));
                        inserted = true;
                    }
                    updated.Add(seg);
                    continue;
                }
                if (seg.X < bestX - 1e-12) updated.Add((seg.X, bestX - seg.X, seg.Y));
                if (!inserted)
                {
                    updated.Add((bestX, rw, bestY + rh));
                    inserted = true;
                }
                if (segRight > right + 1e-12) updated.Add((right, segRight - right, seg.Y));
            }
            if (!inserted) updated.Add((bestX, rw, bestY + rh));
            line = Merge(updated);
        }
        return true;
    }

    private static List<(double X, double Width, double Y)> Merge(List<(double X, double Width, double Y)> line)
    {
        List<(double X, double Width, double Y)> merged = new(line.Count);
        foreach ((double X, double Width, double Y) seg in line)
        {
            if (merged.Count > 0 && Math.Abs(merged[^1].Y - seg.Y) <= 1e-12)
                merged[^1] = (merged[^1].X, merged[^1].Width + seg.Width, seg.Y);
            else
                merged.Add(seg);
        }
        return merged;
    }

    private static bool TryShelf(double[] w, double[] h, int[] order, double scale, Gutters g, double[] ox, double[] oy)
    {
        double x = 0, y = 0, shelfHeight = 0;
        foreach (int i in order)
        {
            double pad = g.Of(scale, w[i], h[i]);
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

    private static bool TryMaxRects(double[] w, double[] h, int[] order, double scale, Gutters g, double[] ox, double[] oy)
    {
        List<Rect> free = [new Rect(0, 0, 1, 1)];
        List<Rect> next = [];
        foreach (int i in order)
        {
            double pad = g.Of(scale, w[i], h[i]);
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
