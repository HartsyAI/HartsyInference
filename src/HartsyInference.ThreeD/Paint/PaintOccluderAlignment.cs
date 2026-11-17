namespace HartsyInference.ThreeD.Paint;

/// <summary>Shifts the reference picture on a nearer part the shape model put off the drawing (a hat brim lower than drawn) so the picture's own edge
/// lands on the part's front-view depth step; the surface behind keeps the placement, and steps without clear evidence keep it too. Only a part
/// the mesh made larger than drawn is moved; one it made smaller shows the surface behind on the surface behind, as drawn.</summary>
internal sealed class PaintOccluderAlignment
{
    /// <summary>Side of the grid the steps and the shift field are worked out on.</summary>
    internal const int GridSize = 1024;

    // Depth jump between neighbouring grid pixels, beyond the surface's own slope, that puts one surface in front of another (camera units of the
    // normalised mesh frame, whose diameter is PaintMeshFrame.DefaultScaleFactor).
    private const float Step = 0.03f;

    // Search reach across a step, and the strip past it that gives each side's typical colours, as shares of the view.
    private const double Reach = 0.04, Strip = 0.012;

    // Least opponent-colour distance between the two sides' typical colours for the picture to show a boundary.
    private const float Contrast = 0.12f;

    // Step pixels within this share of the view pool their evidence; the shift is smoothed along the step over twice this.
    private const double Pool = 0.01;

    // A pooled boundary must explain the colours this much better than the mesh's own edge, and explain most of them.
    private const double MinGain = 0.15, MinFit = 0.5;

    // Acted on only when the picture past the boundary is clearly the nearer part's and the surface behind spans the reach (a face, not a sliver).
    private const double MinNearer = 0.6;

    // Smoothing weight of a step pixel without evidence: it pulls toward no shift only away from evidence.
    private const float SilentWeight = 0.1f;

    private readonly float[] _depth;
    private readonly float[] _shift;

    private PaintOccluderAlignment(float[] depth, float[] shift)
    {
        _depth = depth;
        _shift = shift;
    }

    /// <summary>Works out the shift field for the front view, or returns null when no step needs one.</summary>
    /// <param name="depth">Front-view camera depth at <paramref name="side"/> squared pixels; nearer surfaces have larger values.</param>
    /// <param name="faceIds">Front-view face ids, 0 off the mesh.</param>
    /// <param name="subject">The picture's subject mask.</param>
    public static PaintOccluderAlignment? Create(float[] depth, int[] faceIds, int side, byte[] rgb, bool[] subject, int width, int height,
        PaintReferenceProjection placement, CancellationToken cancellationToken = default)
    {
        const int g = GridSize;
        float[] d = new float[g * g];
        for (int gy = 0; gy < g; gy++)
            for (int gx = 0; gx < g; gx++)
            {
                int p = Math.Min(side - 1, (int)((gy + 0.5) * side / g)) * side + Math.Min(side - 1, (int)((gx + 0.5) * side / g));
                d[gy * g + gx] = faceIds[p] == 0 ? float.NaN : depth[p];
            }

        // Near side of every step, and where a surface ends at the silhouette or passes behind another.
        bool[] edge = new bool[g * g], bounded = new bool[g * g];
        Parallel.For(0, g, new ParallelOptions { CancellationToken = cancellationToken }, y =>
        {
            for (int x = 0; x < g; x++)
            {
                float here = d[y * g + x];
                if (float.IsNaN(here))
                    continue;
                foreach ((int dx, int dy) in (ReadOnlySpan<(int, int)>)[(1, 0), (-1, 0), (0, 1), (0, -1)])
                {
                    float q = At(d, x + dx, y + dy);
                    if (float.IsNaN(q))
                    {
                        bounded[y * g + x] = true;
                        continue;
                    }
                    float q2 = At(d, x + 2 * dx, y + 2 * dy);
                    float beyond = float.IsNaN(q2) ? q : 2 * q - q2;
                    if (here - q > Step && here - beyond > Step)
                        edge[y * g + x] = true;
                    if (q - here > Step && beyond - here > Step)
                        bounded[y * g + x] = true;
                }
            }
        });
        List<int> edges = [];
        for (int p = 0; p < edge.Length; p++)
            if (edge[p])
                edges.Add(p);
        if (edges.Count == 0)
            return null;

        int reach = (int)(Reach * g), strip = (int)(Strip * g), length = 2 * reach + 1;
        float[] normals = new float[edges.Count * 2], profiles = new float[edges.Count * length];
        bool[] evidence = new bool[edges.Count];
        int[] behind = new int[edges.Count];
        Parallel.For(0, edges.Count, new ParallelOptions { CancellationToken = cancellationToken }, i =>
            evidence[i] = Profile(d, edges[i], rgb, subject, width, height, placement, reach, strip, normals.AsSpan(i * 2, 2),
                profiles.AsSpan(i * length, length), out behind[i]));

        float[] offsets = Decide(edges, evidence, behind, profiles, reach, strip, out bool[] decided);
        bool any = false;
        foreach (float offset in offsets)
            any |= offset < -1;
        if (!any)
            return null;

        float[] along = Along(d, edges, normals, offsets, decided);
        float[] field = Spread(d, edge, bounded, edges, along, 3 * reach, cancellationToken);
        return new PaintOccluderAlignment(d, field);
    }

    /// <summary>The shift, in view units, to add to front-view position (<paramref name="u"/>, <paramref name="v"/>) on the surface at camera depth
    /// <paramref name="z"/> before the picture is sampled there.</summary>
    public (double U, double V) Shift(double u, double v, double z)
    {
        const int g = GridSize;
        double gx = u * g - 0.5, gy = v * g - 0.5;
        int x0 = (int)Math.Floor(gx), y0 = (int)Math.Floor(gy);
        double fx = gx - x0, fy = gy - y0;
        // Corners across a step from the texel's own surface are left out of the blend.
        float anchor = float.NaN;
        double closest = 2 * Step;
        for (int k = 0; k < 4; k++)
        {
            float c = At(_depth, x0 + (k & 1), y0 + (k >> 1));
            if (!float.IsNaN(c) && Math.Abs(c - z) < closest)
                (closest, anchor) = (Math.Abs(c - z), c);
        }
        if (float.IsNaN(anchor))
            return (0, 0);
        double su = 0, sv = 0, weight = 0;
        for (int k = 0; k < 4; k++)
        {
            int x = x0 + (k & 1), y = y0 + (k >> 1);
            float c = At(_depth, x, y);
            if (float.IsNaN(c) || Math.Abs(c - anchor) > Step)
                continue;
            double w = ((k & 1) == 1 ? fx : 1 - fx) * ((k >> 1) == 1 ? fy : 1 - fy) + 1e-6;
            su += w * _shift[(y * g + x) * 2];
            sv += w * _shift[(y * g + x) * 2 + 1];
            weight += w;
        }
        return (su / weight, sv / weight);
    }

    private static float At(float[] grid, int x, int y) =>
        x < 0 || y < 0 || x >= GridSize || y >= GridSize ? float.NaN : grid[y * GridSize + x];

    // Normal toward the surface behind, and per position across the step (negative on the nearer part) +1 nearer-part colour to -1 far colour.
    private static bool Profile(float[] d, int pixel, byte[] rgb, bool[] subject, int width, int height, PaintReferenceProjection placement,
        int reach, int strip, Span<float> normal, Span<float> profile, out int behind)
    {
        const int g = GridSize;
        behind = 0;
        int x = pixel % g, y = pixel / g;
        float here = d[pixel];
        double nx = 0, ny = 0;
        for (int dy = -3; dy <= 3; dy++)
            for (int dx = -3; dx <= 3; dx++)
            {
                float w = At(d, x + dx, y + dy);
                if (!float.IsNaN(w) && here - w > Step)
                {
                    nx += dx;
                    ny += dy;
                }
            }
        double norm = Math.Sqrt(nx * nx + ny * ny);
        if (norm < 1e-9)
            return false;
        nx /= norm;
        ny /= norm;
        normal[0] = (float)nx;
        normal[1] = (float)ny;

        float Depth(int s) => At(d, (int)Math.Round(x + s * nx), (int)Math.Round(y + s * ny));
        // Farthest position the surface continues to, following its own slope.
        int Walk(int sign, int start, int limit)
        {
            float previous = Depth(sign * start);
            float slope = 0;
            int last = start;
            for (int s = start + 1; s <= limit; s++)
            {
                float next = Depth(sign * s);
                if (float.IsNaN(next) || Math.Abs(next - previous - slope) > Step)
                    break;
                if (s > start + 1)
                    slope = next - previous;
                previous = next;
                last = s;
            }
            return last;
        }

        int nearLength = Walk(-1, 0, reach + strip);
        float first = Depth(1);
        int farStart = !float.IsNaN(first) && here - first >= Step * 0.5f ? 1 : 2;
        if (float.IsNaN(Depth(farStart)))
            return false;
        int farLength = Walk(1, farStart, reach + strip);
        behind = farLength;
        int nearReach = Math.Min(reach, (int)(0.75 * (nearLength + 1))), farReach = Math.Min(reach, (int)(0.75 * (farLength + 1)));
        int nearFrom = Math.Min(nearLength, nearReach + strip), farTo = Math.Min(farLength, farReach + strip);
        if (nearFrom - nearReach < 2 || farTo - farReach < 2 || nearReach + farReach < 2)
            return false;

        Span<float> nearColours = stackalloc float[(nearFrom - nearReach + 1) * 3], farColours = stackalloc float[(farTo - farReach + 1) * 3];
        Span<float> colour = stackalloc float[3];
        int nearCount = 0, farCount = 0;
        for (int s = nearReach; s <= nearFrom; s++)
            if (SamplePicture(-s, x, y, nx, ny, rgb, subject, width, height, placement, colour))
                colour.CopyTo(nearColours[(nearCount++ * 3)..]);
        for (int s = farReach; s <= farTo; s++)
            if (SamplePicture(s, x, y, nx, ny, rgb, subject, width, height, placement, colour))
                colour.CopyTo(farColours[(farCount++ * 3)..]);
        if (nearCount < 3 || farCount < 3)
            return false;
        nearColours = nearColours[..(nearCount * 3)];
        farColours = farColours[..(farCount * 3)];
        Span<float> nearTypical = stackalloc float[3], farTypical = stackalloc float[3];
        Median(nearColours, nearTypical);
        Median(farColours, farTypical);
        if (Distance(nearTypical, farTypical) < Contrast)
            return false;
        for (int s = -nearReach; s <= farReach; s++)
        {
            if (!SamplePicture(s, x, y, nx, ny, rgb, subject, width, height, placement, colour))
                continue;
            float toNear = Nearest(colour, nearColours), toFar = Nearest(colour, farColours);
            profile[s + reach] = Math.Clamp((toFar - toNear) / (toFar + toNear + 1e-6f), -1f, 1f);
        }
        return true;
    }

    // Opponent colour of the picture at position s across the step; false outside the subject.
    private static bool SamplePicture(int s, int x, int y, double nx, double ny, byte[] rgb, bool[] subject, int width, int height,
        PaintReferenceProjection placement, Span<float> opponent)
    {
        double u = (x + 0.5 + s * nx) / GridSize, v = (y + 0.5 + s * ny) / GridSize;
        double px = placement.Scale * u + placement.OffsetX - 0.5, py = placement.Scale * v + placement.OffsetY - 0.5;
        int ix = (int)Math.Round(px), iy = (int)Math.Round(py);
        if (ix < 0 || iy < 0 || ix >= width || iy >= height || !subject[iy * width + ix])
            return false;
        Span<float> c = stackalloc float[3];
        PaintReferenceProjector.SampleBilinear(rgb, width, height, px, py, c);
        // Ink and shading change brightness, parts differ in hue: brightness counts half.
        opponent[0] = 0.5f * (c[0] + c[1] + c[2]) / 3f;
        opponent[1] = c[0] - c[1];
        opponent[2] = c[2] - 0.5f * (c[0] + c[1]);
        return true;
    }

    private static void Median(ReadOnlySpan<float> colours, Span<float> target)
    {
        int count = colours.Length / 3;
        Span<float> channel = stackalloc float[count];
        for (int c = 0; c < 3; c++)
        {
            for (int k = 0; k < count; k++)
                channel[k] = colours[k * 3 + c];
            channel.Sort();
            target[c] = count % 2 == 1 ? channel[count / 2] : 0.5f * (channel[count / 2 - 1] + channel[count / 2]);
        }
    }

    private static float Nearest(ReadOnlySpan<float> colour, ReadOnlySpan<float> references)
    {
        float best = float.MaxValue;
        for (int k = 0; k < references.Length; k += 3)
            best = Math.Min(best, Distance(colour, references.Slice(k, 3)));
        return best;
    }

    private static float Distance(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2];
        return MathF.Sqrt(x * x + y * y + z * z);
    }

    // Offset (grid pixels) of the picture's boundary from each step pixel, from pooled profiles; negative inside the nearer part.
    private static float[] Decide(List<int> edges, bool[] evidence, int[] behind, float[] profiles, int reach, int strip, out bool[] decided)
    {
        const int g = GridSize;
        int radius = Math.Max(1, (int)(Pool * g)), length = 2 * reach + 1;
        List<int>[] buckets = Buckets(edges, evidence, radius, out int cells);
        float[] offsets = new float[edges.Count];
        bool[] done = new bool[edges.Count];
        Parallel.For(0, edges.Count, () => new double[length + 1], (i, _, pooled) =>
        {
            if (!evidence[i])
                return pooled;
            int x = edges[i] % g, y = edges[i] / g, count = 0;
            Array.Clear(pooled);
            for (int cy = Math.Max(0, y / radius - 1); cy <= Math.Min(cells - 1, y / radius + 1); cy++)
                for (int cx = Math.Max(0, x / radius - 1); cx <= Math.Min(cells - 1, x / radius + 1); cx++)
                    if (buckets[cy * cells + cx] is { } bucket)
                        foreach (int j in bucket)
                            if (Math.Abs(edges[j] % g - x) <= radius && Math.Abs(edges[j] / g - y) <= radius)
                            {
                                count++;
                                for (int k = 0; k < length; k++)
                                    pooled[k + 1] += profiles[j * length + k];
                            }
            if (count < radius)
                return pooled;
            // Prefix sums; score(t) = evidence before t minus evidence after it, t the first position past the boundary.
            double total = 0;
            for (int k = 1; k <= length; k++)
            {
                total += Math.Abs(pooled[k]);
                pooled[k] += pooled[k - 1];
            }
            double sum = pooled[length];
            int best = 0;
            for (int t = 1; t <= length; t++)
                if (2 * pooled[t] - sum > 2 * pooled[best] - sum)
                    best = t;
            int edgeT = reach + 1;
            double gain = (2 * pooled[best] - 2 * pooled[edgeT]) / (total + 1e-9), fit = (2 * pooled[best] - sum) / (total + 1e-9);
            if (gain <= MinGain || fit <= MinFit)
            {
                done[i] = true;
                return pooled;
            }
            int from = Math.Max(0, best - strip);
            double nearer = best < edgeT && best > 0 ? (pooled[best] - pooled[from]) / (best - from) / count : 0;
            if (nearer >= MinNearer && behind[i] >= reach)
            {
                offsets[i] = best - reach - 1;
                done[i] = true;
            }
            return pooled;
        }, _ => { });
        decided = done;
        return offsets;
    }

    // Each step pixel's shift (grid pixels): its offset along its normal, a little past the picture's boundary so the rim takes the nearer part's
    // own drawn outline, averaged twice over the step pixels of the same step within twice Pool, weighted by evidence.
    private static float[] Along(float[] d, List<int> edges, float[] normals, float[] offsets, bool[] decided)
    {
        const int g = GridSize;
        // Steps are told apart as 8-connected runs of step pixels at continuing depth, so a brim's step never averages with the face's behind it.
        int[] index = new int[g * g], run = new int[edges.Count];
        Array.Fill(index, -1);
        Array.Fill(run, -1);
        for (int i = 0; i < edges.Count; i++)
            index[edges[i]] = i;
        Stack<int> stack = new();
        for (int i = 0; i < edges.Count; i++)
        {
            if (run[i] >= 0)
                continue;
            run[i] = i;
            stack.Push(i);
            while (stack.Count > 0)
            {
                int k = stack.Pop(), x = edges[k] % g, y = edges[k] / g;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        if (x + dx < 0 || y + dy < 0 || x + dx >= g || y + dy >= g)
                            continue;
                        int j = index[(y + dy) * g + x + dx];
                        if (j >= 0 && run[j] < 0 && Math.Abs(d[edges[j]] - d[edges[k]]) <= Step)
                        {
                            run[j] = i;
                            stack.Push(j);
                        }
                    }
            }
        }

        float[] shift = new float[edges.Count * 2], weight = new float[edges.Count];
        for (int i = 0; i < edges.Count; i++)
        {
            float offset = offsets[i] < -1 ? offsets[i] - Math.Max(1f, 0.1f * -offsets[i]) : 0f;
            weight[i] = decided[i] ? 1f : SilentWeight;
            shift[i * 2] = offset * normals[i * 2];
            shift[i * 2 + 1] = offset * normals[i * 2 + 1];
        }
        int radius = (int)(2 * Pool * g);
        List<int>[] buckets = Buckets(edges, null, radius, out int cells);
        for (int pass = 0; pass < 2; pass++)
        {
            float[] source = shift, next = new float[edges.Count * 2];
            Parallel.For(0, edges.Count, i =>
            {
                int x = edges[i] % g, y = edges[i] / g;
                double sx = 0, sy = 0, sw = 0;
                for (int cy = Math.Max(0, y / radius - 1); cy <= Math.Min(cells - 1, y / radius + 1); cy++)
                    for (int cx = Math.Max(0, x / radius - 1); cx <= Math.Min(cells - 1, x / radius + 1); cx++)
                        if (buckets[cy * cells + cx] is { } bucket)
                            foreach (int j in bucket)
                                if (run[j] == run[i] && Math.Abs(edges[j] % g - x) <= radius && Math.Abs(edges[j] / g - y) <= radius)
                                {
                                    sx += weight[j] * source[j * 2];
                                    sy += weight[j] * source[j * 2 + 1];
                                    sw += weight[j];
                                }
                next[i * 2] = (float)(sx / sw);
                next[i * 2 + 1] = (float)(sy / sw);
            });
            shift = next;
        }
        return shift;
    }

    // Step pixels (those `include` marks, or all) bucketed by cells of the given side, for neighbour searches within that distance.
    private static List<int>[] Buckets(List<int> edges, bool[]? include, int radius, out int cells)
    {
        const int g = GridSize;
        cells = (g + radius - 1) / radius;
        List<int>[] buckets = new List<int>[cells * cells];
        for (int i = 0; i < edges.Count; i++)
            if (include is null || include[i])
                (buckets[edges[i] / g / radius * cells + edges[i] % g / radius] ??= []).Add(i);
        return buckets;
    }

    // Shift field (two floats per grid pixel, view units) over the nearer parts: each pixel takes the shift of the step it is nearest to over its
    // own surface, faded with distance from the step and toward the part's other ends.
    private static float[] Spread(float[] d, bool[] edge, bool[] bounded, List<int> edges, float[] along, int reach,
        CancellationToken cancellationToken)
    {
        const int g = GridSize;
        int[] source = new int[g * g];
        Array.Fill(source, -1);
        float[] distance = Distances(d, edges, reach, source, edge);
        List<int> boundary = [];
        for (int p = 0; p < bounded.Length; p++)
            if (bounded[p])
                boundary.Add(p);
        float[] toBoundary = Distances(d, boundary, reach, null, null);
        cancellationToken.ThrowIfCancellationRequested();

        float[] shift = new float[g * g * 2];
        for (int p = 0; p < source.Length; p++)
        {
            int i = source[p];
            if (i < 0)
                continue;
            double fx = along[i * 2], fy = along[i * 2 + 1];
            double magnitude = Math.Sqrt(fx * fx + fy * fy), near = Math.Min(distance[p], 1e6), far = Math.Min(toBoundary[p], 1e6);
            double fade = Math.Clamp(1 - near / Math.Max(3 * magnitude, 1e-6), 0, 1) * far / (near + far + 1e-9);
            shift[p * 2] = (float)(fx * fade / g);
            shift[p * 2 + 1] = (float)(fy * fade / g);
        }
        return shift;
    }

    // Distance from the seeds over continuing surface, up to `reach`; records each pixel's seed in `source`, never enters `stop`.
    private static float[] Distances(float[] d, List<int> seeds, int reach, int[]? source, bool[]? stop)
    {
        const int g = GridSize;
        float[] distance = new float[g * g];
        Array.Fill(distance, float.PositiveInfinity);
        Queue<int> queue = new();
        for (int i = 0; i < seeds.Count; i++)
        {
            distance[seeds[i]] = 0;
            if (source is not null)
                source[seeds[i]] = i;
            queue.Enqueue(seeds[i]);
        }
        while (queue.Count > 0)
        {
            int p = queue.Dequeue(), x = p % g, y = p / g;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    int nx = x + dx, ny = y + dy;
                    if ((dx == 0 && dy == 0) || nx < 0 || ny < 0 || nx >= g || ny >= g)
                        continue;
                    int q = ny * g + nx;
                    if (float.IsNaN(d[q]) || Math.Abs(d[q] - d[p]) > Step || (stop is not null && stop[q]))
                        continue;
                    float next = distance[p] + (dx != 0 && dy != 0 ? 1.4142f : 1f);
                    if (next < distance[q] && next < reach)
                    {
                        distance[q] = next;
                        if (source is not null)
                            source[q] = source[p];
                        queue.Enqueue(q);
                    }
                }
        }
        return distance;
    }
}
