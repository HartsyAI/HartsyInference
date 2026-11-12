namespace HartsyInference.ThreeD.Models.Hunyuan3D.Part;

/// <summary>The geometry half of P3-SAM's automatic segmentation (<c>mesh_sam</c> and its helpers in <c>P3-SAM/demo/auto_mask.py</c>): turns one binary mask per prompt
/// point into labelled mesh faces and part boxes. It is a transcription, step for step, of the upstream heuristics (the thresholds are upstream's): rank the masks by
/// predicted IoU, merge masks that overlap by more than 0.9 (NMS), keep clusters of more than two masks, merge those whose point boxes overlap by more than 0.5,
/// add masks that cover the still-unmasked points by more than 0.7, paint the points from the largest part to the smallest, vote a label per face from its
/// sampled points, fill unlabelled faces from their neighbours, drop connected regions below 0.1 % of their component, fill again, give large unlabelled regions
/// their own part and merge small ones into the neighbouring part whose box grows least, and optionally merge small regions into the biggest neighbour.
/// <para>Labels use -1 for "no mask" and -2 for "no sampled point" as upstream does.</para></summary>
public static class P3SamPartLabeler
{
    private const double NmsIou = 0.9, ClusterBoxIou = 0.5, GapCover = 0.7, MinRegionShare = 0.001;

    /// <summary>Labelled faces and the part boxes.</summary>
    /// <param name="FaceIds">One label per face; negative means none.</param>
    /// <param name="PartIds">The labels of the parts, ascending (the order of <paramref name="Boxes"/>).</param>
    /// <param name="Boxes">Per part <c>[minX, minY, minZ, maxX, maxY, maxZ]</c> over the vertices of its faces.</param>
    public sealed record Result(int[] FaceIds, int[] PartIds, float[][] Boxes);

    /// <summary>Runs the pipeline. <paramref name="points"/> (<c>[P, 3]</c>) are the sampled surface points the masks refer to, <paramref name="pointFace"/> the face each
    /// point was sampled from, <paramref name="masks"/> one boolean mask per prompt over the points and <paramref name="iou"/> its predicted IoU.</summary>
    public static Result Label(float[] vertices, int[] faces, float[] points, int[] pointFace, bool[][] masks, float[] iou, bool postProcess = true, double threshold = 0.95)
    {
        int pointCount = points.Length / 3, faceCount = faces.Length / 3;
        int[][] adjacency = Adjacency(faces, faceCount);
        double[] faceAreas = FaceAreas(vertices, faces, faceCount);

        // Rank by predicted IoU, highest first; ties keep their order (Python's sorted is stable).
        int[] rank = [.. Enumerable.Range(0, masks.Length).OrderByDescending(i => iou[i])];
        bool[][] sorted = [.. rank.Select(i => masks[i])];

        // NMS: a mask joins the first earlier cluster it overlaps by more than 0.9, otherwise it opens a cluster.
        List<(int Head, int Count)> clusters = [];
        for (int i = 0; i < sorted.Length; i++)
        {
            bool joined = false;
            for (int c = 0; c < clusters.Count; c++)
                if (MaskIou(sorted[i], sorted[clusters[c].Head]) > NmsIou)
                {
                    clusters[c] = (clusters[c].Head, clusters[c].Count + 1);
                    joined = true;
                    break;
                }

            if (!joined)
                clusters.Add((i, 1));
        }

        int[] filtered = [.. clusters.Where(c => c.Count > 2).Select(c => c.Head)];
        List<int> merged = [];
        bool[] absorbed = new bool[filtered.Length];
        for (int a = 0; a < filtered.Length; a++)
        {
            if (absorbed[a])
                continue;
            merged.Add(filtered[a]);
            for (int b = a + 1; b < filtered.Length; b++)
                if (!absorbed[b] && BoxIou(points, sorted[filtered[b]], sorted[filtered[a]]) > ClusterBoxIou)
                    absorbed[b] = true;
        }

        // Points still without a mask, and masks that cover them.
        bool[] noMask = new bool[pointCount];
        Array.Fill(noMask, true);
        HashSet<int> kept = [.. merged];
        foreach (int m in merged)
            for (int p = 0; p < pointCount; p++)
                if (sorted[m][p])
                    noMask[p] = false;
        for (int m = 0; m < sorted.Length; m++)
        {
            if (kept.Contains(m))
                continue;
            int total = 0, uncovered = 0;
            for (int p = 0; p < pointCount; p++)
                if (sorted[m][p])
                {
                    total++;
                    if (noMask[p])
                        uncovered++;
                }

            if (total > 0 && (double)uncovered / total > GapCover)
            {
                kept.Add(m);
                merged.Add(m);
                for (int p = 0; p < pointCount; p++)
                    if (sorted[m][p])
                        noMask[p] = false;
            }
        }

        // Paint from the largest mask to the smallest: smaller parts overwrite larger ones.
        int[] finalMasks = [.. merged.OrderByDescending(m => sorted[m].Count(v => v))];
        int[] pointLabel = new int[pointCount];
        Array.Fill(pointLabel, -1);
        foreach (int m in finalMasks)
            for (int p = 0; p < pointCount; p++)
                if (sorted[m][p])
                    pointLabel[p] = m;

        // Face labels: the most frequent label among a face's points (ties go to the lowest label, as np.bincount and argmax do).
        int[] faceIds = new int[faceCount];
        Array.Fill(faceIds, -2);
        Dictionary<int, List<int>> votes = [];
        for (int p = 0; p < pointCount; p++)
        {
            int f = pointFace[p];
            if (!votes.TryGetValue(f, out List<int>? list))
                votes[f] = list = [];
            list.Add(pointLabel[p]);
        }

        foreach ((int face, List<int> labels) in votes)
            faceIds[face] = MajorityShifted(labels, 2);

        // First repair on labels shifted by one so that "no mask" (-1) counts as a label and only unsampled faces (-2) are filled.
        for (int f = 0; f < faceCount; f++)
            faceIds[f]++;
        FixLabel(faceIds, adjacency);
        for (int f = 0; f < faceCount; f++)
            faceIds[f]--;

        // Connected regions of equal label, and of the whole mesh.
        List<List<int>> parts = ConnectedRegions(faceIds, adjacency, out _);
        int[] ones = new int[faceCount];
        Array.Fill(ones, 1);
        List<List<int>> components = ConnectedRegions(ones, adjacency, out int[] componentOf);
        double totalArea = faceAreas.Sum();
        double[] componentArea = [.. components.Select(c => c.Sum(f => faceAreas[f]))];
        (List<int> Faces, double Area, double ComponentArea)[] ranked = [.. parts
            .Select(p => (Faces: p, Area: p.Sum(f => faceAreas[f]), ComponentArea: componentArea[componentOf[p[0]]]))
            .OrderByDescending(p => p.Area)];

        int[] faceIds2 = (int[])faceIds.Clone();
        foreach ((List<int> faceList, double area, double componentArea2) in ranked)
            if (!(area / (componentArea2 + 1e-7) > MinRegionShare))
                foreach (int f in faceList)
                    faceIds2[f] = -1;

        // Second repair: every face below zero (no mask, or dropped) takes its neighbours' majority label.
        int[] faceIds3 = (int[])faceIds2.Clone();
        FixLabel(faceIds3, adjacency);

        List<List<int>> parts2 = ConnectedRegions(faceIds3, adjacency, out _);
        double[] partAreas2 = [.. parts2.Select(p => p.Sum(f => faceAreas[f]))];
        int[] partIds2 = [.. parts2.Select(p => faceIds3[p[0]])];
        int maxId = partIds2.Length == 0 ? -1 : partIds2.Max();
        for (int i = 0; i < parts2.Count; i++)
            if (partAreas2[i] / totalArea > MinRegionShare && partIds2[i] is -1 or -2)
                partIds2[i] = ++maxId;

        int[] faceIds4 = (int[])faceIds3.Clone();
        for (int i = 0; i < parts2.Count; i++)
            foreach (int f in parts2[i])
                faceIds4[f] = partIds2[i];

        Dictionary<int, double[]> idBox = [];
        foreach (int id in faceIds4.Where(v => v >= 0).Distinct())
            idBox[id] = Box(vertices, faces, Enumerable.Range(0, faceCount).Where(f => faceIds4[f] == id));
        double[][] partBox = [.. parts2.Select(p => Box(vertices, faces, p))];
        int[][] neighbours = NeighbourParts(parts2, adjacency, partBox, partIds2);

        // Merge regions without a mask into the neighbouring part whose box would grow least.
        for (int i = 0; i < parts2.Count; i++)
        {
            if (partIds2[i] is not (-1 or -2))
                continue;
            double best = 1e10;
            int bestId = -1;
            foreach (int j in neighbours[i])
            {
                if (j < 0 || partIds2[j] is -1 or -2)
                    continue;
                double[] target = idBox[partIds2[j]];
                double increase = BoxIncrease(target, partBox[i]);
                if (best > increase)
                {
                    best = increase;
                    bestId = partIds2[j];
                }
            }

            if (bestId >= 0)
                partIds2[i] = bestId;
        }

        faceIds4 = (int[])faceIds3.Clone();
        for (int i = 0; i < parts2.Count; i++)
            foreach (int f in parts2[i])
                faceIds4[f] = partIds2[i];
        int[] result = faceIds4;

        if (postProcess)
            result = SmallRegionMerge(faceAreas, ConnectedRegions(faceIds4, adjacency, out _), adjacency, faceIds4, threshold);

        int[] ids = [.. result.Where(v => v >= 0).Distinct().OrderBy(v => v)];
        float[][] boxes = [.. ids.Select(id => Box(vertices, faces, Enumerable.Range(0, faceCount).Where(f => result[f] == id)).Select(v => (float)v).ToArray())];
        return new Result(result, ids, boxes);
    }

    // The label with the most votes after adding `shift` (labels are -2 or more): np.argmax(np.bincount(labels + shift)) - shift.
    private static int MajorityShifted(List<int> labels, int shift)
    {
        Dictionary<int, int> counts = [];
        foreach (int l in labels)
            counts[l + shift] = counts.GetValueOrDefault(l + shift) + 1;
        int best = int.MaxValue, bestCount = -1;
        foreach ((int key, int count) in counts)
            if (count > bestCount || (count == bestCount && key < best))
            {
                best = key;
                bestCount = count;
            }

        return best - shift;
    }

    // fix_label: faces below zero take the most frequent non-negative label among their neighbours; repeats up to 50 passes, labels updated in place.
    internal static void FixLabel(int[] faceIds, int[][] adjacency)
    {
        List<int> pending = [.. Enumerable.Range(0, faceIds.Length).Where(f => faceIds[f] < 0)];
        for (int loop = 1; loop <= 50; loop++)
        {
            bool changed = false;
            List<int> next = [];
            foreach (int f in pending)
            {
                if (f >= adjacency.Length)
                    continue;
                List<int> labels = [];
                foreach (int n in adjacency[f])
                    if (faceIds[n] >= 0)
                        labels.Add(faceIds[n]);
                if (labels.Count == 0)
                {
                    next.Add(f);
                    continue;
                }

                faceIds[f] = MajorityShifted(labels, 0);
                changed = true;
            }

            pending = next;
            if (!changed)
                break;
        }
    }

    // get_connected_region: breadth-first regions of faces that share a label and an edge, in face order.
    internal static List<List<int>> ConnectedRegions(int[] faceIds, int[][] adjacency, out int[] partOf)
    {
        bool[] seen = new bool[faceIds.Length];
        partOf = new int[faceIds.Length];
        Array.Fill(partOf, -1);
        List<List<int>> parts = [];
        for (int i = 0; i < faceIds.Length; i++)
        {
            if (seen[i])
                continue;
            List<int> part = [];
            Queue<int> queue = new([i]);
            while (queue.Count > 0)
            {
                int face = queue.Dequeue();
                if (seen[face])
                    continue;
                seen[face] = true;
                part.Add(face);
                partOf[face] = parts.Count;
                if (face >= adjacency.Length)
                    continue;
                foreach (int n in adjacency[face])
                    if (!seen[n] && faceIds[n] == faceIds[face])
                        queue.Enqueue(n);
            }

            parts.Add(part);
        }

        return parts;
    }

    // find_neighbor_part: the parts that share an edge with each part; a part without a mask that touches nothing takes the nearest part by box distance.
    private static int[][] NeighbourParts(List<List<int>> parts, int[][] adjacency, double[][]? boxes, int[]? ids)
    {
        Dictionary<int, int> face2part = [];
        for (int i = 0; i < parts.Count; i++)
            foreach (int f in parts[i])
                face2part[f] = i;
        int[][] result = new int[parts.Count][];
        for (int i = 0; i < parts.Count; i++)
        {
            HashSet<int> set = [];
            foreach (int f in parts[i])
            {
                if (f >= adjacency.Length)
                    continue;
                foreach (int n in adjacency[f])
                {
                    if (!face2part.TryGetValue(n, out int other) || other == i)
                        continue;
                    if (ids is not null && ids[other] is -1 or -2)
                        continue;
                    set.Add(other);
                }
            }

            List<int> list = [.. set];
            if (boxes is not null && ids is not null && ids[i] is -1 or -2 && list.Count == 0)
            {
                double minDistance = double.PositiveInfinity;
                int minIndex = -1;
                for (int j = 0; j < parts.Count; j++)
                {
                    if (j == i || ids[j] is -1 or -2)
                        continue;
                    double distance = BoxDistance(boxes[i], boxes[j]);
                    if (distance < minDistance)
                    {
                        minDistance = distance;
                        minIndex = j;
                    }
                    else if (distance == minDistance && BoxVolume(boxes[j]) < BoxVolume(boxes[minIndex]))
                        minIndex = j;
                }

                list = [minIndex];
            }

            result[i] = [.. list];
        }

        return result;
    }

    // do_post_process: a small region (under 1 % of the area) in the tail of the cumulative area (beyond `threshold`) joins the biggest neighbouring region
    // that is not itself in the tail.
    private static int[] SmallRegionMerge(double[] faceAreas, List<List<int>> parts, int[][] adjacency, int[] faceIds, double threshold)
    {
        double total = faceAreas.Sum();
        (List<int> Faces, double Share)[] ordered = [.. parts.Select(p => (Faces: p, Share: p.Sum(f => faceAreas[f]) / total)).OrderByDescending(p => p.Share)];
        List<List<int>> sortedParts = [.. ordered.Select(o => o.Faces)];
        double[] shares = [.. ordered.Select(o => o.Share)];
        double[] cumulative = new double[shares.Length];
        double running = 0;
        for (int i = 0; i < shares.Length; i++)
            cumulative[i] = running += shares[i];
        int[][] neighbours = NeighbourParts(sortedParts, adjacency, null, null);
        int[] result = (int[])faceIds.Clone();
        for (int i = 0; i < sortedParts.Count; i++)
        {
            if (!(cumulative[i] > threshold && shares[i] < 0.01) || neighbours[i].Length == 0)
                continue;
            double maxArea = 0;
            int maxPart = -1;
            foreach (int j in neighbours[i])
            {
                if (cumulative[j] > threshold)
                    continue;
                if (shares[j] > maxArea)
                {
                    maxArea = shares[j];
                    maxPart = j;
                }
            }

            if (maxPart == -1)
                continue;
            int target = faceIds[sortedParts[maxPart][0]];
            sortedParts[maxPart].AddRange(sortedParts[i]);
            foreach (int f in sortedParts[i])
                result[f] = target;
            sortedParts[i] = [];
        }

        return result;
    }

    // Faces that share an edge with exactly one other face, as trimesh's face_adjacency lists them; rows cover faces up to the largest adjacent index.
    internal static int[][] Adjacency(int[] faces, int faceCount)
    {
        Dictionary<(int, int), List<int>> edges = [];
        for (int f = 0; f < faceCount; f++)
            for (int k = 0; k < 3; k++)
            {
                int a = faces[f * 3 + k], b = faces[f * 3 + (k + 1) % 3];
                (int, int) key = a < b ? (a, b) : (b, a);
                if (!edges.TryGetValue(key, out List<int>? list))
                    edges[key] = list = [];
                list.Add(f);
            }

        int rows = 0;
        List<(int, int)> pairs = [];
        foreach (((int, int) key, List<int> list) in edges.OrderBy(e => e.Key))
            if (list.Count == 2)
            {
                pairs.Add((list[0], list[1]));
                rows = Math.Max(rows, Math.Max(list[0], list[1]) + 1);
            }

        List<int>[] lists = new List<int>[rows];
        for (int i = 0; i < rows; i++)
            lists[i] = [];
        foreach ((int a, int b) in pairs)
        {
            lists[a].Add(b);
            lists[b].Add(a);
        }

        return [.. lists.Select(l => l.ToArray())];
    }

    private static double[] FaceAreas(float[] vertices, int[] faces, int faceCount)
    {
        double[] area = new double[faceCount];
        for (int f = 0; f < faceCount; f++)
        {
            int a = faces[f * 3] * 3, b = faces[f * 3 + 1] * 3, c = faces[f * 3 + 2] * 3;
            double ux = vertices[b] - vertices[a], uy = vertices[b + 1] - vertices[a + 1], uz = vertices[b + 2] - vertices[a + 2];
            double vx = vertices[c] - vertices[a], vy = vertices[c + 1] - vertices[a + 1], vz = vertices[c + 2] - vertices[a + 2];
            double cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
            area[f] = 0.5 * Math.Sqrt(cx * cx + cy * cy + cz * cz);
        }

        return area;
    }

    private static double MaskIou(bool[] a, bool[] b)
    {
        int both = 0, either = 0;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] && b[i])
                both++;
            if (a[i] || b[i])
                either++;
        }

        return either == 0 ? double.NaN : (double)both / either;
    }

    // Bounding boxes of the points selected by two masks; the IoU of those boxes (cal_bbox_iou). An empty mask has no box and overlaps nothing.
    private static double BoxIou(float[] points, bool[] a, bool[] b)
    {
        double[]? boxA = PointBox(points, a), boxB = PointBox(points, b);
        if (boxA is null || boxB is null)
            return 0;
        return Iou3d(boxA, boxB);
    }

    private static double[]? PointBox(float[] points, bool[] mask)
    {
        double[] box = [double.MaxValue, double.MaxValue, double.MaxValue, double.MinValue, double.MinValue, double.MinValue];
        bool any = false;
        for (int p = 0; p < mask.Length; p++)
        {
            if (!mask[p])
                continue;
            any = true;
            for (int c = 0; c < 3; c++)
            {
                box[c] = Math.Min(box[c], points[p * 3 + c]);
                box[3 + c] = Math.Max(box[3 + c], points[p * 3 + c]);
            }
        }

        return any ? box : null;
    }

    private static double Iou3d(double[] a, double[] b)
    {
        double ix = Math.Min(a[3], b[3]) - Math.Max(a[0], b[0]), iy = Math.Min(a[4], b[4]) - Math.Max(a[1], b[1]), iz = Math.Min(a[5], b[5]) - Math.Max(a[2], b[2]);
        if (ix <= 0 || iy <= 0 || iz <= 0)
            return 0;
        double inter = ix * iy * iz, union = BoxVolume(a) + BoxVolume(b) - inter;
        return union > 0 ? inter / union : 0;
    }

    private static double BoxVolume(double[] box) => (box[3] - box[0]) * (box[4] - box[1]) * (box[5] - box[2]);

    private static double BoxDistance(double[] a, double[] b)
    {
        double dx = Math.Max(0, Math.Max(b[3] - a[0], a[3] - b[0])), dy = Math.Max(0, Math.Max(b[4] - a[1], a[4] - b[1])), dz = Math.Max(0, Math.Max(b[5] - a[2], a[5] - b[2]));
        return dx == 0 && dy == 0 && dz == 0 ? 0 : Math.Sqrt(dx * dx + dy * dy + dz * dz);
    }

    // aabb_increase and the caller's max(np.max(min_increase), np.max(max_increase)): the relative growth |after - before| / |before| of each side of `before`
    // when joined with `other`. np.max propagates NaN (a zero side gives 0 / 0), Python's max(a, b) keeps a unless b is greater.
    private static double BoxIncrease(double[] before, double[] other)
    {
        double[] low = new double[3], high = new double[3];
        for (int c = 0; c < 3; c++)
        {
            double lo = Math.Min(before[c], other[c]), hi = Math.Max(before[3 + c], other[3 + c]);
            low[c] = Math.Abs(lo - before[c]) / Math.Abs(before[c]);
            high[c] = Math.Abs(hi - before[3 + c]) / Math.Abs(before[3 + c]);
        }

        double a = NumpyMax(low), b = NumpyMax(high);
        return b > a ? b : a;
    }

    private static double NumpyMax(double[] values) => values.Any(double.IsNaN) ? double.NaN : values.Max();

    private static double[] Box(float[] vertices, int[] faces, IEnumerable<int> faceList)
    {
        double[] box = [double.MaxValue, double.MaxValue, double.MaxValue, double.MinValue, double.MinValue, double.MinValue];
        foreach (int f in faceList)
            for (int k = 0; k < 3; k++)
            {
                int v = faces[f * 3 + k] * 3;
                for (int c = 0; c < 3; c++)
                {
                    box[c] = Math.Min(box[c], vertices[v + c]);
                    box[3 + c] = Math.Max(box[3 + c], vertices[v + c]);
                }
            }

        return box;
    }
}
