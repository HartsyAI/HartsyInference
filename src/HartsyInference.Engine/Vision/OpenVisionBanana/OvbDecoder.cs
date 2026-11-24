namespace HartsyInference.Engine.Vision.OpenVisionBanana;

/// <summary>Turns Open Vision Banana's painted answer into labels, depth and normals, after the release's evaluation code
/// (<c>common/semseg/diversify.py</c>, the semantic-segmentation voting in <c>evaluation/semseg</c>,
/// <c>common/surface_normal_encoding.py</c>, and the depth evaluator's grayscale read). Pure CPU on RGB24 buffers.</summary>
public static class OvbDecoder
{
    /// <summary>The label of a pixel that snapped to black (or to nothing): not a class.</summary>
    public const byte Ignore = 255;

    /// <summary>Snaps every pixel to its nearest legend colour by squared RGB distance, with black as an extra target that
    /// decodes to <see cref="Ignore"/> so the model's background never lands on a dark class. A tie goes to the lowest class,
    /// and to a class over black. Returns the label map and the squared distance to the winning target.</summary>
    public static (byte[] Labels, float[] Distance) SnapToLegend(ReadOnlySpan<byte> rgb, int width, int height,
        IReadOnlyList<(byte R, byte G, byte B)> colors)
    {
        ArgumentNullException.ThrowIfNull(colors);
        CheckRgb(rgb, width, height);
        if (colors.Count is < 1 or >= Ignore)
            throw new ArgumentOutOfRangeException(nameof(colors), colors.Count, $"A legend holds 1 to {Ignore - 1} colours.");
        int targets = colors.Count + 1;
        int[] palette = new int[targets * 3];
        for (int i = 0; i < colors.Count; i++)
            (palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2]) = (colors[i].R, colors[i].G, colors[i].B);
        int pixels = width * height;
        byte[] labels = new byte[pixels];
        float[] distance = new float[pixels];
        for (int p = 0; p < pixels; p++)
        {
            int r = rgb[p * 3], g = rgb[p * 3 + 1], b = rgb[p * 3 + 2];
            int best = 0, bestDistance = int.MaxValue;
            for (int t = 0; t < targets; t++)
            {
                int dr = r - palette[t * 3], dg = g - palette[t * 3 + 1], db = b - palette[t * 3 + 2];
                int d = dr * dr + dg * dg + db * db;
                if (d < bestDistance)
                {
                    bestDistance = d;
                    best = t;
                }
            }
            labels[p] = best == colors.Count ? Ignore : (byte)best;
            distance[p] = bestDistance;
        }
        return (labels, distance);
    }

    /// <summary>Per-pixel majority vote over label maps decoded from different colour legends (legend ensembling): the maps
    /// share label space, so the vote is valid there. <see cref="Ignore"/> abstains; a pixel every map abstained on stays
    /// <see cref="Ignore"/>; ties go to the lowest class. Also returns the winner's share among the non-abstaining votes
    /// (0 where all abstained), a stability diagnostic and not a probability.</summary>
    public static (byte[] Labels, float[] Agreement) MajorityVote(IReadOnlyList<byte[]> maps, int classCount)
    {
        ArgumentNullException.ThrowIfNull(maps);
        if (maps.Count == 0)
            throw new ArgumentException("A vote needs at least one label map.", nameof(maps));
        if (classCount is < 1 or >= Ignore)
            throw new ArgumentOutOfRangeException(nameof(classCount), classCount, $"A vote covers 1 to {Ignore - 1} classes.");
        int pixels = maps[0].Length;
        foreach (byte[] map in maps)
        {
            if (map.Length != pixels)
                throw new ArgumentException($"A vote over label maps of different sizes: {pixels} and {map.Length} pixels.", nameof(maps));
        }
        byte[] labels = new byte[pixels];
        float[] agreement = new float[pixels];
        int[] counts = new int[classCount];
        for (int p = 0; p < pixels; p++)
        {
            Array.Clear(counts);
            int voted = 0;
            foreach (byte[] map in maps)
            {
                byte label = map[p];
                if (label < classCount)
                {
                    counts[label]++;
                    voted++;
                }
            }
            int winner = 0, top = 0;
            for (int c = 0; c < classCount; c++)
            {
                if (counts[c] > top)
                {
                    top = counts[c];
                    winner = c;
                }
            }
            labels[p] = top > 0 ? (byte)winner : Ignore;
            agreement[p] = voted > 0 ? top / (float)voted : 0f;
        }
        return (labels, agreement);
    }

    /// <summary>Merges single-group decodes into one label map: per pixel the pass that decoded a real class with the smallest
    /// snap distance wins (the first pass on a tie); a pixel void in every pass stays <see cref="Ignore"/>. This is the grouped
    /// protocol for taxonomies larger than one prompt can legend.</summary>
    public static byte[] MergeGrouped(IReadOnlyList<byte[]> labelMaps, IReadOnlyList<float[]> distanceMaps)
    {
        ArgumentNullException.ThrowIfNull(labelMaps);
        ArgumentNullException.ThrowIfNull(distanceMaps);
        if (labelMaps.Count == 0 || labelMaps.Count != distanceMaps.Count)
            throw new ArgumentException("Grouped merging needs one distance map per label map, and at least one.", nameof(distanceMaps));
        int pixels = labelMaps[0].Length;
        byte[] merged = new byte[pixels];
        for (int p = 0; p < pixels; p++)
        {
            byte label = Ignore;
            float best = float.PositiveInfinity;
            for (int pass = 0; pass < labelMaps.Count; pass++)
            {
                if (labelMaps[pass].Length != pixels || distanceMaps[pass].Length != pixels)
                    throw new ArgumentException("Grouped merging over maps of different sizes.", nameof(labelMaps));
                if (labelMaps[pass][p] != Ignore && distanceMaps[pass][p] < best)
                {
                    best = distanceMaps[pass][p];
                    label = labelMaps[pass][p];
                }
            }
            merged[p] = label;
        }
        return merged;
    }

    /// <summary>The relative inverse depth in <c>[0, 1]</c>, closer is larger: the painted image's luma over 255, as the
    /// release's depth evaluator reads it (<c>convert("L") / 255</c>; PIL's integer BT.601 weights).</summary>
    public static float[] DepthFromRgb(ReadOnlySpan<byte> rgb, int width, int height)
    {
        CheckRgb(rgb, width, height);
        float[] depth = new float[width * height];
        for (int p = 0; p < depth.Length; p++)
            depth[p] = ((rgb[p * 3] * 19595 + rgb[p * 3 + 1] * 38470 + rgb[p * 3 + 2] * 7471 + 0x8000) >> 16) / 255f;
        return depth;
    }

    /// <summary>Camera-space unit normals <c>[height, width, 3]</c> (x right, y up, z toward the camera) from the painted
    /// image: <c>R=(1-x)/2, G=(1+y)/2, B=(1+z)/2</c>, so x is the negated red component. A pixel whose decoded vector is
    /// shorter than <paramref name="minNorm"/> (the release's 0.05) is invalid and left zero.</summary>
    public static (float[] Normals, bool[] Valid) NormalsFromRgb(ReadOnlySpan<byte> rgb, int width, int height, float minNorm = 0.05f)
    {
        CheckRgb(rgb, width, height);
        float[] normals = new float[width * height * 3];
        bool[] valid = new bool[width * height];
        for (int p = 0; p < valid.Length; p++)
        {
            float x = -(rgb[p * 3] / 255f * 2f - 1f);
            float y = rgb[p * 3 + 1] / 255f * 2f - 1f;
            float z = rgb[p * 3 + 2] / 255f * 2f - 1f;
            float norm = MathF.Sqrt(x * x + y * y + z * z);
            if (!(norm >= minNorm) || !float.IsFinite(norm))
                continue;
            valid[p] = true;
            normals[p * 3] = x / norm;
            normals[p * 3 + 1] = y / norm;
            normals[p * 3 + 2] = z / norm;
        }
        return (normals, valid);
    }

    private static void CheckRgb(ReadOnlySpan<byte> rgb, int width, int height)
    {
        if (width < 1 || height < 1)
            throw new ArgumentOutOfRangeException(nameof(width), $"An image is at least 1x1; got {width}x{height}.");
        if (rgb.Length != (long)width * height * 3)
            throw new ArgumentException($"The {rgb.Length} bytes are not {width}x{height} RGB24.", nameof(rgb));
    }
}
