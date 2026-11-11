namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// The released config's <c>ObjectCentricSSI</c> pointmap normaliser (<c>use_scene_scale</c> true). The shift is the per-axis median of the
/// masked points; the scale is the median over every pixel of the largest centred axis magnitude (the scene, not the object), the same on
/// all three axes. Pointmaps are <c>3 × H × W</c> row-major (channel first) and may hold NaN for invalid points.
/// </summary>
public static class SamPointmapNormalizer
{
    /// <summary>Per-axis shift and the common scale of <paramref name="pointmap"/> under <paramref name="mask"/> (selected where above 0.5, same H × W).</summary>
    public static (float[] Scale, float[] Shift) ComputeScaleAndShift(
        ReadOnlySpan<float> pointmap, ReadOnlySpan<float> mask, int height, int width, float scaleFactor = 1f)
    {
        int pixels = RequireLayout(pointmap, height, width);
        if (mask.Length != pixels)
            throw new ArgumentException($"The mask has {mask.Length} values for {pixels} pixels.", nameof(mask));
        float[] shift = new float[3];
        List<float> values = new(pixels);
        int[] counts = new int[3];
        for (int axis = 0; axis < 3; axis++)
        {
            values.Clear();
            for (int i = 0; i < pixels; i++)
            {
                if (mask[i] > 0.5f && float.IsFinite(pointmap[axis * pixels + i]))
                    values.Add(pointmap[axis * pixels + i]);
            }

            counts[axis] = values.Count;
            shift[axis] = values.Count > 0 ? LowerMedian(values) : 0f;
        }

        if (counts[0] + counts[1] + counts[2] == 0)
            return ([scaleFactor, scaleFactor, scaleFactor], [0f, 0f, 0f]);
        if (counts[0] == 0 || counts[1] == 0 || counts[2] == 0)
            throw new InvalidOperationException("An axis has no finite masked point, so its median is undefined.");

        values.Clear();
        for (int i = 0; i < pixels; i++)
        {
            float largest = 0f;
            bool valid = true;
            for (int axis = 0; axis < 3; axis++)
            {
                float centred = MathF.Abs(pointmap[axis * pixels + i] - shift[axis]);
                valid &= !float.IsNaN(centred);
                largest = Math.Max(largest, centred);
            }

            if (valid)
                values.Add(largest);
        }

        if (values.Count == 0)
            throw new InvalidOperationException("The pointmap holds no valid pixel to take a scene scale from.");
        float scale = LowerMedian(values) * scaleFactor;
        return ([scale, scale, scale], shift);
    }

    /// <summary>In place: <c>(p − shift) / scale</c> per axis.</summary>
    public static void Normalize(Span<float> pointmap, int height, int width, ReadOnlySpan<float> scale, ReadOnlySpan<float> shift)
    {
        int pixels = RequireLayout(pointmap, height, width);
        RequireVectors(scale, shift);
        for (int axis = 0; axis < 3; axis++)
        {
            Span<float> channel = pointmap.Slice(axis * pixels, pixels);
            for (int i = 0; i < channel.Length; i++)
                channel[i] = (channel[i] - shift[axis]) / scale[axis];
        }
    }

    /// <summary>In place: <c>p · scale + shift</c> per axis, the inverse of <see cref="Normalize"/>.</summary>
    public static void Denormalize(Span<float> pointmap, int height, int width, ReadOnlySpan<float> scale, ReadOnlySpan<float> shift)
    {
        int pixels = RequireLayout(pointmap, height, width);
        RequireVectors(scale, shift);
        for (int axis = 0; axis < 3; axis++)
        {
            Span<float> channel = pointmap.Slice(axis * pixels, pixels);
            for (int i = 0; i < channel.Length; i++)
                channel[i] = channel[i] * scale[axis] + shift[axis];
        }
    }

    /// <summary>Nearest-neighbour resize of one channel (<c>floor(dst · src / dst size)</c>, torchvision's rule).</summary>
    public static float[] ResizeNearest(ReadOnlySpan<float> source, int sourceHeight, int sourceWidth, int height, int width)
    {
        if (source.Length != sourceHeight * sourceWidth || height <= 0 || width <= 0)
            throw new ArgumentException("The source size does not match its dimensions, or the target is empty.");
        float[] result = new float[height * width];
        for (int y = 0; y < height; y++)
        {
            int sy = Math.Min(sourceHeight - 1, (int)((long)y * sourceHeight / height));
            for (int x = 0; x < width; x++)
                result[y * width + x] = source[sy * sourceWidth + Math.Min(sourceWidth - 1, (int)((long)x * sourceWidth / width))];
        }

        return result;
    }

    /// <summary>The lower of the two middle values, which is what <c>torch.nanmedian</c> returns for an even count.</summary>
    private static float LowerMedian(List<float> values)
    {
        values.Sort();
        return values[(values.Count - 1) / 2];
    }

    private static int RequireLayout(ReadOnlySpan<float> pointmap, int height, int width)
    {
        int pixels = height * width;
        if (height <= 0 || width <= 0 || pointmap.Length != 3 * pixels)
            throw new ArgumentException($"A pointmap of {height}x{width} holds {3 * pixels} values; got {pointmap.Length}.", nameof(pointmap));
        return pixels;
    }

    private static void RequireVectors(ReadOnlySpan<float> scale, ReadOnlySpan<float> shift)
    {
        if (scale.Length != 3 || shift.Length != 3)
            throw new ArgumentException("Scale and shift are 3-vectors.");
        for (int axis = 0; axis < 3; axis++)
        {
            if (!(scale[axis] != 0f) || !float.IsFinite(scale[axis]) || !float.IsFinite(shift[axis]))
                throw new ArgumentException("Scale must be finite and non-zero and the shift finite.");
        }
    }
}
