using HartsyInference.LLM.Multimodal;

namespace HartsyInference.LLM.Kimi.Vision;

/// <summary>Resamples MoonViT-V2's learned <c>[side, side, hidden]</c> table onto a patch grid, bilinearly with <c>align_corners=False</c> (PyTorch's
/// <c>F.interpolate</c> default), one row per token in the merge-block order of the patch rows.</summary>
internal static class KimiVisionPosEmbed
{
    /// <summary>Writes <c>[gh·gw, hidden]</c> into <paramref name="dst"/>, sampling the table at each token's grid position.</summary>
    public static void Resample(Span<float> dst, ReadOnlySpan<float> table, int side, int hidden, int gh, int gw, int merge)
    {
        if (dst.Length != gh * gw * hidden || table.Length != side * side * hidden)
            throw new ArgumentException("The position table and destination do not match the grid.", nameof(dst));
        int[] rows = new int[gh * gw], cols = new int[gh * gw];
        VisionPatchOps.BlockOrderPositions(gh, gw, merge, rows, cols);
        float scaleY = (float)side / gh, scaleX = (float)side / gw;
        for (int k = 0; k < rows.Length; k++)
        {
            float sy = MathF.Max(0f, (rows[k] + 0.5f) * scaleY - 0.5f), sx = MathF.Max(0f, (cols[k] + 0.5f) * scaleX - 0.5f);
            int y0 = Math.Min((int)sy, side - 1), x0 = Math.Min((int)sx, side - 1);
            int y1 = Math.Min(y0 + 1, side - 1), x1 = Math.Min(x0 + 1, side - 1);
            float wy = sy - y0, wx = sx - x0;
            ReadOnlySpan<float> r00 = table.Slice((y0 * side + x0) * hidden, hidden), r01 = table.Slice((y0 * side + x1) * hidden, hidden);
            ReadOnlySpan<float> r10 = table.Slice((y1 * side + x0) * hidden, hidden), r11 = table.Slice((y1 * side + x1) * hidden, hidden);
            Span<float> d = dst.Slice(k * hidden, hidden);
            for (int c = 0; c < hidden; c++)
                d[c] = (1f - wy) * ((1f - wx) * r00[c] + wx * r01[c]) + wy * ((1f - wx) * r10[c] + wx * r11[c]);
        }
    }
}
