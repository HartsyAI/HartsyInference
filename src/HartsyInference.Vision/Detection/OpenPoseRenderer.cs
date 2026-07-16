namespace HartsyInference.Vision.Detection;

/// <summary>Renders detected <see cref="PoseDetection"/>s as an OpenPose-18 body skeleton (colored limb "sticks" +
/// joint dots on black), matching the controlnet_aux <c>draw_bodypose</c> convention that DWPose→ControlNet and
/// Wan-Animate's <c>pose_patch_embedding</c> were trained on. COCO-17 keypoints are remapped to OpenPose-18 (the
/// neck is synthesized as the shoulder midpoint). Pure raster (no backend) → unit-testable.</summary>
public static class OpenPoseRenderer
{
    private const int StickWidth = 4;
    private const int JointRadius = 4;
    private const float LimbAlpha = 0.6f;   // controlnet_aux blends each limb at 0.6 onto the canvas

    /// <summary>OpenPose-18 index → COCO-17 index. Index 1 (neck) is a special case (shoulder midpoint), marked -1.</summary>
    private static readonly int[] OpToCoco = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3];

    /// <summary>The 17 limbs as OpenPose-18 endpoint index pairs (controlnet_aux limbSeq[:17], 0-indexed).</summary>
    private static readonly int[][] Limbs =
    [
        [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
        [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
    ];

    /// <summary>The classic 18-entry OpenPose color palette (RGB), indexed by limb (0..16) and joint (0..17).</summary>
    private static readonly byte[][] Colors =
    [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
        [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
        [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85],
    ];

    /// <summary>Renders all people to a black <paramref name="width"/>×<paramref name="height"/> HWC rgb24 canvas
    /// (keypoints in source-image pixels). Joints/limbs below <paramref name="visThreshold"/> are skipped.</summary>
    public static byte[] RenderBodyPose(IReadOnlyList<PoseDetection> people, int width, int height, float visThreshold = 0.3f)
        => RenderBodyPose(people, width, height, 1f, 1f, visThreshold);

    /// <summary>Scaled variant: keypoint coordinates (source-image pixels) are multiplied by
    /// <paramref name="scaleX"/>/<paramref name="scaleY"/> before drawing — used when the skeleton canvas is a
    /// different resolution than the image the detector ran on (e.g. ControlNet conditioning at gen resolution).</summary>
    public static byte[] RenderBodyPose(IReadOnlyList<PoseDetection> people, int width, int height,
        float scaleX, float scaleY, float visThreshold = 0.3f)
    {
        ArgumentNullException.ThrowIfNull(people);
        if (width <= 0 || height <= 0)
            throw new ArgumentException($"Canvas dimensions must be positive; got {width}x{height}.");
        byte[] canvas = new byte[(long)width * height * 3];   // zero = black

        // Reused per-person OpenPose-18 keypoint buffers (allocated once, not per person — CA2014).
        Span<float> px = stackalloc float[18];
        Span<float> py = stackalloc float[18];
        Span<bool> vis = stackalloc bool[18];
        foreach (PoseDetection person in people)
        {
            // Build the 18 OpenPose keypoints (x, y, visible) from the 17 COCO keypoints.
            for (int op = 0; op < 18; op++)
            {
                if (op == 1)   // neck = shoulder midpoint (COCO 5 = L-shoulder, 6 = R-shoulder)
                {
                    Keypoint ls = person.Keypoints[5], rs = person.Keypoints[6];
                    vis[1] = ls.Confidence >= visThreshold && rs.Confidence >= visThreshold;
                    px[1] = (ls.X + rs.X) * 0.5f * scaleX; py[1] = (ls.Y + rs.Y) * 0.5f * scaleY;
                    continue;
                }
                Keypoint kp = person.Keypoints[OpToCoco[op]];
                vis[op] = kp.Confidence >= visThreshold;
                px[op] = kp.X * scaleX; py[op] = kp.Y * scaleY;
            }

            for (int i = 0; i < Limbs.Length; i++)
            {
                int a = Limbs[i][0], b = Limbs[i][1];
                if (!vis[a] || !vis[b]) continue;
                DrawLimb(canvas, width, height, px[a], py[a], px[b], py[b], Colors[i]);
            }
            for (int op = 0; op < 18; op++)
            {
                if (!vis[op]) continue;
                DrawJoint(canvas, width, height, px[op], py[op], Colors[op]);
            }
        }
        return canvas;
    }

    /// <summary>Fills the rotated-ellipse "stick" between two joints (semi-major = half the limb length along its
    /// direction, semi-minor = <see cref="StickWidth"/>), alpha-blended — the controlnet_aux ellipse2Poly stick.</summary>
    private static void DrawLimb(byte[] canvas, int w, int h, float x0, float y0, float x1, float y1, byte[] color)
    {
        float dx = x0 - x1, dy = y0 - y1;
        float len = MathF.Sqrt(dx * dx + dy * dy);
        if (len < 1e-3f) return;
        float ux = dx / len, uy = dy / len;   // unit along the limb
        float mx = (x0 + x1) * 0.5f, my = (y0 + y1) * 0.5f;
        float semiMajor = len * 0.5f, semiMinor = StickWidth;

        int minX = Math.Max(0, (int)MathF.Floor(Math.Min(x0, x1) - StickWidth));
        int maxX = Math.Min(w - 1, (int)MathF.Ceiling(Math.Max(x0, x1) + StickWidth));
        int minY = Math.Max(0, (int)MathF.Floor(Math.Min(y0, y1) - StickWidth));
        int maxY = Math.Min(h - 1, (int)MathF.Ceiling(Math.Max(y0, y1) + StickWidth));
        for (int y = minY; y <= maxY; y++)
        {
            for (int x = minX; x <= maxX; x++)
            {
                float rx = x - mx, ry = y - my;
                float along = rx * ux + ry * uy;         // projection onto the limb axis
                float perp = -rx * uy + ry * ux;         // perpendicular distance
                float e = along * along / (semiMajor * semiMajor) + perp * perp / (semiMinor * semiMinor);
                if (e <= 1f) Blend(canvas, w, x, y, color, LimbAlpha);
            }
        }
    }

    private static void DrawJoint(byte[] canvas, int w, int h, float cx, float cy, byte[] color)
    {
        int minX = Math.Max(0, (int)(cx - JointRadius)), maxX = Math.Min(w - 1, (int)(cx + JointRadius));
        int minY = Math.Max(0, (int)(cy - JointRadius)), maxY = Math.Min(h - 1, (int)(cy + JointRadius));
        float r2 = JointRadius * JointRadius;
        for (int y = minY; y <= maxY; y++)
            for (int x = minX; x <= maxX; x++)
            {
                float dxp = x - cx, dyp = y - cy;
                if (dxp * dxp + dyp * dyp <= r2) Blend(canvas, w, x, y, color, 1f);   // joints drawn solid
            }
    }

    private static void Blend(byte[] canvas, int w, int x, int y, byte[] color, float alpha)
    {
        long p = ((long)y * w + x) * 3;
        float inv = 1f - alpha;
        canvas[p] = (byte)(canvas[p] * inv + color[0] * alpha);
        canvas[p + 1] = (byte)(canvas[p + 1] * inv + color[1] * alpha);
        canvas[p + 2] = (byte)(canvas[p + 2] * inv + color[2] * alpha);
    }
}
