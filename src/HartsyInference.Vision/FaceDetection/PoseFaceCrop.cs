using HartsyInference.Vision.Detection;

namespace HartsyInference.Vision.FaceDetection;

/// <summary>Derives a square face-crop rectangle from YOLO-pose keypoints — the 5 face points (0 nose, 1/2 eyes,
/// 3/4 ears) bound the mid-face, expanded to a square that includes the forehead and chin. Used to build
/// Wan-Animate's face-motion driving clip without a dedicated face detector (the pose model already localizes
/// the face). Pure geometry (no backend) so it unit-tests trivially.</summary>
public static class PoseFaceCrop
{
    /// <summary>Face keypoint indices in COCO-17 order (nose, eyes, ears).</summary>
    private static readonly int[] FaceKpts = [0, 1, 2, 3, 4];

    /// <summary>A square crop in source-image pixels. May extend past the image edges (caller pads) so the face
    /// stays centered rather than shifting when it's near a border.</summary>
    public readonly record struct Rect(float X, float Y, float Size);

    /// <summary>Computes the square face crop for one person. Uses face keypoints above <paramref name="visThreshold"/>;
    /// falls back to the upper portion of the person box when too few are visible. <paramref name="expand"/> scales
    /// the keypoint span to include forehead+chin (~2.2 works for head-and-shoulders driving clips).</summary>
    public static Rect ComputeSquareCrop(PoseDetection person, int imageWidth, int imageHeight,
        float visThreshold = 0.3f, float expand = 2.2f)
    {
        ArgumentNullException.ThrowIfNull(person);

        float minX = float.MaxValue, minY = float.MaxValue, maxX = float.MinValue, maxY = float.MinValue;
        int valid = 0;
        foreach (int i in FaceKpts)
        {
            if (i >= person.Keypoints.Count) continue;
            Keypoint kp = person.Keypoints[i];
            if (kp.Confidence < visThreshold) continue;
            minX = MathF.Min(minX, kp.X); maxX = MathF.Max(maxX, kp.X);
            minY = MathF.Min(minY, kp.Y); maxY = MathF.Max(maxY, kp.Y);
            valid++;
        }

        float cx, cy, side;
        if (valid >= 2)
        {
            cx = (minX + maxX) * 0.5f;
            cy = (minY + maxY) * 0.5f;
            side = MathF.Max(maxX - minX, maxY - minY) * expand;
            // The nose/eye/ear cluster sits above the mouth+chin; nudge the crop down so the chin is included.
            cy += side * 0.12f;
        }
        else
        {
            // No reliable face points — take the top square of the person box (head is usually there).
            YoloDetection b = person.Box;
            side = MathF.Min(b.Width, b.Height);
            cx = (b.X1 + b.X2) * 0.5f;
            cy = b.Y1 + side * 0.5f;
        }

        side = MathF.Max(side, 8f);
        return new Rect(cx - side * 0.5f, cy - side * 0.5f, side);
    }
}
