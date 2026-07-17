using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Detection;

namespace HartsyInference.Vision.FaceDetection;

/// <summary>Decodes the YOLOv8-Face detector's landmark branch — the 5-point detector-side output — into
/// <see cref="Landmark5"/> in source-image coordinates. The dense 68/468-point path (FAN, MediaPipe FaceMesh) is a
/// separate cropped-face network and stays a documented non-goal here.
///
/// <para>The detector's landmark branch already applies the grid decode <c>(2·r + g)·stride</c> inside
/// <see cref="FaceDetectHead"/>, so its output tensor holds points in letterbox-canvas pixels. This extractor's job
/// is the per-detection gather + the letterbox inversion back to source pixels via
/// <see cref="YoloPreprocessor.Transform.InvertPoint"/>.</para></summary>
public static class LandmarkExtractor
{
    /// <summary>Full raw decode of one landmark point for testing / reference: applies the head's grid offset
    /// <c>(2·raw + g)·stride</c> then the letterbox inversion, mapping a raw branch offset at grid cell
    /// (<paramref name="gridX"/>, <paramref name="gridY"/>) to a source-image pixel. Keeping the whole mapping in one
    /// place lets a unit test pin the un-letterbox math end-to-end without a forward pass.</summary>
    public static (float x, float y) DecodeRawPoint(float rawX, float rawY, int gridX, int gridY, float stride, YoloPreprocessor.Transform transform)
    {
        float canvasX = (rawX * 2f + gridX) * stride;
        float canvasY = (rawY * 2f + gridY) * stride;
        return transform.InvertPoint(canvasX, canvasY);
    }

    /// <summary>Gathers the five landmarks for one anchor from a decoded landmark tensor <c>[1, 5·ndim, A]</c>
    /// (canvas pixels, as produced by <see cref="FaceDetectHead.Forward"/>) and inverts each point into source pixels.
    /// <paramref name="kptDims"/> is 3 when a visibility channel is present, 2 otherwise.</summary>
    public static Landmark5 ExtractForAnchor(Tensor landmarks, YoloPreprocessor.Transform transform, int anchorIndex, int numPoints, int kptDims)
    {
        ArgumentNullException.ThrowIfNull(landmarks);
        if (numPoints != Landmark5.Count)
            throw new ArgumentException($"Landmark5 requires exactly {Landmark5.Count} points; got {numPoints}.", nameof(numPoints));
        if (landmarks.Shape.Rank != 3 || landmarks.Shape[0] != 1 || (int)landmarks.Shape[1] != numPoints * kptDims)
            throw new ArgumentException($"Landmark tensor must be [1, {numPoints * kptDims}, A]; got {landmarks.Shape}.", nameof(landmarks));
        int totalAnchors = (int)landmarks.Shape[2];
        if (anchorIndex < 0 || anchorIndex >= totalAnchors)
            throw new ArgumentOutOfRangeException(nameof(anchorIndex), anchorIndex, $"Anchor index out of range [0, {totalAnchors}).");

        ReadOnlySpan<float> lm = landmarks.AsReadOnlySpan<float>();
        Span<FaceLandmark> pts = stackalloc FaceLandmark[Landmark5.Count];
        for (int k = 0; k < numPoints; k++)
        {
            float cx = lm[(k * kptDims + 0) * totalAnchors + anchorIndex];
            float cy = lm[(k * kptDims + 1) * totalAnchors + anchorIndex];
            float score = kptDims == 3 ? lm[(k * kptDims + 2) * totalAnchors + anchorIndex] : 1f;
            (float sx, float sy) = transform.InvertPoint(cx, cy);
            pts[k] = new FaceLandmark(sx, sy, score);
        }
        return new Landmark5(pts[0], pts[1], pts[2], pts[3], pts[4]);
    }
}
