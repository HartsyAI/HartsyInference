using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Detection;

namespace HartsyInference.Vision.FaceDetection;

/// <summary>Decodes the <see cref="YoloV8FaceModel"/> outputs — box/class detections <c>[1, 4+1, A]</c> and decoded
/// landmarks <c>[1, 5·ndim, A]</c> (both in letterbox-canvas pixels) — into <see cref="DetectedFace"/>s in
/// source-image coordinates. Mirrors <see cref="YoloPostProcessor"/> for the box path (confidence filter → xywh→xyxy
/// → letterbox-invert → NMS), then gathers each survivor's five landmarks via <see cref="NonMaxSuppression.RunIndices"/>
/// + <see cref="LandmarkExtractor.ExtractForAnchor"/>.</summary>
public static class FaceDetectionPostProcessor
{
    public static IReadOnlyList<DetectedFace> Decode(
        Tensor detections,
        Tensor landmarks,
        YoloPreprocessor.Transform transform,
        int numClasses,
        int numPoints,
        int kptDims,
        float confidenceThreshold = 0.25f,
        float iouThreshold = 0.45f,
        int maxDetections = 300)
    {
        ArgumentNullException.ThrowIfNull(detections);
        ArgumentNullException.ThrowIfNull(landmarks);
        if (detections.Shape.Rank != 3 || detections.Shape[0] != 1)
            throw new ArgumentException($"Face decoder expects detections [1, 4+nc, A]; got {detections.Shape}.", nameof(detections));
        int totalAnchors = (int)detections.Shape[2];
        if ((int)detections.Shape[1] != 4 + numClasses)
            throw new ArgumentException($"Detection channels {detections.Shape[1]} != 4 + numClasses ({4 + numClasses}).");
        if (landmarks.Shape.Rank != 3 || landmarks.Shape[0] != 1 || (int)landmarks.Shape[2] != totalAnchors
            || (int)landmarks.Shape[1] != numPoints * kptDims)
            throw new ArgumentException($"Landmark tensor must be [1, {numPoints * kptDims}, {totalAnchors}]; got {landmarks.Shape}.", nameof(landmarks));

        ReadOnlySpan<float> det = detections.AsReadOnlySpan<float>();

        List<YoloDetection> candBoxes = new(capacity: 256);
        List<int> candAnchors = new(capacity: 256);
        for (int a = 0; a < totalAnchors; a++)
        {
            float bestScore = 0f;
            int bestClass = -1;
            for (int k = 0; k < numClasses; k++)
            {
                float p = det[(4 + k) * totalAnchors + a];
                if (p > bestScore) { bestScore = p; bestClass = k; }
            }
            if (bestScore < confidenceThreshold || bestClass < 0)
                continue;

            float cx = det[0 * totalAnchors + a];
            float cy = det[1 * totalAnchors + a];
            float bw = det[2 * totalAnchors + a];
            float bh = det[3 * totalAnchors + a];
            (float x1, float y1, float x2, float y2) = transform.Invert(
                cx - bw * 0.5f, cy - bh * 0.5f, cx + bw * 0.5f, cy + bh * 0.5f);
            candBoxes.Add(new YoloDetection(x1, y1, x2, y2, bestScore, bestClass));
            candAnchors.Add(a);
        }

        IReadOnlyList<int> keep = NonMaxSuppression.RunIndices(candBoxes, iouThreshold, maxDetections, classAgnostic: false);
        List<DetectedFace> results = new(keep.Count);
        foreach (int ki in keep)
        {
            int a = candAnchors[ki];
            Landmark5 lm5 = LandmarkExtractor.ExtractForAnchor(landmarks, transform, a, numPoints, kptDims);
            results.Add(new DetectedFace(candBoxes[ki], lm5));
        }
        return results;
    }
}
