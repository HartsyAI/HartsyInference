using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>Decodes the <see cref="YoloV11PoseModel"/> outputs — box/class detections <c>[1, 4+nc, A]</c> and
/// decoded keypoints <c>[1, nk·ndim, A]</c> (both in letterbox-canvas pixels) — into <see cref="PoseDetection"/>s
/// in source-image coordinates. Mirrors <see cref="YoloPostProcessor"/> for the box path (confidence filter →
/// xywh→xyxy → letterbox-invert → NMS), then gathers each survivor's keypoints via
/// <see cref="NonMaxSuppression.RunIndices"/> and un-projects each point with <see cref="YoloPreprocessor.Transform.InvertPoint"/>.</summary>
public static class YoloPosePostProcessor
{
    public static IReadOnlyList<PoseDetection> Decode(
        Tensor detections,
        Tensor keypoints,
        YoloPreprocessor.Transform transform,
        int numClasses,
        int numKeypoints,
        int kptDims,
        float confidenceThreshold = 0.25f,
        float iouThreshold = 0.45f,
        int maxDetections = 300,
        bool classAgnostic = false)
    {
        ArgumentNullException.ThrowIfNull(detections);
        ArgumentNullException.ThrowIfNull(keypoints);
        if (detections.Shape.Rank != 3 || detections.Shape[0] != 1)
            throw new ArgumentException($"Pose decoder expects detections [1, 4+nc, A]; got {detections.Shape}.", nameof(detections));
        int totalAnchors = (int)detections.Shape[2];
        if ((int)detections.Shape[1] != 4 + numClasses)
            throw new ArgumentException($"Detection channels {detections.Shape[1]} != 4 + numClasses ({4 + numClasses}).");
        if (keypoints.Shape.Rank != 3 || keypoints.Shape[0] != 1 || (int)keypoints.Shape[2] != totalAnchors
            || (int)keypoints.Shape[1] != numKeypoints * kptDims)
            throw new ArgumentException($"Keypoint tensor must be [1, {numKeypoints * kptDims}, {totalAnchors}]; got {keypoints.Shape}.", nameof(keypoints));

        ReadOnlySpan<float> det = detections.AsReadOnlySpan<float>();
        ReadOnlySpan<float> kpt = keypoints.AsReadOnlySpan<float>();

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

        IReadOnlyList<int> keep = NonMaxSuppression.RunIndices(candBoxes, iouThreshold, maxDetections, classAgnostic);
        List<PoseDetection> results = new(keep.Count);
        foreach (int ki in keep)
        {
            int a = candAnchors[ki];
            Keypoint[] kps = new Keypoint[numKeypoints];
            for (int k = 0; k < numKeypoints; k++)
            {
                float kx = kpt[(k * kptDims + 0) * totalAnchors + a];
                float ky = kpt[(k * kptDims + 1) * totalAnchors + a];
                float kv = kptDims == 3 ? kpt[(k * kptDims + 2) * totalAnchors + a] : 1f;
                (float sx, float sy) = transform.InvertPoint(kx, ky);
                kps[k] = new Keypoint(sx, sy, kv);
            }
            results.Add(new PoseDetection(candBoxes[ki], kps));
        }
        return results;
    }
}
