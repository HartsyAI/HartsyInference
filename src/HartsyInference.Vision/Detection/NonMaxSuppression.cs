namespace HartsyInference.Vision.Detection;

/// <summary>Greedy Non-Maximum Suppression for YOLO-style detections. Matches Ultralytics
/// <c>non_max_suppression</c> semantics — class-specific by default (boxes of different classes
/// never suppress each other), with optional class-agnostic mode for use cases like generic
/// foreground extraction.
/// <para><b>Algorithm.</b> Sort all candidates by confidence descending. Repeatedly take the
/// top-scoring box, append it to the output, and drop every remaining box that has IoU above
/// <c>iouThreshold</c> with the picked box <i>and</i> is the same class (unless agnostic).
/// Stops at <c>maxDetections</c>.</para>
/// <para><b>Why not torchvision's class-id-offset trick?</b> Ultralytics implements class-specific
/// NMS by adding <c>classId * maxBoxDimension</c> to box coordinates before a single
/// class-agnostic NMS pass, so the IoU between different-class boxes is always zero. That's a
/// nice optimization for GPU NMS but offers no benefit in pure C# — our explicit branch is
/// clearer and easier to debug.</para></summary>
public static class NonMaxSuppression
{
    /// <summary>Runs greedy NMS over the candidate detections.</summary>
    /// <param name="candidates">Detections after confidence filtering. The list itself is not mutated.</param>
    /// <param name="iouThreshold">IoU above which a lower-scored same-class box is suppressed. Ultralytics default 0.45.</param>
    /// <param name="maxDetections">Cap on output count. Ultralytics default 300. Pass <c>int.MaxValue</c> to disable.</param>
    /// <param name="classAgnostic">If true, boxes of any class can suppress each other. Default false (Ultralytics default).</param>
    public static IReadOnlyList<YoloDetection> Run(
        IReadOnlyList<YoloDetection> candidates,
        float iouThreshold = 0.45f,
        int maxDetections = 300,
        bool classAgnostic = false)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        if (iouThreshold < 0f || iouThreshold > 1f)
            throw new ArgumentOutOfRangeException(nameof(iouThreshold), iouThreshold, "IoU threshold must be in [0, 1].");
        if (maxDetections <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxDetections), maxDetections, "maxDetections must be positive.");

        if (candidates.Count == 0)
            return Array.Empty<YoloDetection>();

        IReadOnlyList<int> keptIdx = RunIndices(candidates, iouThreshold, maxDetections, classAgnostic);
        YoloDetection[] kept = new YoloDetection[keptIdx.Count];
        for (int i = 0; i < keptIdx.Count; i++)
            kept[i] = candidates[keptIdx[i]];
        return kept;
    }

    /// <summary>Same greedy NMS as <see cref="Run"/> but returns the surviving <b>indices into
    /// <paramref name="candidates"/></b> (confidence-descending order) instead of the detections. This lets a
    /// caller carry a per-candidate payload (e.g. pose keypoints) through NMS without re-implementing it.</summary>
    public static IReadOnlyList<int> RunIndices(
        IReadOnlyList<YoloDetection> candidates,
        float iouThreshold = 0.45f,
        int maxDetections = 300,
        bool classAgnostic = false)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        if (iouThreshold < 0f || iouThreshold > 1f)
            throw new ArgumentOutOfRangeException(nameof(iouThreshold), iouThreshold, "IoU threshold must be in [0, 1].");
        if (maxDetections <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxDetections), maxDetections, "maxDetections must be positive.");

        int n = candidates.Count;
        if (n == 0)
            return Array.Empty<int>();

        // Sort indices by confidence descending (the candidate list itself is not mutated).
        int[] order = new int[n];
        for (int i = 0; i < n; i++)
            order[i] = i;
        Array.Sort(order, (a, b) => candidates[b].Confidence.CompareTo(candidates[a].Confidence));

        bool[] suppressed = new bool[n];
        List<int> kept = new(Math.Min(maxDetections, n));
        for (int ii = 0; ii < n; ii++)
        {
            int i = order[ii];
            if (suppressed[i])
                continue;
            kept.Add(i);
            if (kept.Count >= maxDetections)
                break;
            YoloDetection pick = candidates[i];
            for (int jj = ii + 1; jj < n; jj++)
            {
                int j = order[jj];
                if (suppressed[j])
                    continue;
                if (!classAgnostic && candidates[j].ClassId != pick.ClassId)
                    continue;
                if (pick.Iou(candidates[j]) > iouThreshold)
                    suppressed[j] = true;
            }
        }
        return kept;
    }

    /// <summary>Convenience wrapper that filters by confidence threshold first, then runs NMS.
    /// Mirrors <c>ultralytics.utils.ops.non_max_suppression</c> defaults end-to-end.</summary>
    public static IReadOnlyList<YoloDetection> RunWithConfidenceFilter(
        IReadOnlyList<YoloDetection> candidates,
        float confidenceThreshold = 0.25f,
        float iouThreshold = 0.45f,
        int maxDetections = 300,
        bool classAgnostic = false)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        if (confidenceThreshold < 0f || confidenceThreshold > 1f)
            throw new ArgumentOutOfRangeException(nameof(confidenceThreshold), confidenceThreshold, "Confidence threshold must be in [0, 1].");

        if (candidates.Count == 0)
            return Array.Empty<YoloDetection>();

        // Single-pass filter to avoid an intermediate List allocation when most candidates pass
        // (typical for high-recall confidence thresholds like 0.05). When few pass (e.g. 0.5+),
        // we waste a little capacity but skip the second pass — a fair trade.
        List<YoloDetection> filtered = new(candidates.Count);
        for (int i = 0; i < candidates.Count; i++)
        {
            YoloDetection d = candidates[i];
            if (d.Confidence >= confidenceThreshold)
                filtered.Add(d);
        }

        return Run(filtered, iouThreshold, maxDetections, classAgnostic);
    }
}
