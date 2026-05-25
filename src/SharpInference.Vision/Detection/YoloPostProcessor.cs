using SharpInference.Core.Tensors;

namespace SharpInference.Vision.Detection;

/// <summary>Decodes the YOLO model's output tensor (<c>[B, 4 + numClasses, totalAnchors]</c>)
/// into a list of <see cref="YoloDetection"/> by:
/// <list type="number">
///   <item>Walking each anchor, finding its max-scored class.</item>
///   <item>Dropping anchors below <c>confidenceThreshold</c>.</item>
///   <item>Converting box <c>xywh</c> → <c>xyxy</c>.</item>
///   <item>Running <see cref="NonMaxSuppression.Run"/> on the survivors.</item>
///   <item>Un-projecting boxes through the preprocessor's <see cref="YoloPreprocessor.Transform"/>
///         so coordinates land in the original source-image frame.</item>
/// </list></summary>
public static class YoloPostProcessor
{
    /// <summary>Decodes a single-batch YOLO output. The output tensor must have shape
    /// <c>[1, 4 + numClasses, totalAnchors]</c> — exactly what <see cref="YoloModel.Forward"/>
    /// produces. Box values in the first 4 channels are <c>(cx, cy, w, h)</c> in letterbox-canvas
    /// pixels; class channels are already sigmoid-activated.</summary>
    public static IReadOnlyList<YoloDetection> Decode(
        Tensor modelOutput,
        YoloPreprocessor.Transform transform,
        int numClasses,
        float confidenceThreshold = 0.25f,
        float iouThreshold = 0.45f,
        int maxDetections = 300,
        bool classAgnostic = false)
    {
        ArgumentNullException.ThrowIfNull(modelOutput);
        if (modelOutput.Shape.Rank != 3 || modelOutput.Shape[0] != 1)
            throw new ArgumentException($"Decoder expects [1, 4+nc, anchors]; got {modelOutput.Shape}.", nameof(modelOutput));
        int channels = (int)modelOutput.Shape[1];
        int totalAnchors = (int)modelOutput.Shape[2];
        if (channels != 4 + numClasses)
            throw new ArgumentException($"Channel count {channels} != 4 + numClasses ({4 + numClasses}).");

        ReadOnlySpan<float> data = modelOutput.AsReadOnlySpan<float>();
        // Channel-major layout: data[c * totalAnchors + a] is channel c at anchor a.
        List<YoloDetection> candidates = new(capacity: 256);

        for (int a = 0; a < totalAnchors; a++)
        {
            float bestScore = 0f;
            int bestClass = -1;
            for (int k = 0; k < numClasses; k++)
            {
                float p = data[(4 + k) * totalAnchors + a];
                if (p > bestScore)
                {
                    bestScore = p;
                    bestClass = k;
                }
            }
            if (bestScore < confidenceThreshold || bestClass < 0)
                continue;

            float cx = data[0 * totalAnchors + a];
            float cy = data[1 * totalAnchors + a];
            float bw = data[2 * totalAnchors + a];
            float bh = data[3 * totalAnchors + a];
            float lx1 = cx - bw * 0.5f;
            float ly1 = cy - bh * 0.5f;
            float lx2 = cx + bw * 0.5f;
            float ly2 = cy + bh * 0.5f;

            // Un-project from letterbox-canvas coords to source-image coords.
            (float x1, float y1, float x2, float y2) = transform.Invert(lx1, ly1, lx2, ly2);
            candidates.Add(new YoloDetection(x1, y1, x2, y2, bestScore, bestClass));
        }

        return NonMaxSuppression.Run(candidates, iouThreshold, maxDetections, classAgnostic);
    }
}
