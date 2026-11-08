namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>Score, selection, box and mask post-processing for SAM 3 predictions. Matches the upstream processor:
/// no NMS and no top-k, strict thresholds.</summary>
internal static class Sam3PostProcessor
{
    /// <summary>Writes <c>sigmoid(query) * sigmoid(presence)</c> for each query.</summary>
    public static void ComputeScores(ReadOnlySpan<float> classLogits, float presenceLogit, Span<float> scores)
    {
        if (scores.Length != classLogits.Length)
        {
            throw new ArgumentException("Score buffer must match the query count.", nameof(scores));
        }

        float presence = Sigmoid(presenceLogit);
        for (int i = 0; i < classLogits.Length; i++)
        {
            scores[i] = Sigmoid(classLogits[i]) * presence;
        }
    }

    /// <summary>Returns the indices whose score is strictly above <paramref name="threshold"/>, highest score first;
    /// equal scores keep index order.</summary>
    public static int[] SelectAbove(ReadOnlySpan<float> scores, float threshold)
    {
        int count = 0;
        for (int i = 0; i < scores.Length; i++)
        {
            if (scores[i] > threshold)
            {
                count++;
            }
        }

        int[] order = new int[count];
        int filled = 0;
        for (int i = 0; i < scores.Length; i++)
        {
            if (scores[i] > threshold)
            {
                order[filled++] = i;
            }
        }

        // Insertion sort keeps the order stable and the cost is bounded by the query count (200).
        for (int i = 1; i < order.Length; i++)
        {
            int current = order[i];
            int j = i - 1;
            while (j >= 0 && scores[order[j]] < scores[current])
            {
                order[j + 1] = order[j];
                j--;
            }

            order[j + 1] = current;
        }

        return order;
    }

    /// <summary>Converts normalized <c>cxcywh</c> boxes to <c>xyxy</c> pixel coordinates of a <paramref name="width"/>×<paramref name="height"/> image.</summary>
    public static void CxcywhToXyxyPixels(ReadOnlySpan<float> cxcywh, int width, int height, Span<float> xyxy)
    {
        if (cxcywh.Length % 4 != 0 || xyxy.Length != cxcywh.Length)
        {
            throw new ArgumentException("Boxes must be packed groups of four with matching output length.");
        }

        for (int i = 0; i < cxcywh.Length; i += 4)
        {
            float halfW = cxcywh[i + 2] * 0.5f;
            float halfH = cxcywh[i + 3] * 0.5f;
            xyxy[i] = (cxcywh[i] - halfW) * width;
            xyxy[i + 1] = (cxcywh[i + 1] - halfH) * height;
            xyxy[i + 2] = (cxcywh[i] + halfW) * width;
            xyxy[i + 3] = (cxcywh[i + 1] + halfH) * height;
        }
    }

    /// <summary>Binarizes mask logits: 1 where the logit is strictly above <paramref name="threshold"/> (equivalent to a
    /// sigmoid probability above 0.5 when the threshold is 0).</summary>
    public static void ThresholdLogits(ReadOnlySpan<float> logits, float threshold, Span<byte> mask)
    {
        if (mask.Length != logits.Length)
        {
            throw new ArgumentException("Mask buffer must match the logit count.", nameof(mask));
        }

        for (int i = 0; i < logits.Length; i++)
        {
            mask[i] = logits[i] > threshold ? (byte)1 : (byte)0;
        }
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
}
