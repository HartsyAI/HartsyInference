using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection.GroundingDino;

/// <summary>A single Grounding DINO detection: an axis-aligned box in image pixels, its confidence, and the decoded
/// caption phrase it grounds.</summary>
public readonly record struct GroundingDinoDetection(float X0, float Y0, float X1, float Y1, float Score, string Label);

/// <summary>Post-processing for Grounding DINO, mirroring <c>post_process_grounded_object_detection</c>: per-query
/// confidence is the max token probability (sigmoid of the contrastive logits); queries above <c>threshold</c> are
/// kept, their boxes converted from normalized cxcywh to image-pixel xyxy, and their label decoded from the caption
/// tokens whose probability exceeds <c>textThreshold</c>.</summary>
public static unsafe class GroundingDinoPipeline
{
    public static List<GroundingDinoDetection> PostProcess(Tensor logits, Tensor predBoxes, int[] inputIds,
        IReadOnlyList<string> idToToken, int imageHeight, int imageWidth, float threshold = 0.4f, float textThreshold = 0.3f)
    {
        int nq = (int)logits.Shape[1];
        int maxText = (int)logits.Shape[2];
        int t = inputIds.Length;
        float* lp = (float*)logits.DataPointer;
        float* bp = (float*)predBoxes.DataPointer;

        List<GroundingDinoDetection> results = new();
        for (int q = 0; q < nq; q++)
        {
            float best = 0f;
            for (int j = 0; j < t; j++)
            {
                float p = GdMath.Sigmoid(lp[(long)q * maxText + j]);
                if (p > best) best = p;
            }
            if (best <= threshold)
                continue;

            List<int> phraseTokens = new();
            for (int j = 0; j < t; j++)
            {
                float p = GdMath.Sigmoid(lp[(long)q * maxText + j]);
                if (p > textThreshold)
                    phraseTokens.Add(inputIds[j]);
            }
            string label = Decode(phraseTokens, idToToken);

            float cx = bp[(long)q * 4 + 0], cy = bp[(long)q * 4 + 1], bw = bp[(long)q * 4 + 2], bh = bp[(long)q * 4 + 3];
            float x0 = (cx - bw / 2f) * imageWidth;
            float y0 = (cy - bh / 2f) * imageHeight;
            float x1 = (cx + bw / 2f) * imageWidth;
            float y1 = (cy + bh / 2f) * imageHeight;
            results.Add(new GroundingDinoDetection(x0, y0, x1, y1, best, label));
        }
        results.Sort((a, b) => b.Score.CompareTo(a.Score));
        return results;
    }

    private static string Decode(List<int> ids, IReadOnlyList<string> idToToken)
    {
        System.Text.StringBuilder sb = new();
        foreach (int id in ids)
        {
            if (id < 0 || id >= idToToken.Count) continue;
            string tok = idToToken[id];
            if (tok is "[CLS]" or "[SEP]" or "[PAD]") continue;
            if (tok.StartsWith("##"))
                sb.Append(tok.Substring(2));
            else
            {
                if (sb.Length > 0) sb.Append(' ');
                sb.Append(tok);
            }
        }
        return sb.ToString();
    }

    /// <summary>Loads a BERT <c>vocab.txt</c> as an id→token list (line number = token id).</summary>
    public static string[] LoadVocab(string vocabPath) => File.ReadAllLines(vocabPath);
}
