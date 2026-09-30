namespace HartsyInference.Cpu.Tests.Moe;

/// <summary>Verbatim routing ports from tests/HartsyInference.LLM.Tests/MoeTests.cs (score plus selection only), the
/// independent oracle the CPU MoE reference must match bit for bit.</summary>
public static class MoeRoutingPorts
{
    // Softmax router with plain scan top-k, as in MoeTests.Reference.
    public static void SoftmaxRoute(float[] logits, int t, int e, int topK, bool norm, int[] idxOut, float[] wOut)
    {
        float[] score = new float[e];
        float mx = float.NegativeInfinity;
        for (int i = 0; i < e; i++) { score[i] = logits[t * e + i]; mx = MathF.Max(mx, score[i]); }
        float sum = 0f;
        for (int i = 0; i < e; i++) { score[i] = MathF.Exp(score[i] - mx); sum += score[i]; }
        for (int i = 0; i < e; i++) score[i] /= sum;

        int[] pick = new int[topK];
        float wsum = 0f;
        for (int kk = 0; kk < topK; kk++)
        {
            int best = -1; float bv = float.NegativeInfinity;
            for (int i = 0; i < e; i++)
            {
                bool taken = false;
                for (int j = 0; j < kk; j++) if (pick[j] == i) { taken = true; break; }
                if (!taken && score[i] > bv) { bv = score[i]; best = i; }
            }
            pick[kk] = best; wsum += bv;
        }
        for (int kk = 0; kk < topK; kk++)
        {
            int ex = pick[kk];
            idxOut[t * topK + kk] = ex;
            wOut[t * topK + kk] = norm ? score[ex] / wsum : score[ex];
        }
    }

    // DeepSeek-V3 noaux_tc router, as in MoeTests.GroupRoutingReference.
    public static void GroupRoute(float[] logits, float[] bias, int t, int e, int topK, int nGroup, int topkGroup,
        float scaling, int[] idxOut, float[] wOut)
    {
        int perGroup = e / nGroup;
        float[] sig = new float[e], choice = new float[e];
        for (int i = 0; i < e; i++)
        {
            sig[i] = 1f / (1f + MathF.Exp(-logits[t * e + i]));
            choice[i] = sig[i] + bias[i];
        }
        float[] groupScore = new float[nGroup];
        for (int g = 0; g < nGroup; g++)
        {
            float t1 = float.NegativeInfinity, t2 = float.NegativeInfinity;
            for (int j = 0; j < perGroup; j++) { float v = choice[g * perGroup + j]; if (v > t1) { t2 = t1; t1 = v; } else if (v > t2) t2 = v; }
            groupScore[g] = t1 + t2;
        }
        bool[] kept = new bool[nGroup];
        for (int kk = 0; kk < topkGroup; kk++)
        {
            int bg = -1; float bv = float.NegativeInfinity;
            for (int g = 0; g < nGroup; g++) if (!kept[g] && groupScore[g] > bv) { bv = groupScore[g]; bg = g; }
            if (bg >= 0) kept[bg] = true;
        }
        float[] tmp = new float[e];
        for (int i = 0; i < e; i++) tmp[i] = kept[i / perGroup] ? choice[i] : 0f;
        int[] pick = new int[topK];
        float wsum = 0f;
        for (int kk = 0; kk < topK; kk++)
        {
            int best = -1; float bv = float.NegativeInfinity;
            for (int i = 0; i < e; i++)
            {
                bool taken = false; for (int j = 0; j < kk; j++) if (pick[j] == i) { taken = true; break; }
                if (!taken && tmp[i] > bv) { bv = tmp[i]; best = i; }
            }
            pick[kk] = best; wsum += sig[best];
        }
        for (int kk = 0; kk < topK; kk++)
        {
            idxOut[t * topK + kk] = pick[kk];
            wOut[t * topK + kk] = sig[pick[kk]] / (wsum + 1e-20f) * scaling;
        }
    }
}
