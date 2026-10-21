namespace HartsyInference.Core.Backends;

/// <summary>The numerical model of INT8 attention (SageAttention v1, arXiv:2410.02367) for one head, the oracle for the
/// INT8 kernel rows: Q and K quantized to per-row absmax INT8, K optionally smoothed first by subtracting its per-channel
/// mean over the keys (each query row's scores shift by one constant, so the softmax is unchanged), exact INT32 dot
/// products, dequantized scores, F32 softmax and PV.</summary>
public static class SageAttentionReference
{
    /// <summary>Attention over one head. <paramref name="q"/> is <c>[seqQ, dim]</c>, <paramref name="k"/> and
    /// <paramref name="v"/> are <c>[seqK, dim]</c>, all row-major.</summary>
    public static float[] Run(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v, int seqQ, int seqK, int dim,
        float scale, bool smoothK)
    {
        if (seqQ < 0 || seqK <= 0 || dim <= 0)
            throw new ArgumentOutOfRangeException(nameof(seqK), $"Needs at least one key and a positive dim; got seqQ {seqQ}, seqK {seqK}, dim {dim}.");
        if (q.Length < seqQ * dim || k.Length < seqK * dim || v.Length < seqK * dim)
            throw new ArgumentException($"Q, K and V hold {q.Length}, {k.Length} and {v.Length} values; [{seqQ}|{seqK}, {dim}] needs more.");
        float[] kWork = k.ToArray();
        if (smoothK)
        {
            for (int d = 0; d < dim; d++)
            {
                double mean = 0;
                for (int j = 0; j < seqK; j++) mean += kWork[j * dim + d];
                float m = (float)(mean / seqK);
                for (int j = 0; j < seqK; j++) kWork[j * dim + d] -= m;
            }
        }

        sbyte[] q8 = new sbyte[seqQ * dim];
        float[] qScale = new float[seqQ];
        QuantizeRows(q, q8, qScale, seqQ, dim);
        sbyte[] k8 = new sbyte[seqK * dim];
        float[] kScale = new float[seqK];
        QuantizeRows(kWork, k8, kScale, seqK, dim);

        float[] output = new float[seqQ * dim];
        float[] scores = new float[seqK];
        for (int i = 0; i < seqQ; i++)
        {
            float max = float.NegativeInfinity;
            for (int j = 0; j < seqK; j++)
            {
                int dot = 0;
                for (int d = 0; d < dim; d++) dot += q8[i * dim + d] * k8[j * dim + d];
                scores[j] = dot * qScale[i] * kScale[j] * scale;
                max = Math.Max(max, scores[j]);
            }
            double sum = 0;
            for (int j = 0; j < seqK; j++)
            {
                scores[j] = MathF.Exp(scores[j] - max);
                sum += scores[j];
            }
            float inv = (float)(1.0 / sum);
            for (int d = 0; d < dim; d++)
            {
                double acc = 0;
                for (int j = 0; j < seqK; j++) acc += (double)scores[j] * v[j * dim + d];
                output[i * dim + d] = (float)(acc * inv);
            }
        }
        return output;
    }

    /// <summary>Per-row absmax INT8: scale = absmax / 127 (1 for an all-zero row), round to nearest, clamp to ±127.</summary>
    public static void QuantizeRows(ReadOnlySpan<float> src, Span<sbyte> dst, Span<float> scales, int rows, int dim)
    {
        if (src.Length < rows * dim || dst.Length < rows * dim || scales.Length < rows)
            throw new ArgumentException($"[{rows}, {dim}] needs {rows * dim} values and {rows} scales.");
        for (int r = 0; r < rows; r++)
        {
            float amax = 0f;
            for (int d = 0; d < dim; d++) amax = Math.Max(amax, Math.Abs(src[r * dim + d]));
            float scale = amax > 0f ? amax / 127f : 1f;
            scales[r] = scale;
            float inv = 1f / scale;
            for (int d = 0; d < dim; d++)
                dst[r * dim + d] = (sbyte)Math.Clamp(MathF.Round(src[r * dim + d] * inv), -127f, 127f);
        }
    }
}
