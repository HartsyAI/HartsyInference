namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Host reference for upstream <c>MoE.forward</c> after routing: routed SwiGLU experts plus the shared expert, accumulated in F32.</summary>
public static class DeepSeekV41MoeExecutor
{
    // Tokens run through one expert together. A stored-form weight is decoded once per call, so batching decodes each expert once per forward instead of once
    // per token; the cap only bounds the scratch arrays. Every output element is the same sequential dot whatever the batch, so results are bit-identical.
    private const int BatchRows = 256;

    /// <summary>Computes <c>y[t] = sum_j w[t,j] * expert_{idx[t,j]}(x[t]) + shared(x[t])</c>.</summary>
    /// <param name="x">Tokens, <c>[tokens, dim]</c>.</param>
    /// <param name="tokens">Row count.</param>
    /// <param name="topkIdx">Chosen expert per (token, slot), <c>[tokens, k]</c>; out-of-range ids are skipped.</param>
    /// <param name="topkWeight">Combine weight per (token, slot), <c>[tokens, k]</c>.</param>
    /// <param name="k">Experts per token.</param>
    /// <param name="numExperts">Routed experts in the layer.</param>
    /// <param name="experts">Source of routed expert weights, asked only for experts that received tokens.</param>
    /// <param name="shared">The shared expert every token goes through.</param>
    /// <param name="swigluLimit">Clamp from the config; zero or negative disables it.</param>
    /// <param name="y">Receives <c>[tokens, dim]</c>; overwritten.</param>
    public static void Run(ReadOnlySpan<float> x, int tokens, ReadOnlySpan<int> topkIdx, ReadOnlySpan<float> topkWeight, int k,
        int numExperts, IDeepSeekV41ExpertSource experts, DeepSeekV41SwigluWeights shared, float swigluLimit, Span<float> y)
    {
        shared.Validate();
        int dim = shared.Dim;
        if (x.Length != (long)tokens * dim || y.Length != x.Length) throw new ArgumentException("x and y must each hold tokens x dim values.");
        if (topkIdx.Length != (long)tokens * k || topkWeight.Length != topkIdx.Length) throw new ArgumentException("topk arrays must hold tokens x k entries.");

        y.Clear();
        List<(int Token, float Weight)>[] routed = new List<(int, float)>[numExperts];
        for (int t = 0; t < tokens; t++)
            for (int j = 0; j < k; j++)
            {
                int e = topkIdx[t * k + j];
                if ((uint)e >= (uint)numExperts) continue;
                (routed[e] ??= []).Add((t, topkWeight[t * k + j]));
            }

        for (int e = 0; e < numExperts; e++)
        {
            if (routed[e] is not { Count: > 0 } rows) continue;
            DeepSeekV41SwigluWeights w = experts.GetExpert(e);
            w.Validate();
            if (w.Dim != dim || w.Inter != shared.Inter)
                throw new InvalidOperationException($"Expert {e} is {w.Dim} x {w.Inter}, expected {dim} x {shared.Inter}.");
            for (int start = 0; start < rows.Count; start += BatchRows)
            {
                int n = Math.Min(BatchRows, rows.Count - start);
                float[] batch = new float[n * dim], weights = new float[n];
                for (int i = 0; i < n; i++)
                {
                    (int token, float weight) = rows[start + i];
                    x.Slice(token * dim, dim).CopyTo(batch.AsSpan(i * dim, dim));
                    weights[i] = weight;
                }
                float[] outputs = ForwardBatch(w, batch, n, weights, swigluLimit);
                for (int i = 0; i < n; i++)
                {
                    Span<float> row = y.Slice(rows[start + i].Token * dim, dim);
                    ReadOnlySpan<float> o = outputs.AsSpan(i * dim, dim);
                    for (int d = 0; d < dim; d++) row[d] += o[d];
                }
            }
        }

        float[] ones = new float[Math.Min(BatchRows, tokens)];
        Array.Fill(ones, 1f);
        for (int start = 0; start < tokens; start += BatchRows)
        {
            int n = Math.Min(BatchRows, tokens - start);
            float[] outputs = ForwardBatch(shared, x.Slice(start * dim, n * dim).ToArray(), n, ones, swigluLimit);
            Span<float> rows = y.Slice(start * dim, n * dim);
            for (int i = 0; i < rows.Length; i++) rows[i] += outputs[i];
        }
    }

    /// <summary>One SwiGLU expert over <paramref name="n"/> tokens: <c>x</c> is <c>[n, dim]</c>, <c>weights[i]</c> scales token i, the result is <c>[n, dim]</c>.</summary>
    internal static float[] ForwardBatch(DeepSeekV41SwigluWeights w, float[] x, int n, float[] weights, float limit)
    {
        int dim = w.Dim, inter = w.Inter;
        float[] gates = w.W1.Linear(x, n, dim, inter), ups = w.W3.Linear(x, n, dim, inter), hidden = new float[n * inter];
        for (int t = 0; t < n; t++)
            for (int i = 0; i < inter; i++)
            {
                float gate = gates[t * inter + i], up = ups[t * inter + i];
                if (limit > 0f)
                {
                    up = Math.Clamp(up, -limit, limit);
                    gate = MathF.Min(gate, limit);
                }
                hidden[t * inter + i] = weights[t] * (gate / (1f + MathF.Exp(-gate)) * up);
            }
        return w.W2.Linear(hidden, n, inter, dim);
    }
}
