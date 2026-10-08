namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Host reference for upstream <c>MoE.forward</c> after routing: routed SwiGLU experts plus the shared expert, accumulated in F32.</summary>
public static class DeepSeekV41MoeExecutor
{
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

        float[] scratch = new float[dim], hidden = new float[shared.Inter];
        for (int e = 0; e < numExperts; e++)
        {
            if (routed[e] is not { Count: > 0 } rows) continue;
            DeepSeekV41SwigluWeights w = experts.GetExpert(e);
            w.Validate();
            if (w.Dim != dim || w.Inter != shared.Inter)
                throw new InvalidOperationException($"Expert {e} is {w.Dim} x {w.Inter}, expected {dim} x {shared.Inter}.");
            foreach ((int token, float weight) in rows)
            {
                Forward(w, x.Slice(token * dim, dim), weight, swigluLimit, hidden, scratch);
                Span<float> row = y.Slice(token * dim, dim);
                for (int d = 0; d < dim; d++) row[d] += scratch[d];
            }
        }

        for (int t = 0; t < tokens; t++)
        {
            Forward(shared, x.Slice(t * dim, dim), 1f, swigluLimit, hidden, scratch);
            Span<float> row = y.Slice(t * dim, dim);
            for (int d = 0; d < dim; d++) row[d] += scratch[d];
        }
    }

    /// <summary>One expert on one token: clamp, <c>silu(gate) * up</c>, scale by <paramref name="weight"/>, then the down projection.</summary>
    internal static void Forward(DeepSeekV41SwigluWeights w, ReadOnlySpan<float> x, float weight, float limit, float[] hidden, Span<float> output)
    {
        int dim = w.Dim, inter = w.Inter;
        // two sequential dots per hidden unit, exactly as a fused loop would compute them
        float[] gates = w.W1.Linear(x, 1, dim, inter), ups = w.W3.Linear(x, 1, dim, inter);
        for (int i = 0; i < inter; i++)
        {
            float gate = gates[i], up = ups[i];
            if (limit > 0f)
            {
                up = Math.Clamp(up, -limit, limit);
                gate = MathF.Min(gate, limit);
            }
            hidden[i] = weight * (gate / (1f + MathF.Exp(-gate)) * up);
        }
        w.W2.Linear(hidden.AsSpan(0, inter), 1, inter, dim).CopyTo(output);
    }
}
