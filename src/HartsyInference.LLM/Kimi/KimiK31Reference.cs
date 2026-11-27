using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Backends.Gating;

namespace HartsyInference.LLM.Kimi;

/// <summary>
/// Scalar F32 references for the Kimi K3.1 Pro pieces that have no existing engine owner: the KDA forget gate and
/// delta-rule recurrence, block attention residuals, the DSA indexer's top-k selection, and the sigmoid router.
/// These are the CPU correctness oracle that kernels are checked against; they are not on a hot path.
/// Every formula is the one the NVIDIA NeMo Automodel K3 port runs (<c>models/kimi_k3/model.py</c>, <c>kda_fused</c>),
/// which is the only public K3.1 code. The checkpoint shapes were read from the safetensors headers.
/// </summary>
public static class KimiK31Reference
{
    /// <summary>Keys each query attends to in a DSA layer once the context is longer than this (<c>index_topk</c>).</summary>
    public const int IndexTopK = 2048;

    /// <summary>Stable logistic function.</summary>
    public static float Sigmoid(float x) => x >= 0f ? 1f / (1f + MathF.Exp(-x)) : MathF.Exp(x) / (1f + MathF.Exp(x));

    /// <summary>
    /// KDA forget gate (log-decay) for one key channel of one head:
    /// <c>g = lowerBound · sigmoid(exp(A_log) · (rawGate + dt_bias))</c>. The checkpoint sets <c>gate_lower_bound</c>
    /// to -5.0, so the bounded form applies: g lies in (lowerBound, 0). The decay applied to the state is <c>exp(g)</c>.
    /// <paramref name="rawGate"/> is <c>f_b_proj(f_a_proj(x))</c> for that channel; <paramref name="aLog"/> is the
    /// channel's <c>A_log</c> entry; <paramref name="dtBias"/> is the channel's <c>dt_bias</c>.
    /// </summary>
    public static float KdaLogDecay(float rawGate, float dtBias, float aLog, float lowerBound)
        => lowerBound * Sigmoid(MathF.Exp(aLog) * (rawGate + dtBias));

    /// <summary>
    /// KDA delta-rule recurrence for one head, F32, in place on <paramref name="state"/> (<c>[dk, dv]</c>, row-major).
    /// Per token: <c>S ← Diag(exp(g_t)) · S</c>; <c>kv = Sᵀ·k_t</c>; <c>δ = (v_t − kv) · β_t</c>; <c>S ← S + k_t·δᵀ</c>;
    /// <c>o_t = Sᵀ·q_t</c>. The decay is per key channel (rows of S). <paramref name="q"/> and <paramref name="k"/> must
    /// already be L2-normalized (and <paramref name="q"/> scaled by <c>dk^-0.5</c>) by the caller, as the kernel does.
    /// Shapes: <paramref name="q"/>, <paramref name="k"/>, <paramref name="logDecay"/> are <c>[tokens, dk]</c>;
    /// <paramref name="v"/> and <paramref name="output"/> are <c>[tokens, dv]</c>; <paramref name="beta"/> is <c>[tokens]</c>.
    /// </summary>
    public static void KdaRecurrenceHead(
        ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v, ReadOnlySpan<float> logDecay,
        ReadOnlySpan<float> beta, int tokens, int dk, int dv, Span<float> state, Span<float> output)
    {
        if (state.Length != dk * dv) throw new ArgumentException($"state must hold dk*dv = {dk * dv} values.", nameof(state));
        if (output.Length < tokens * dv) throw new ArgumentException("output is too small.", nameof(output));
        Span<float> kv = stackalloc float[dv];
        for (int t = 0; t < tokens; t++)
        {
            ReadOnlySpan<float> qt = q.Slice(t * dk, dk), kt = k.Slice(t * dk, dk), vt = v.Slice(t * dv, dv);
            ReadOnlySpan<float> gt = logDecay.Slice(t * dk, dk);
            for (int i = 0; i < dk; i++)
            {
                float decay = MathF.Exp(gt[i]);
                Span<float> row = state.Slice(i * dv, dv);
                for (int j = 0; j < dv; j++) row[j] *= decay;
            }
            kv.Clear();
            for (int i = 0; i < dk; i++)
            {
                ReadOnlySpan<float> row = state.Slice(i * dv, dv);
                for (int j = 0; j < dv; j++) kv[j] += row[j] * kt[i];
            }
            float b = beta[t];
            for (int i = 0; i < dk; i++)
            {
                Span<float> row = state.Slice(i * dv, dv);
                for (int j = 0; j < dv; j++) row[j] += kt[i] * (vt[j] - kv[j]) * b;
            }
            Span<float> ot = output.Slice(t * dv, dv);
            ot.Clear();
            for (int i = 0; i < dk; i++)
            {
                ReadOnlySpan<float> row = state.Slice(i * dv, dv);
                for (int j = 0; j < dv; j++) ot[j] += row[j] * qt[i];
            }
        }
    }

    /// <summary>
    /// DSA indexer score of one query position against one key position:
    /// <c>Σ_h relu(q_h · k) · w_h</c>, where <paramref name="headWeights"/> are <c>weights_proj(x)</c> already multiplied by
    /// the score scale (<c>(index_head_dim · index_num_heads)^-0.5</c>). <paramref name="query"/> is <c>[heads, dim]</c>.
    /// There is no softmax before top-k; the ReLU and the head-weighted sum are the whole score.
    /// </summary>
    public static float DsaIndexerScore(ReadOnlySpan<float> query, ReadOnlySpan<float> key, ReadOnlySpan<float> headWeights, int heads, int dim)
    {
        if (query.Length != heads * dim || key.Length != dim || headWeights.Length != heads)
            throw new ArgumentException("query must be [heads, dim], key [dim], headWeights [heads].");
        float score = 0f;
        for (int h = 0; h < heads; h++)
        {
            float dot = 0f;
            ReadOnlySpan<float> qh = query.Slice(h * dim, dim);
            for (int i = 0; i < dim; i++) dot += qh[i] * key[i];
            score += MathF.Max(dot, 0f) * headWeights[h];
        }
        return score;
    }

    /// <summary>
    /// Causal top-k key selection for one query. Candidates are keys <c>0..queryPos</c>. If there are at most
    /// <paramref name="topK"/> of them, all are kept (short contexts attend fully). Otherwise the <paramref name="topK"/>
    /// highest-scoring keys are kept; non-finite scores are dropped and ties go to the lower key index. The result is
    /// ascending by key position.
    /// </summary>
    public static int[] SelectCausalTopK(ReadOnlySpan<float> scores, int queryPos, int topK = IndexTopK)
    {
        if (queryPos < 0 || queryPos >= scores.Length) throw new ArgumentOutOfRangeException(nameof(queryPos));
        int candidates = queryPos + 1;
        if (candidates <= topK)
        {
            int[] all = new int[candidates];
            for (int i = 0; i < candidates; i++) all[i] = i;
            return all;
        }
        float[] values = scores.Slice(0, candidates).ToArray();
        List<int> order = new(candidates);
        for (int i = 0; i < candidates; i++)
        {
            if (float.IsFinite(values[i])) order.Add(i);
        }
        order.Sort((a, b) =>
        {
            int byScore = values[b].CompareTo(values[a]);
            return byScore != 0 ? byScore : a.CompareTo(b);
        });
        int keep = Math.Min(topK, order.Count);
        List<int> chosen = order.GetRange(0, keep);
        chosen.Sort();
        return chosen.ToArray();
    }

    /// <summary>
    /// Sigmoid router with a selection-only correction bias (<c>noaux_tc</c>). Selection uses <c>sigmoid(logit) + bias</c>;
    /// the weight of a chosen expert is its raw sigmoid score. Weights are renormalized when
    /// <paramref name="topK"/> &gt; 1 and <paramref name="renormalize"/> is set, then scaled by <paramref name="routedScale"/>.
    /// </summary>
    public static void SigmoidRoute(ReadOnlySpan<float> logits, ReadOnlySpan<float> bias, int topK, bool renormalize,
        float routedScale, Span<int> indices, Span<float> weights)
    {
        int experts = logits.Length;
        if (bias.Length != experts || indices.Length < topK || weights.Length < topK)
            throw new ArgumentException("bias must match the expert count; indices and weights must hold topK entries.");
        float[] scores = new float[experts];
        float[] choice = new float[experts];
        for (int e = 0; e < experts; e++)
        {
            scores[e] = Sigmoid(logits[e]);
            choice[e] = scores[e] + bias[e];
        }
        int[] order = Enumerable.Range(0, experts).OrderByDescending(e => choice[e]).ThenBy(e => e).Take(topK).ToArray();
        float sum = 0f;
        for (int i = 0; i < topK; i++)
        {
            indices[i] = order[i];
            weights[i] = scores[order[i]];
            sum += weights[i];
        }
        float denom = renormalize && topK > 1 ? sum + 1e-20f : 1f;
        for (int i = 0; i < topK; i++) weights[i] = weights[i] / denom * routedScale;
    }

    /// <summary>
    /// The routed-expert / shared-expert activation (<c>situ</c>): <c>4·tanh(gate/4)·sigmoid(gate)</c> times
    /// <c>25·tanh(up/25)</c>. The checkpoint's <c>w1</c> is the gate and <c>w3</c> the up projection (NeMo splits
    /// <c>x.chunk(2)</c> as gate first). The gate and value factors come from the shared
    /// <see cref="GluReference"/> under <see cref="GluActivation.Situ"/>.
    /// </summary>
    public static float Situ(float gate, float up) => GluReference.Gate(gate, GluActivation.Situ) * GluReference.Value(up, GluActivation.Situ);
}
