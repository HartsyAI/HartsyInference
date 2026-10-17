using HartsyInference.LLM.Looped;

namespace HartsyInference.LLM.Tests.Looped;

/// <summary>A scalar, double-precision implementation of the looped normalized-GPT forward written from the reference model (vLLM's
/// <c>dory.py</c> and llama.cpp's <c>dory.cpp</c>) with no engine ops at all, so it checks what the backend-composed
/// <see cref="LoopedTransformer"/> computes rather than how. Keeps its own key/value store so prompts can be fed in chunks.</summary>
internal sealed class LoopedReference
{
    private readonly LoopedTransformerConfig _cfg;
    private readonly IReadOnlyDictionary<string, float[]> _w;
    // Per cache slot, per position: flattened [kvHeads, headDim] keys and values of the last loop that wrote them.
    private readonly List<double[]>[] _keys;
    private readonly List<double[]>[] _values;

    public LoopedReference(LoopedTransformerConfig cfg, IReadOnlyDictionary<string, float[]> weights)
    {
        _cfg = cfg;
        _w = weights;
        int slots = LoopedSchedule.MaxSlotCount(cfg);
        _keys = new List<double[]>[slots];
        _values = new List<double[]>[slots];
        for (int i = 0; i < slots; i++) { _keys[i] = []; _values[i] = []; }
    }

    /// <summary>Runs one chunk of <paramref name="tokens"/> at positions <paramref name="posStart"/>.. with <paramref name="loops"/> passes; returns one logits row per token.</summary>
    public double[][] ForwardChunk(int[] tokens, int posStart, int loops)
    {
        int t = tokens.Length;
        int h = _cfg.HiddenSize;
        double[][] hidden = new double[t][];
        float[] embed = _w["model.embed_tokens.weight"];
        for (int i = 0; i < t; i++)
        {
            hidden[i] = new double[h];
            for (int j = 0; j < h; j++) hidden[i][j] = embed[tokens[i] * h + j];
        }
        int inputAttn = _cfg.AttentionLayersIn(0, _cfg.InputLayers);
        int outputAttn = _cfg.AttentionLayersIn(_cfg.InputLayers + _cfg.RecurrentLayers, _cfg.OutputLayers);
        int recurrentAttn = _cfg.AttentionLayersIn(_cfg.InputLayers, _cfg.RecurrentLayers);
        int slot = 0;
        for (int p = 0; p < _cfg.InputLayers; p++)
            hidden = Layer(p, hidden, posStart, _cfg.IsAttention(p) ? slot++ : -1);
        double[][] inputH = hidden;
        double[][] state = inputH;
        for (int k = 0; k < loops; k++)
        {
            double[][] cur = new double[t][];
            for (int i = 0; i < t; i++) cur[i] = JustNorm(Add(state[i], inputH[i]));
            int loopBase = inputAttn + outputAttn + (_cfg.KvCacheMode == LoopedKvCacheMode.PerLoop ? k * recurrentAttn : 0);
            int j = 0;
            for (int r = 0; r < _cfg.RecurrentLayers; r++)
            {
                int p = _cfg.InputLayers + r;
                cur = Layer(p, cur, posStart, _cfg.IsAttention(p) ? loopBase + j++ : -1);
            }
            state = new double[t][];
            for (int i = 0; i < t; i++) state[i] = JustNorm(cur[i]);
        }
        hidden = state;
        int outputSlot = inputAttn;
        for (int o = 0; o < _cfg.OutputLayers; o++)
        {
            int p = _cfg.InputLayers + _cfg.RecurrentLayers + o;
            hidden = Layer(p, hidden, posStart, _cfg.IsAttention(p) ? outputSlot++ : -1);
        }
        float[] head = _w["lm_head.weight"];
        float[] logitScale = _w["lm_head.logit_scale.weight"];
        double[][] logits = new double[t][];
        for (int i = 0; i < t; i++)
        {
            logits[i] = MatVec(head, _cfg.VocabSize, h, hidden[i]);
            for (int v = 0; v < _cfg.VocabSize; v++) logits[i][v] *= logitScale[v] / _cfg.InitStd;
        }
        return logits;
    }

    private double[][] Layer(int p, double[][] hidden, int posStart, int slot)
    {
        int t = hidden.Length;
        string prefix = $"model.layers.{p}";
        double[][] sublayer = _cfg.IsAttention(p) ? Attention(prefix, p, hidden, posStart, slot) : Mlp(prefix, hidden);
        float[] alphaRaw = _w[$"{prefix}.residual_alpha.weight"];
        double[][] result = new double[t][];
        for (int i = 0; i < t; i++)
        {
            double[] a = JustNorm(hidden[i]);
            double[] b = JustNorm(sublayer[i]);
            double[] mixed = new double[a.Length];
            for (int j = 0; j < a.Length; j++)
            {
                double alpha = Math.Abs(alphaRaw[j] * _cfg.AlphaInit / _cfg.InitStd);
                mixed[j] = a[j] + alpha * (b[j] - a[j]);
            }
            result[i] = JustNorm(mixed);
        }
        return result;
    }

    private double[][] Attention(string prefix, int p, double[][] hidden, int posStart, int slot)
    {
        int t = hidden.Length;
        int h = _cfg.HiddenSize, hq = _cfg.NumHeads, hkv = _cfg.NumKvHeads, d = _cfg.HeadDim;
        int nq = _cfg.QDim, nk = _cfg.KvDim;
        float[] wq = _w[$"{prefix}.self_attn.q_proj.weight"], wk = _w[$"{prefix}.self_attn.k_proj.weight"];
        float[] wv = _w[$"{prefix}.self_attn.v_proj.weight"], wo = _w[$"{prefix}.self_attn.o_proj.weight"];
        float[] wg = _w[$"{prefix}.self_attn.gate_proj.weight"];
        float[] sqk = _w[$"{prefix}.self_attn.qk_scale.weight"], sGate = _w[$"{prefix}.self_attn.gate_scale.weight"];
        double theta = _cfg.RopeThetas[p];
        double gateInit = _cfg.EffectiveGateInit;
        double[][] q = new double[t][];
        double[][] gate = new double[t][];
        // Write every token's key/value first: a token attends keys up to and including its own position within the chunk.
        for (int i = 0; i < t; i++)
        {
            int pos = posStart + i;
            double[] qi = MatVec(wq, nq, h, hidden[i]);
            double[] ki = MatVec(wk, nk, h, hidden[i]);
            double[] vi = MatVec(wv, nk, h, hidden[i]);
            gate[i] = MatVec(wg, nq, h, hidden[i]);
            for (int head = 0; head < hq; head++) Rope(qi, head * d, d, pos, theta);
            for (int head = 0; head < hkv; head++) Rope(ki, head * d, d, pos, theta);
            for (int head = 0; head < hq; head++)
            {
                double[] normed = JustNorm(qi.AsSpan(head * d, d).ToArray());
                for (int j = 0; j < d; j++)
                {
                    double s = sqk[head * d + j] * Math.Sqrt(d) / _cfg.InitStd;
                    qi[head * d + j] = s * s * normed[j];
                }
            }
            for (int head = 0; head < hkv; head++)
            {
                double[] normed = JustNorm(ki.AsSpan(head * d, d).ToArray());
                normed.CopyTo(ki, head * d);
            }
            q[i] = qi;
            if (_keys[slot].Count > pos) { _keys[slot][pos] = ki; _values[slot][pos] = vi; }
            else { _keys[slot].Add(ki); _values[slot].Add(vi); }
        }
        int window = _cfg.IsSliding(p) ? _cfg.SlidingWindow : 0;
        double[][] output = new double[t][];
        for (int i = 0; i < t; i++)
        {
            int pos = posStart + i;
            int kMin = window > 0 ? Math.Max(0, pos - window + 1) : 0;
            double[] merged = new double[nq];
            for (int head = 0; head < hq; head++)
            {
                int kvHead = head / _cfg.KvGroup;
                double[] scores = new double[pos - kMin + 1];
                double max = double.NegativeInfinity;
                for (int j = kMin; j <= pos; j++)
                {
                    double dot = 0;
                    for (int e = 0; e < d; e++) dot += q[i][head * d + e] * _keys[slot][j][kvHead * d + e];
                    scores[j - kMin] = dot / Math.Sqrt(d);
                    max = Math.Max(max, scores[j - kMin]);
                }
                double sum = 0;
                for (int j = 0; j < scores.Length; j++) { scores[j] = Math.Exp(scores[j] - max); sum += scores[j]; }
                for (int j = kMin; j <= pos; j++)
                {
                    double weight = scores[j - kMin] / sum;
                    for (int e = 0; e < d; e++) merged[head * d + e] += weight * _values[slot][j][kvHead * d + e];
                }
            }
            for (int j = 0; j < nq; j++)
            {
                double s = sGate[j] * Math.Sqrt(gateInit) / _cfg.InitStd;
                merged[j] *= 1.0 / (1.0 + Math.Exp(-gate[i][j] * s * s));
            }
            output[i] = MatVec(wo, h, nq, JustNorm(merged));
        }
        return output;
    }

    private double[][] Mlp(string prefix, double[][] hidden)
    {
        int t = hidden.Length;
        int h = _cfg.HiddenSize, ff = _cfg.IntermediateSize;
        float[] wGate = _w[$"{prefix}.mlp.gate_proj.weight"], wUp = _w[$"{prefix}.mlp.up_proj.weight"], wDown = _w[$"{prefix}.mlp.down_proj.weight"];
        float[] sg = _w[$"{prefix}.mlp.gate_scale.weight"], su = _w[$"{prefix}.mlp.up_scale.weight"];
        double[][] output = new double[t][];
        for (int i = 0; i < t; i++)
        {
            double[] g = MatVec(wGate, ff, h, hidden[i]);
            double[] u = MatVec(wUp, ff, h, hidden[i]);
            double[] m = new double[ff];
            for (int j = 0; j < ff; j++)
            {
                double gs = g[j] * sg[j] * Math.Sqrt(h);
                double us = u[j] * su[j] * Math.Sqrt(h);
                m[j] = gs / (1.0 + Math.Exp(-gs)) * us;
            }
            output[i] = MatVec(wDown, h, ff, JustNorm(m));
        }
        return output;
    }

    private double[] JustNorm(double[] x)
    {
        double sum = 0;
        foreach (double v in x) sum += v * v;
        double inv = 1.0 / Math.Sqrt(sum + _cfg.NormEps);
        double[] y = new double[x.Length];
        for (int i = 0; i < x.Length; i++) y[i] = x[i] * inv;
        return y;
    }

    /// <summary>NEOX rotate-half RoPE over one head: dim <c>i</c> pairs with <c>i + d/2</c>, frequency <c>theta^(-2i/d)</c>.</summary>
    private static void Rope(double[] x, int offset, int d, int pos, double theta)
    {
        int half = d / 2;
        for (int i = 0; i < half; i++)
        {
            double freq = Math.Pow(theta, -2.0 * i / d);
            double angle = pos * freq;
            double c = Math.Cos(angle), s = Math.Sin(angle);
            double a = x[offset + i], b = x[offset + i + half];
            x[offset + i] = a * c - b * s;
            x[offset + i + half] = b * c + a * s;
        }
    }

    private static double[] MatVec(float[] w, int rows, int cols, double[] x)
    {
        double[] y = new double[rows];
        for (int r = 0; r < rows; r++)
        {
            double acc = 0;
            for (int c = 0; c < cols; c++) acc += w[r * cols + c] * x[c];
            y[r] = acc;
        }
        return y;
    }

    private static double[] Add(double[] a, double[] b)
    {
        double[] y = new double[a.Length];
        for (int i = 0; i < a.Length; i++) y[i] = a[i] + b[i];
        return y;
    }
}
