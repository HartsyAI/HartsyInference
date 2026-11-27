using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

/// <summary>
/// A scalar double-precision transcription of the upstream TRELLIS.2 flow forward (ModulatedTransformerCrossBlock with share_mod,
/// QK RMS norm, 3D RoPE and the model-level embedder; plus Pixal3D's projection branch), written from the upstream Python independently of the engine's ops. It shares
/// this repository's reading of upstream and is not a parity reference; it catches wiring mistakes in the engine's block.
/// </summary>
internal sealed class Trellis2Reference
{
    private readonly Trellis2FlowConfig _config;
    private readonly Dictionary<string, float[]> _w;

    internal Trellis2Reference(Trellis2FlowConfig config, IReadOnlyDictionary<string, Tensor> weights)
    {
        _config = config;
        _w = weights.ToDictionary(p => p.Key, p => Trellis2TestData.ToArray(p.Value));
    }

    /// <summary>Velocity rows <c>[tokens][out]</c> for input rows <c>[tokens][in]</c> at positions <c>[tokens][3]</c>; <paramref name="proj"/>
    /// holds one feature row per token for a projection flow.</summary>
    internal double[][] Forward(double[][] input, double[][] positions, double tModel, double[][] cond, double[][]? proj = null)
    {
        if (_config.HasProjection != (proj is not null))
            throw new ArgumentException("A projection flow needs its feature rows and a plain flow takes none.", nameof(proj));
        int c = _config.ModelChannels;
        double[] freq = new double[256];
        for (int i = 0; i < 128; i++)
        {
            double angle = tModel * Math.Exp(-Math.Log(10000.0) * i / 128.0);
            freq[i] = Math.Cos(angle);
            freq[128 + i] = Math.Sin(angle);
        }

        double[] emb = Linear(Linear(freq, "t_embedder.mlp.0", c).Select(Silu).ToArray(), "t_embedder.mlp.2", c);
        double[] shared = Linear(emb.Select(Silu).ToArray(), "adaLN_modulation.1", 6 * c);
        double[][] x = input.Select(row => Linear(row, "input_layer", c)).ToArray();
        for (int b = 0; b < _config.NumBlocks; b++)
            x = Block(x, $"blocks.{b}", shared, positions, cond, proj);
        return x.Select(row => Linear(LayerNorm(row, null, null, 1e-5), "out_layer", _config.OutChannels)).ToArray();
    }

    private double[][] Block(double[][] x, string p, double[] shared, double[][] positions, double[][] cond, double[][]? proj)
    {
        // Written out, not read from the config, so a wrong scope in the engine's key map cannot cancel against itself here.
        string scope = _config.HasProjection ? $"{p}.cross_attn.cross_attn_block" : $"{p}.cross_attn";
        int c = _config.ModelChannels, heads = _config.NumHeads, hd = _config.HeadDim, n = x.Length;
        double[] mod = new double[6 * c];
        for (int i = 0; i < mod.Length; i++)
            mod[i] = _w[$"{p}.modulation"][i] + shared[i];
        double[] Chunk(int k) => mod[(k * c)..((k + 1) * c)];
        double[] shiftMsa = Chunk(0), scaleMsa = Chunk(1), gateMsa = Chunk(2), shiftMlp = Chunk(3), scaleMlp = Chunk(4), gateMlp = Chunk(5);

        double[][] h = x.Select(row => Modulate(LayerNorm(row, null, null, 1e-6), scaleMsa, shiftMsa)).ToArray();
        double[][] qkv = h.Select(row => Linear(row, $"{p}.self_attn.to_qkv", 3 * c)).ToArray();
        double[][] q = qkv.Select(r => r[..c]).ToArray(), k = qkv.Select(r => r[c..(2 * c)]).ToArray(), v = qkv.Select(r => r[(2 * c)..]).ToArray();
        q = HeadNorm(q, $"{p}.self_attn.q_rms_norm.gamma");
        k = HeadNorm(k, $"{p}.self_attn.k_rms_norm.gamma");
        for (int t = 0; t < n; t++)
        {
            q[t] = Rope(q[t], positions[t]);
            k[t] = Rope(k[t], positions[t]);
        }

        double[][] attended = Attention(q, k, v, heads, hd);
        for (int t = 0; t < n; t++)
        {
            double[] o = Linear(attended[t], $"{p}.self_attn.to_out", c);
            for (int i = 0; i < c; i++)
                x[t][i] += o[i] * gateMsa[i];
        }

        double[][] hc = x.Select(row => LayerNorm(row, _w[$"{p}.norm2.weight"], _w[$"{p}.norm2.bias"], 1e-6)).ToArray();
        double[][] cq = hc.Select(row => Linear(row, $"{scope}.to_q", c)).ToArray();
        double[][] ckv = cond.Select(row => Linear(row, $"{scope}.to_kv", 2 * c)).ToArray();
        cq = HeadNorm(cq, $"{scope}.q_rms_norm.gamma");
        double[][] ck = HeadNorm(ckv.Select(r => r[..c]).ToArray(), $"{scope}.k_rms_norm.gamma");
        double[][] cv = ckv.Select(r => r[c..]).ToArray();
        double[][] cross = Attention(cq, ck, cv, heads, hd);
        for (int t = 0; t < n; t++)
        {
            double[] o = Linear(cross[t], $"{scope}.to_out", c);
            double[]? lifted = proj is null ? null : Linear(proj[t], $"{p}.cross_attn.proj_linear", c);
            for (int i = 0; i < c; i++)
                x[t][i] += o[i] + (lifted?[i] ?? 0.0);
        }

        for (int t = 0; t < n; t++)
        {
            double[] m = Modulate(LayerNorm(x[t], null, null, 1e-6), scaleMlp, shiftMlp);
            double[] hidden = Linear(m, $"{p}.mlp.mlp.0", _config.MlpHidden).Select(Gelu).ToArray();
            double[] o = Linear(hidden, $"{p}.mlp.mlp.2", c);
            for (int i = 0; i < c; i++)
                x[t][i] += o[i] * gateMlp[i];
        }

        return x;
    }

    private double[][] HeadNorm(double[][] rows, string gammaKey)
    {
        int hd = _config.HeadDim;
        float[] gamma = _w[gammaKey];
        return rows.Select(row =>
        {
            double[] o = new double[row.Length];
            for (int head = 0; head < _config.NumHeads; head++)
            {
                double norm = 0;
                for (int d = 0; d < hd; d++)
                    norm += row[head * hd + d] * row[head * hd + d];
                norm = Math.Max(Math.Sqrt(norm), 1e-12);
                for (int d = 0; d < hd; d++)
                    o[head * hd + d] = row[head * hd + d] / norm * gamma[head * hd + d] * Math.Sqrt(hd);
            }

            return o;
        }).ToArray();
    }

    // Pairs (2j, 2j + 1) of each head turn by position[axis] * 10000^(-k/f), axis = j / f, k = j % f; channels past 6f pass through.
    private double[] Rope(double[] row, double[] position)
    {
        int hd = _config.HeadDim, f = hd / 6;
        double[] o = (double[])row.Clone();
        for (int head = 0; head < _config.NumHeads; head++)
        {
            for (int j = 0; j < 3 * f; j++)
            {
                double angle = position[j / f] * Math.Pow(10000.0, -(double)(j % f) / f);
                double cos = Math.Cos(angle), sin = Math.Sin(angle);
                double a = row[head * hd + 2 * j], b = row[head * hd + 2 * j + 1];
                o[head * hd + 2 * j] = a * cos - b * sin;
                o[head * hd + 2 * j + 1] = a * sin + b * cos;
            }
        }

        return o;
    }

    private static double[][] Attention(double[][] q, double[][] k, double[][] v, int heads, int hd)
    {
        double[][] o = q.Select(r => new double[r.Length]).ToArray();
        for (int head = 0; head < heads; head++)
        {
            for (int i = 0; i < q.Length; i++)
            {
                double[] logits = new double[k.Length];
                for (int j = 0; j < k.Length; j++)
                {
                    for (int d = 0; d < hd; d++)
                        logits[j] += q[i][head * hd + d] * k[j][head * hd + d];
                    logits[j] /= Math.Sqrt(hd);
                }

                double max = logits.Max(), sum = 0;
                for (int j = 0; j < logits.Length; j++)
                {
                    logits[j] = Math.Exp(logits[j] - max);
                    sum += logits[j];
                }

                for (int j = 0; j < logits.Length; j++)
                {
                    for (int d = 0; d < hd; d++)
                        o[i][head * hd + d] += logits[j] / sum * v[j][head * hd + d];
                }
            }
        }

        return o;
    }

    private double[] Linear(double[] row, string name, int outDim)
    {
        float[] w = _w[name + ".weight"], b = _w[name + ".bias"];
        double[] o = new double[outDim];
        for (int i = 0; i < outDim; i++)
        {
            double acc = b[i];
            for (int j = 0; j < row.Length; j++)
                acc += w[i * row.Length + j] * row[j];
            o[i] = acc;
        }

        return o;
    }

    private static double[] LayerNorm(double[] row, float[]? weight, float[]? bias, double eps)
    {
        double mean = row.Average(), variance = row.Select(v => (v - mean) * (v - mean)).Average();
        double[] o = new double[row.Length];
        for (int i = 0; i < o.Length; i++)
            o[i] = (row[i] - mean) / Math.Sqrt(variance + eps) * (weight?[i] ?? 1.0) + (bias?[i] ?? 0.0);
        return o;
    }

    private static double[] Modulate(double[] row, double[] scale, double[] shift)
    {
        double[] o = new double[row.Length];
        for (int i = 0; i < o.Length; i++)
            o[i] = row[i] * (1 + scale[i]) + shift[i];
        return o;
    }

    private static double Silu(double x) => x / (1 + Math.Exp(-x));

    private static double Gelu(double x) => 0.5 * x * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}
