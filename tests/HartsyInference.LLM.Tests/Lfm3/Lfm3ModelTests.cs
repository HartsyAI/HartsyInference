using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Models.Lfm3;
using Xunit;

namespace HartsyInference.LLM.Tests.Lfm3;

/// <summary>A tiny random-weight LFM3 against an independent double-precision host reference, and chunked prefill/decode against one shot.</summary>
public sealed unsafe class Lfm3ModelTests : IDisposable
{
    private const string Json = """
        {
          "model_type": "lfm3", "bos_token_id": 1, "eos_token_id": 2, "conv_L_cache": 3, "conv_bias": false,
          "head_dim": 4, "hidden_size": 16, "intermediate_size": 24, "max_position_embeddings": 64,
          "norm_eps": 1e-05, "num_attention_heads": 4, "num_hidden_layers": 4, "num_key_value_heads": 2,
          "layer_types": ["conv", "full_attention", "conv", "sliding_attention"],
          "rope_parameters": {
            "full_attention": { "partial_rotary_factor": 0.5, "rope_theta": 1000.0, "rope_type": "default" },
            "sliding_attention": { "rope_theta": 50.0, "rope_type": "default" }
          },
          "sliding_window": 3, "tie_word_embeddings": true, "use_sandwich_norm": true, "vocab_size": 11
        }
        """;

    private static readonly int[] Prompt = [3, 7, 1, 9, 4, 4, 10, 2, 5];

    private readonly Lfm3Config _config = Lfm3Config.FromJson(Json);
    private readonly Dictionary<string, Tensor> _weights = new();
    private readonly Dictionary<string, float[]> _host = new();
    private readonly CpuBackend _backend = new();
    private uint _state = 0x1F2E3D4Cu;

    public Lfm3ModelTests()
    {
        int h = _config.HiddenSize, hd = _config.HeadDim, inner = _config.IntermediateSize;
        Add("model.embed_tokens.weight", _config.VocabSize, h);
        AddNorm("model.embedding_norm.weight", h);
        for (int i = 0; i < _config.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            foreach (string n in (string[])["operator_norm", "ffn_norm", "post_operator_norm", "post_ffn_norm"])
            {
                AddNorm($"{p}.{n}.weight", h);
            }
            Add($"{p}.feed_forward.w1.weight", inner, h);
            Add($"{p}.feed_forward.w3.weight", inner, h);
            Add($"{p}.feed_forward.w2.weight", h, inner);
            if (_config.Layers[i] == Lfm3LayerKind.Conv)
            {
                Add($"{p}.conv.in_proj.weight", 3 * h, h);
                Add($"{p}.conv.out_proj.weight", h, h);
                Add($"{p}.conv.conv.weight", [h, 1, _config.ConvKernel]);
                continue;
            }
            Add($"{p}.self_attn.q_proj.weight", _config.NumHeads * hd, h);
            Add($"{p}.self_attn.k_proj.weight", _config.NumKvHeads * hd, h);
            Add($"{p}.self_attn.v_proj.weight", _config.NumKvHeads * hd, h);
            Add($"{p}.self_attn.o_proj.weight", h, _config.NumHeads * hd);
            Add($"{p}.self_attn.gate_proj.weight", _config.NumHeads, h);
            AddNorm($"{p}.self_attn.q_norm.weight", hd);
            AddNorm($"{p}.self_attn.k_norm.weight", hd);
        }
    }

    [Fact]
    public void MatchesHostReference()
    {
        float[] expected = Reference(Prompt);
        using Lfm3Model model = new(_weights, _config, 32);
        float[] actual = model.ForwardLastLogits(_backend, Prompt);
        Assert.Equal(expected.Length, actual.Length);
        for (int v = 0; v < expected.Length; v++)
        {
            Assert.InRange(actual[v], expected[v] - 2e-4f, expected[v] + 2e-4f);
        }
    }

    [Fact]
    public void ChunkedPrefillAndDecodeMatchOneShot()
    {
        using Lfm3Model whole = new(_weights, _config, 32);
        float[] expected = whole.ForwardLastLogits(_backend, Prompt);
        using Lfm3Model chunked = new(_weights, _config, 32);
        int[][] chunks = [Prompt[..4], Prompt[4..5], Prompt[5..6], Prompt[6..]];
        float[] actual = [];
        foreach (int[] chunk in chunks)
        {
            actual = chunked.ForwardLastLogits(_backend, chunk);
        }
        Assert.Equal(Prompt.Length, chunked.Position);
        for (int v = 0; v < expected.Length; v++)
        {
            Assert.InRange(actual[v], expected[v] - 1e-4f, expected[v] + 1e-4f);
        }
    }

    [Fact]
    public void ResetStateRestartsTheSequence()
    {
        using Lfm3Model model = new(_weights, _config, 32);
        float[] first = model.ForwardLastLogits(_backend, Prompt);
        model.ResetState();
        Assert.Equal(0, model.Position);
        Assert.Equal(first, model.ForwardLastLogits(_backend, Prompt));
    }

    [Fact]
    public void RefusesAMissingTensorAndAnOverlongSequence()
    {
        Dictionary<string, Tensor> missing = new(_weights);
        missing.Remove("model.layers.1.self_attn.gate_proj.weight");
        Assert.Throws<KeyNotFoundException>(() => new Lfm3Model(missing, _config, 32));
        using Lfm3Model model = new(_weights, _config, 4);
        Assert.Throws<InvalidOperationException>(() => model.ForwardLastLogits(_backend, Prompt));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.ForwardLastLogits(_backend, [11]));
    }

    public void Dispose()
    {
        foreach (Tensor t in _weights.Values)
        {
            t.Dispose();
        }
        _backend.Dispose();
    }

    private float Next() => ((_state = _state * 1664525u + 1013904223u) >> 8) / (float)(1 << 24) - 0.5f;

    private void Add(string key, params int[] dims)
    {
        Tensor t = new(new TensorShape(dims.Select(d => (long)d).ToArray()), DType.F32);
        float[] data = new float[t.ElementCount];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = Next() * 0.8f;
        }
        Store(key, t, data);
    }

    private void AddNorm(string key, int width)
    {
        Tensor t = new(new TensorShape(width), DType.F32);
        float[] data = new float[width];
        for (int i = 0; i < width; i++)
        {
            data[i] = 1f + Next() * 0.4f;
        }
        Store(key, t, data);
    }

    private void Store(string key, Tensor t, float[] data)
    {
        data.AsSpan().CopyTo(new Span<float>((void*)t.DataPointer, data.Length));
        _weights[key] = t;
        _host[key] = data;
    }

    private float[] Reference(int[] ids)
    {
        int n = ids.Length, h = _config.HiddenSize;
        double[][] x = new double[n][];
        float[] embed = _host["model.embed_tokens.weight"];
        for (int t = 0; t < n; t++)
        {
            x[t] = Enumerable.Range(0, h).Select(c => (double)embed[ids[t] * h + c]).ToArray();
        }
        for (int i = 0; i < _config.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            double[][] normed = x.Select(r => Rms(r, _host[$"{p}.operator_norm.weight"])).ToArray();
            double[][] mixed = _config.Layers[i] == Lfm3LayerKind.Conv ? RefConv(normed, p) : RefAttention(normed, p, i);
            for (int t = 0; t < n; t++)
            {
                double[] post = Rms(mixed[t], _host[$"{p}.post_operator_norm.weight"]);
                x[t] = x[t].Zip(post, (a, b) => a + b).ToArray();
            }
            for (int t = 0; t < n; t++)
            {
                double[] f = Rms(x[t], _host[$"{p}.ffn_norm.weight"]);
                double[] g = Mat(f, _host[$"{p}.feed_forward.w1.weight"], _config.IntermediateSize);
                double[] u = Mat(f, _host[$"{p}.feed_forward.w3.weight"], _config.IntermediateSize);
                double[] act = g.Select((v, k) => v / (1 + Math.Exp(-v)) * u[k]).ToArray();
                double[] post = Rms(Mat(act, _host[$"{p}.feed_forward.w2.weight"], h), _host[$"{p}.post_ffn_norm.weight"]);
                x[t] = x[t].Zip(post, (a, b) => a + b).ToArray();
            }
        }
        double[] last = Rms(x[n - 1], _host["model.embedding_norm.weight"]);
        return Mat(last, embed, _config.VocabSize).Select(v => (float)v).ToArray();
    }

    private double[][] RefConv(double[][] normed, string p)
    {
        int n = normed.Length, h = _config.HiddenSize, k = _config.ConvKernel;
        double[][] proj = normed.Select(r => Mat(r, _host[$"{p}.conv.in_proj.weight"], 3 * h)).ToArray();
        double[][] bx = proj.Select(r => Enumerable.Range(0, h).Select(c => r[c] * r[2 * h + c]).ToArray()).ToArray();
        float[] w = _host[$"{p}.conv.conv.weight"];
        double[][] result = new double[n][];
        for (int t = 0; t < n; t++)
        {
            double[] y = new double[h];
            for (int c = 0; c < h; c++)
            {
                double acc = 0;
                for (int tap = 0; tap < k; tap++)
                {
                    int src = t - (k - 1) + tap;
                    acc += src >= 0 ? w[c * k + tap] * bx[src][c] : 0;
                }
                y[c] = acc * proj[t][h + c];
            }
            result[t] = Mat(y, _host[$"{p}.conv.out_proj.weight"], h);
        }
        return result;
    }

    private double[][] RefAttention(double[][] normed, string p, int layer)
    {
        int n = normed.Length, hq = _config.NumHeads, hkv = _config.NumKvHeads, hd = _config.HeadDim, group = hq / hkv;
        string a = $"{p}.self_attn";
        Lfm3RopeParameters rope = _config.RopeOf(layer);
        double[][][] q = new double[n][][], k = new double[n][][], v = new double[n][][];
        for (int t = 0; t < n; t++)
        {
            q[t] = Heads(Mat(normed[t], _host[$"{a}.q_proj.weight"], hq * hd), hq, hd).Select(r => Rope(Rms(r, _host[$"{a}.q_norm.weight"]), t, rope)).ToArray();
            k[t] = Heads(Mat(normed[t], _host[$"{a}.k_proj.weight"], hkv * hd), hkv, hd).Select(r => Rope(Rms(r, _host[$"{a}.k_norm.weight"]), t, rope)).ToArray();
            v[t] = Heads(Mat(normed[t], _host[$"{a}.v_proj.weight"], hkv * hd), hkv, hd);
        }
        bool sliding = _config.Layers[layer] == Lfm3LayerKind.SlidingAttention;
        double[][] result = new double[n][];
        for (int t = 0; t < n; t++)
        {
            double[] gate = Mat(normed[t], _host[$"{a}.gate_proj.weight"], hq);
            double[] flat = new double[hq * hd];
            for (int head = 0; head < hq; head++)
            {
                int lo = sliding ? Math.Max(0, t - _config.SlidingWindow + 1) : 0;
                double[] score = new double[t - lo + 1];
                for (int s = lo; s <= t; s++)
                {
                    score[s - lo] = Enumerable.Range(0, hd).Sum(d => q[t][head][d] * k[s][head / group][d]) / Math.Sqrt(hd);
                }
                double max = score.Max(), sum = score.Sum(sc => Math.Exp(sc - max));
                double sigma = 1 / (1 + Math.Exp(-gate[head]));
                for (int d = 0; d < hd; d++)
                {
                    double o = 0;
                    for (int s = lo; s <= t; s++)
                    {
                        o += Math.Exp(score[s - lo] - max) / sum * v[s][head / group][d];
                    }
                    flat[head * hd + d] = o * sigma;
                }
            }
            result[t] = Mat(flat, _host[$"{a}.o_proj.weight"], _config.HiddenSize);
        }
        return result;
    }

    private double[] Rope(double[] x, int pos, Lfm3RopeParameters rope)
    {
        int hd = x.Length, half = hd / 2, angles = rope.RotatedPairs(hd);
        double[] y = (double[])x.Clone();
        for (int i = 0; i < angles; i++)
        {
            double angle = pos * Math.Pow(rope.Theta, -2.0 * i / hd), c = Math.Cos(angle), s = Math.Sin(angle);
            y[i] = x[i] * c - x[i + half] * s;
            y[i + half] = x[i + half] * c + x[i] * s;
        }
        return y;
    }

    private double[] Rms(double[] x, float[] w)
    {
        double scale = 1 / Math.Sqrt(x.Sum(v => v * v) / x.Length + _config.NormEps);
        return x.Select((v, i) => v * scale * w[i]).ToArray();
    }

    private static double[][] Heads(double[] flat, int heads, int hd) =>
        Enumerable.Range(0, heads).Select(h => flat[(h * hd)..((h + 1) * hd)]).ToArray();

    private static double[] Mat(double[] x, float[] w, int outDim) =>
        Enumerable.Range(0, outDim).Select(o => Enumerable.Range(0, x.Length).Sum(i => w[o * x.Length + i] * x[i])).ToArray();
}
