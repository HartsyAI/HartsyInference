using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Models.Spark;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Spark;

/// <summary>A tiny random-weight Spark-X against an independent double-precision host reference, and chunked prefill/decode against one shot.</summary>
public sealed unsafe class SparkModelTests : IDisposable
{
    private const string Json = """
        {
          "model_type": "spark2_5", "bos_token_id": 0, "eos_token_id": 1, "head_dim": 8, "hidden_size": 16, "intermediate_size": 24,
          "hidden_act": "gelu", "gate_attn_act_mode": "sigmoid", "headwise_attn_output_gate": true,
          "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
          "max_position_embeddings": 64, "num_attention_heads": 4, "num_hidden_layers": 4, "num_key_value_heads": 2,
          "rms_norm_eps": 1e-06, "sliding_window": 3, "tie_word_embeddings": true, "vocab_size": 11,
          "rope_parameters": {
            "full_attention": { "partial_rotary_factor": 0.5, "rope_theta": 1000 },
            "sliding_attention": { "partial_rotary_factor": 1.0, "rope_theta": 50 }
          }
        }
        """;

    private static readonly int[] Prompt = [3, 7, 1, 9, 4, 4, 10, 2, 5];

    private readonly SparkConfig _config = SparkConfig.FromJson(Json);
    private readonly Dictionary<string, Tensor> _weights = new();
    private readonly Dictionary<string, float[]> _host = new();
    private readonly CpuBackend _backend = new();
    private readonly ITestOutputHelper _out;
    private uint _state = 0x2A3B4C5Du;

    public SparkModelTests(ITestOutputHelper output)
    {
        _out = output;
        int h = _config.HiddenSize, hd = _config.HeadDim, hq = _config.NumHeads, hkv = _config.NumKvHeads;
        Add("model.embedding.weight", false, _config.VocabSize, h);
        Add("model.norm.weight", true, h);
        for (int i = 0; i < _config.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            Add($"{p}.input_layernorm.weight", true, h);
            Add($"{p}.post_attention_layernorm.weight", true, h);
            Add($"{p}.mlp.gate_proj.weight", false, _config.IntermediateSize, h);
            Add($"{p}.mlp.up_proj.weight", false, _config.IntermediateSize, h);
            Add($"{p}.mlp.down_proj.weight", false, h, _config.IntermediateSize);
            Add($"{p}.self_attn.q_k_v_proj.weight", false, (hq + 2 * hkv) * hd, h);
            Add($"{p}.self_attn.g_proj.weight", false, hq, h);
            Add($"{p}.self_attn.out_proj.weight", false, h, hq * hd);
        }
    }

    [Fact]
    public void MatchesHostReference()
    {
        float[] expected = Reference(Prompt);
        using SparkModel model = new(_weights, _config, 32);
        float[] actual = model.ForwardLastLogits(_backend, Prompt);
        for (int v = 0; v < expected.Length; v++)
        {
            Assert.InRange(actual[v], expected[v] - 2e-4f, expected[v] + 2e-4f);
        }
    }

    [Fact]
    public void ChunkedPrefillAndDecodeMatchOneShot()
    {
        using SparkModel whole = new(_weights, _config, 32);
        float[] expected = whole.ForwardLastLogits(_backend, Prompt);
        using SparkModel chunked = new(_weights, _config, 32);
        float[] actual = [];
        foreach (int[] chunk in new[] { Prompt[..4], Prompt[4..5], Prompt[5..6], Prompt[6..] })
        {
            actual = chunked.ForwardLastLogits(_backend, chunk);
        }
        Assert.Equal(Prompt.Length, chunked.Position);
        Assert.NotEqual(0f, expected.Max(MathF.Abs));
        for (int v = 0; v < expected.Length; v++)
        {
            Assert.InRange(actual[v], expected[v] - 1e-4f, expected[v] + 1e-4f);
        }
    }

    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    [Trait("Category", "GpuIntegration")]
    public void MatchesHostReference_OnDevice(string kind)
    {
        if (!BackendGate.TryOpen(kind, _out.WriteLine, out IBackend? device)) return;
        using IBackend backend = device!;
        if (BackendGate.SkipKnownRtx4090CudaDivergence(kind, backend, _out.WriteLine)) return;
        float[] expected = Reference(Prompt);
        using SparkModel model = new(_weights, _config, 32);
        float[] actual = model.ForwardLastLogits(backend, Prompt);
        BackendGate.LogMaxDifference(expected, actual, _out.WriteLine);
        for (int v = 0; v < expected.Length; v++)
        {
            Assert.InRange(actual[v], expected[v] - 2e-4f, expected[v] + 2e-4f);
        }
    }

    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    [Trait("Category", "GpuIntegration")]
    public void ChunkedPrefillAndDecodeMatchOneShot_OnDevice(string kind)
    {
        if (!BackendGate.TryOpen(kind, _out.WriteLine, out IBackend? device)) return;
        using IBackend backend = device!;
        if (BackendGate.SkipKnownRtx4090CudaDivergence(kind, backend, _out.WriteLine)) return;
        using SparkModel whole = new(_weights, _config, 32);
        float[] expected = whole.ForwardLastLogits(backend, Prompt);
        using SparkModel chunked = new(_weights, _config, 32);
        float[] actual = [];
        foreach (int[] chunk in new[] { Prompt[..4], Prompt[4..5], Prompt[5..6], Prompt[6..] })
        {
            actual = chunked.ForwardLastLogits(backend, chunk);
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
        using SparkModel model = new(_weights, _config, 32);
        float[] first = model.ForwardLastLogits(_backend, Prompt);
        model.ResetState();
        Assert.Equal(0, model.Position);
        Assert.Equal(first, model.ForwardLastLogits(_backend, Prompt));
    }

    [Fact]
    public void RefusesAMissingTensorAndAnOverlongSequence()
    {
        Dictionary<string, Tensor> missing = new(_weights);
        missing.Remove("model.layers.1.self_attn.g_proj.weight");
        Assert.Throws<KeyNotFoundException>(() => new SparkModel(missing, _config, 32));
        using SparkModel model = new(_weights, _config, 4);
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

    private void Add(string key, bool norm, params int[] dims)
    {
        Tensor t = new(new TensorShape(dims.Select(d => (long)d).ToArray()), DType.F32);
        float[] data = new float[t.ElementCount];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = norm ? 1f + Next() * 0.4f : Next() * 0.8f;
        }
        data.AsSpan().CopyTo(new Span<float>((void*)t.DataPointer, data.Length));
        _weights[key] = t;
        _host[key] = data;
    }

    private float[] Reference(int[] ids)
    {
        int n = ids.Length, h = _config.HiddenSize;
        float[] embed = _host["model.embedding.weight"];
        double[][] x = ids.Select(id => Enumerable.Range(0, h).Select(c => (double)embed[id * h + c]).ToArray()).ToArray();
        for (int i = 0; i < _config.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            double[][] normed = x.Select(r => Rms(r, _host[$"{p}.input_layernorm.weight"])).ToArray();
            double[][] attended = RefAttention(normed, p, i);
            for (int t = 0; t < n; t++)
            {
                x[t] = x[t].Zip(attended[t], (a, b) => a + b).ToArray();
            }
            for (int t = 0; t < n; t++)
            {
                double[] f = Rms(x[t], _host[$"{p}.post_attention_layernorm.weight"]);
                double[] g = Mat(f, _host[$"{p}.mlp.gate_proj.weight"], _config.IntermediateSize);
                double[] u = Mat(f, _host[$"{p}.mlp.up_proj.weight"], _config.IntermediateSize);
                double[] act = g.Select((v, k) => 0.5 * v * (1 + Erf(v / Math.Sqrt(2))) * u[k]).ToArray();
                double[] down = Mat(act, _host[$"{p}.mlp.down_proj.weight"], h);
                x[t] = x[t].Zip(down, (a, b) => a + b).ToArray();
            }
        }
        double[] last = Rms(x[n - 1], _host["model.norm.weight"]);
        return Mat(last, embed, _config.VocabSize).Select(v => (float)v).ToArray();
    }

    private double[][] RefAttention(double[][] normed, string p, int layer)
    {
        int n = normed.Length, hq = _config.NumHeads, hkv = _config.NumKvHeads, hd = _config.HeadDim, group = hq / hkv;
        string a = $"{p}.self_attn";
        LayerRope rope = new(_config.RopeOf(layer).Theta, _config.RopeOf(layer).RotaryDim(hd));
        double[][][] q = new double[n][][], k = new double[n][][], v = new double[n][][];
        for (int t = 0; t < n; t++)
        {
            double[] qkv = Mat(normed[t], _host[$"{a}.q_k_v_proj.weight"], (hq + 2 * hkv) * hd);
            q[t] = Heads(qkv[..(hq * hd)], hq, hd).Select(r => Rope(r, t, rope)).ToArray();
            k[t] = Heads(qkv[(hq * hd)..((hq + hkv) * hd)], hkv, hd).Select(r => Rope(r, t, rope)).ToArray();
            v[t] = Heads(qkv[((hq + hkv) * hd)..], hkv, hd);
        }
        bool sliding = _config.SlidingLayers[layer];
        double[][] result = new double[n][];
        for (int t = 0; t < n; t++)
        {
            double[] gate = Mat(normed[t], _host[$"{a}.g_proj.weight"], hq);
            double[] flat = new double[hq * hd];
            for (int head = 0; head < hq; head++)
            {
                int lo = sliding ? Math.Max(0, t - _config.SlidingWindow + 1) : 0;
                double[] score = new double[t - lo + 1];
                for (int s = lo; s <= t; s++)
                {
                    score[s - lo] = Enumerable.Range(0, hd).Sum(d => q[t][head][d] * k[s][head / group][d]) / Math.Sqrt(hd);
                }
                double max = score.Max(), sum = score.Sum(sc => Math.Exp(sc - max)), sigma = 1 / (1 + Math.Exp(-gate[head]));
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
            result[t] = Mat(flat, _host[$"{a}.out_proj.weight"], _config.HiddenSize);
        }
        return result;
    }

    private readonly record struct LayerRope(float Theta, int RotaryDim);

    private static double[] Rope(double[] x, int pos, LayerRope rope)
    {
        int half = rope.RotaryDim / 2;
        double[] y = (double[])x.Clone();
        for (int i = 0; i < half; i++)
        {
            double angle = pos * Math.Pow(rope.Theta, -2.0 * i / rope.RotaryDim), c = Math.Cos(angle), s = Math.Sin(angle);
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

    private static double Erf(double x)
    {
        double t = 1 / (1 + 0.3275911 * Math.Abs(x));
        double y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.Exp(-x * x);
        return x >= 0 ? y : -y;
    }

    private static double[][] Heads(double[] flat, int heads, int hd) =>
        Enumerable.Range(0, heads).Select(h => flat[(h * hd)..((h + 1) * hd)]).ToArray();

    private static double[] Mat(double[] x, float[] w, int outDim) =>
        Enumerable.Range(0, outDim).Select(o => Enumerable.Range(0, x.Length).Sum(i => w[o * x.Length + i] * x[i])).ToArray();
}
