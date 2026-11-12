using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Sam3dObjects;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Sam3dObjects;

public sealed class SamMotFlowTests
{
    // A tiny network with the released structure: an ungrouped "shape" grid (2^3 tokens) that is protected, and a group "rot" of two one-token latents.
    private static readonly SamMotFlowConfig Config = new(
        ModelChannels: 8,
        Heads: 2,
        NumBlocks: 2,
        CondChannels: 6,
        MlpRatio: 2.0,
        Latents: [new SamMotLatent("rot", 2), new SamMotLatent("shape", 3), new SamMotLatent("trans", 3)],
        Groups: new Dictionary<string, IReadOnlyList<string>> { ["rot"] = ["rot", "trans"] },
        Protected: ["shape"],
        IsShortcut: true);

    private static readonly Dictionary<string, int> Tokens = new() { ["rot"] = 1, ["shape"] = 8, ["trans"] = 1, ["a"] = 1, ["b"] = 2, ["c"] = 3 };

    private sealed class Owned(params IEnumerable<Tensor>[] sets) : IDisposable
    {
        public void Dispose()
        {
            foreach (IEnumerable<Tensor> set in sets)
            {
                foreach (Tensor t in set) t.Dispose();
            }
        }
    }

    private static Tensor Filled(Random rng, double scale, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>())
            v = (float)((rng.NextDouble() - 0.5) * scale);
        return t;
    }

    private static Dictionary<string, Tensor> Weights(Random rng, SamMotFlowConfig config)
    {
        int c = config.ModelChannels, heads = config.Heads, hd = config.HeadDim, hidden = config.MlpHidden, cond = config.CondChannels;
        Dictionary<string, Tensor> w = [];
        void Timestep(string p)
        {
            w[p + "mlp.0.weight"] = Filled(rng, 1.0, c, 256);
            w[p + "mlp.0.bias"] = Filled(rng, 0.2, c);
            w[p + "mlp.2.weight"] = Filled(rng, 1.0, c, c);
            w[p + "mlp.2.bias"] = Filled(rng, 0.2, c);
        }

        Timestep("t_embedder.");
        if (config.IsShortcut)
            Timestep("d_embedder.");
        foreach (SamMotLatent latent in config.Latents)
        {
            string p = $"latent_mapping.{latent.Name}.";
            w[p + "input_layer.weight"] = Filled(rng, 1.0, c, latent.Channels);
            w[p + "input_layer.bias"] = Filled(rng, 0.2, c);
            w[p + "out_layer.weight"] = Filled(rng, 1.0, latent.Channels, c);
            w[p + "out_layer.bias"] = Filled(rng, 0.2, latent.Channels);
            w[p + "pos_emb"] = Filled(rng, 1.0, Tokens[latent.Name], c);
        }

        for (int i = 0; i < config.NumBlocks; i++)
        {
            string b = $"blocks.{i}.";
            w[b + "adaLN_modulation.1.weight"] = Filled(rng, 0.6, 6 * c, c);
            w[b + "adaLN_modulation.1.bias"] = Filled(rng, 0.4, 6 * c);
            foreach (string s in config.StreamNames)
            {
                w[$"{b}norm2.{s}.weight"] = Filled(rng, 1.0, c);
                w[$"{b}norm2.{s}.bias"] = Filled(rng, 0.4, c);
                w[$"{b}self_attn.to_qkv.{s}.weight"] = Filled(rng, 1.0, 3 * c, c);
                w[$"{b}self_attn.to_qkv.{s}.bias"] = Filled(rng, 0.2, 3 * c);
                w[$"{b}self_attn.q_rms_norm.{s}.gamma"] = Filled(rng, 1.0, heads, hd);
                w[$"{b}self_attn.k_rms_norm.{s}.gamma"] = Filled(rng, 1.0, heads, hd);
                w[$"{b}self_attn.to_out.{s}.weight"] = Filled(rng, 1.0, c, c);
                w[$"{b}self_attn.to_out.{s}.bias"] = Filled(rng, 0.2, c);
                w[$"{b}cross_attn.{s}.to_q.weight"] = Filled(rng, 1.0, c, c);
                w[$"{b}cross_attn.{s}.to_q.bias"] = Filled(rng, 0.2, c);
                w[$"{b}cross_attn.{s}.to_kv.weight"] = Filled(rng, 1.0, 2 * c, cond);
                w[$"{b}cross_attn.{s}.to_kv.bias"] = Filled(rng, 0.2, 2 * c);
                w[$"{b}cross_attn.{s}.to_out.weight"] = Filled(rng, 1.0, c, c);
                w[$"{b}cross_attn.{s}.to_out.bias"] = Filled(rng, 0.2, c);
                w[$"{b}mlp.{s}.mlp.0.weight"] = Filled(rng, 1.0, hidden, c);
                w[$"{b}mlp.{s}.mlp.0.bias"] = Filled(rng, 0.2, hidden);
                w[$"{b}mlp.{s}.mlp.2.weight"] = Filled(rng, 1.0, c, hidden);
                w[$"{b}mlp.{s}.mlp.2.bias"] = Filled(rng, 0.2, c);
            }
        }

        return w;
    }

    private static Dictionary<string, Tensor> Latents(Random rng, SamMotFlowConfig config)
    {
        Dictionary<string, Tensor> latents = [];
        foreach (SamMotLatent latent in config.Latents)
            latents[latent.Name] = Filled(rng, 2.0, 1, Tokens[latent.Name], latent.Channels);
        return latents;
    }

    // ---- Independent reference of the upstream forward, in doubles, written from the Python source rather than from the C# layout. ----

    private static float[] V(Dictionary<string, Tensor> w, string key) => w[key].AsReadOnlySpan<float>().ToArray();

    private static double[][] Rows(Tensor t)
    {
        int n = (int)t.Shape[1], d = (int)t.Shape[2];
        ReadOnlySpan<float> data = t.AsReadOnlySpan<float>();
        double[][] rows = new double[n][];
        for (int i = 0; i < n; i++)
        {
            rows[i] = new double[d];
            for (int j = 0; j < d; j++) rows[i][j] = data[i * d + j];
        }

        return rows;
    }

    private static double[][] Linear(double[][] x, float[] w, float[] b)
    {
        int outDim = b.Length, inDim = w.Length / outDim;
        double[][] y = new double[x.Length][];
        for (int i = 0; i < x.Length; i++)
        {
            y[i] = new double[outDim];
            for (int o = 0; o < outDim; o++)
            {
                double s = b[o];
                for (int k = 0; k < inDim; k++) s += w[o * inDim + k] * x[i][k];
                y[i][o] = s;
            }
        }

        return y;
    }

    private static double[][] LayerNorm(double[][] x, double eps, float[]? w = null, float[]? b = null)
    {
        double[][] y = new double[x.Length][];
        for (int i = 0; i < x.Length; i++)
        {
            int d = x[i].Length;
            double mean = x[i].Average(), var = x[i].Sum(v => (v - mean) * (v - mean)) / d;
            y[i] = new double[d];
            for (int j = 0; j < d; j++)
                y[i][j] = (x[i][j] - mean) / Math.Sqrt(var + eps) * (w?[j] ?? 1) + (b?[j] ?? 0);
        }

        return y;
    }

    private static double[][] Map(double[][] x, Func<double, double> f) => x.Select(r => r.Select(f).ToArray()).ToArray();

    private static double[][] Zip(double[][] a, double[][] b, Func<double, double, double> f)
        => a.Select((r, i) => r.Select((v, j) => f(v, b[i][j])).ToArray()).ToArray();

    private static double[][] Cat(IEnumerable<double[][]> parts) => parts.SelectMany(p => p).ToArray();

    private static double[][] Cols(double[][] x, int start, int count) => x.Select(r => r[start..(start + count)]).ToArray();

    private static double[][] RmsHeads(double[][] x, float[] gamma, int heads)
    {
        int hd = x[0].Length / heads;
        return x.Select(r =>
        {
            double[] y = new double[r.Length];
            for (int h = 0; h < heads; h++)
            {
                double rms = Math.Sqrt(r.Skip(h * hd).Take(hd).Sum(v => v * v) / hd);
                for (int d = 0; d < hd; d++) y[h * hd + d] = r[h * hd + d] / rms * gamma[h * hd + d];
            }

            return y;
        }).ToArray();
    }

    private static double[][] Attention(double[][] q, double[][] k, double[][] v, int heads)
    {
        int hd = q[0].Length / heads;
        double[][] o = q.Select(r => new double[r.Length]).ToArray();
        for (int h = 0; h < heads; h++)
        {
            for (int i = 0; i < q.Length; i++)
            {
                double[] scores = new double[k.Length];
                for (int j = 0; j < k.Length; j++)
                {
                    double dot = 0;
                    for (int d = 0; d < hd; d++) dot += q[i][h * hd + d] * k[j][h * hd + d];
                    scores[j] = dot / Math.Sqrt(hd);
                }

                double max = scores.Max(), sum = scores.Sum(s => Math.Exp(s - max));
                for (int j = 0; j < k.Length; j++)
                {
                    double p = Math.Exp(scores[j] - max) / sum;
                    for (int d = 0; d < hd; d++) o[i][h * hd + d] += p * v[j][h * hd + d];
                }
            }
        }

        return o;
    }

    private static double[] Sinusoid(double t, int dim = 256)
    {
        double[] e = new double[dim];
        for (int i = 0; i < dim / 2; i++)
        {
            double angle = t * Math.Exp(-Math.Log(10000) * i / (dim / 2));
            e[i] = Math.Cos(angle);
            e[dim / 2 + i] = Math.Sin(angle);
        }

        return e;
    }

    private static double[] TimestepMlp(Dictionary<string, Tensor> w, string p, double t)
    {
        double[][] h = Linear([Sinusoid(t)], V(w, p + "mlp.0.weight"), V(w, p + "mlp.0.bias"));
        h = Map(h, v => v / (1 + Math.Exp(-v)));
        return Linear(h, V(w, p + "mlp.2.weight"), V(w, p + "mlp.2.bias"))[0];
    }

    private static double[][] Modulate(double[][] x, double[] scale, double[] shift)
        => LayerNorm(x, 1e-6).Select(r => r.Select((val, j) => val * (1 + scale[j]) + shift[j]).ToArray()).ToArray();

    private static double[][] GatedAdd(double[][] x, double[][] value, double[] gate)
        => x.Select((r, i) => r.Select((val, j) => val + gate[j] * value[i][j]).ToArray()).ToArray();

    /// <summary>The velocities by latent name.</summary>
    private static Dictionary<string, double[][]> Reference(SamMotFlowConfig config, Dictionary<string, Tensor> w, Dictionary<string, Tensor> latents, double t, Tensor cond, double? d)
    {
        int c = config.ModelChannels, heads = config.Heads;
        double[] emb = TimestepMlp(w, "t_embedder.", t);
        if (d is not null)
        {
            double[] dEmb = TimestepMlp(w, "d_embedder.", d.Value);
            for (int i = 0; i < c; i++) emb[i] += dEmb[i];
        }

        double[][] act = [emb.Select(v => v / (1 + Math.Exp(-v))).ToArray()];
        Dictionary<string, double[][]> x = [];
        foreach (SamMotLatent latent in config.Latents)
        {
            string p = $"latent_mapping.{latent.Name}.";
            double[][] projected = Linear(Rows(latents[latent.Name]), V(w, p + "input_layer.weight"), V(w, p + "input_layer.bias"));
            float[] pos = V(w, p + "pos_emb");
            x[latent.Name] = projected.Select((r, i) => r.Select((v, j) => v + pos[i * c + j]).ToArray()).ToArray();
        }

        Dictionary<string, double[][]> streams = [];
        foreach (string s in config.StreamNames)
            streams[s] = config.Groups.TryGetValue(s, out IReadOnlyList<string>? members) ? Cat(members.Select(m => x[m])) : x[s];

        double[][] condRows = Rows(cond);
        for (int i = 0; i < config.NumBlocks; i++)
        {
            string b = $"blocks.{i}.";
            double[] mod = Linear(act, V(w, b + "adaLN_modulation.1.weight"), V(w, b + "adaLN_modulation.1.bias"))[0];
            double[] Chunk(int n) => mod[(n * c)..((n + 1) * c)];
            double[] shiftMsa = Chunk(0), scaleMsa = Chunk(1), gateMsa = Chunk(2), shiftMlp = Chunk(3), scaleMlp = Chunk(4), gateMlp = Chunk(5);

            Dictionary<string, double[][]> q = [], k = [], v = [];
            foreach (string s in config.StreamNames)
            {
                double[][] qkv = Linear(Modulate(streams[s], scaleMsa, shiftMsa), V(w, $"{b}self_attn.to_qkv.{s}.weight"), V(w, $"{b}self_attn.to_qkv.{s}.bias"));
                q[s] = RmsHeads(Cols(qkv, 0, c), V(w, $"{b}self_attn.q_rms_norm.{s}.gamma"), heads);
                k[s] = RmsHeads(Cols(qkv, c, c), V(w, $"{b}self_attn.k_rms_norm.{s}.gamma"), heads);
                v[s] = Cols(qkv, 2 * c, c);
            }

            // A protected stream attends only to itself; every other stream attends to the other streams' tokens and the protected ones.
            Dictionary<string, double[][]> attended = [];
            foreach (string s in config.Protected)
                attended[s] = Attention(q[s], k[s], v[s], heads);
            string[] others = config.StreamNames.Where(s => !config.Protected.Contains(s)).ToArray();
            double[][] joint = Attention(
                Cat(others.Select(s => q[s])),
                Cat(others.Concat(config.Protected).Select(s => k[s])),
                Cat(others.Concat(config.Protected).Select(s => v[s])),
                heads);
            int start = 0;
            foreach (string s in others)
            {
                attended[s] = joint[start..(start + q[s].Length)];
                start += q[s].Length;
            }

            foreach (string s in config.StreamNames)
            {
                double[][] self = Linear(attended[s], V(w, $"{b}self_attn.to_out.{s}.weight"), V(w, $"{b}self_attn.to_out.{s}.bias"));
                double[][] x1 = GatedAdd(streams[s], self, gateMsa);

                double[][] normed = LayerNorm(x1, 1e-6, V(w, $"{b}norm2.{s}.weight"), V(w, $"{b}norm2.{s}.bias"));
                double[][] cq = Linear(normed, V(w, $"{b}cross_attn.{s}.to_q.weight"), V(w, $"{b}cross_attn.{s}.to_q.bias"));
                double[][] kv = Linear(condRows, V(w, $"{b}cross_attn.{s}.to_kv.weight"), V(w, $"{b}cross_attn.{s}.to_kv.bias"));
                double[][] cross = Linear(Attention(cq, Cols(kv, 0, c), Cols(kv, c, c), heads), V(w, $"{b}cross_attn.{s}.to_out.weight"), V(w, $"{b}cross_attn.{s}.to_out.bias"));
                double[][] x2 = Zip(x1, cross, (a, bb) => a + bb);

                double[][] up = Linear(Modulate(x2, scaleMlp, shiftMlp), V(w, $"{b}mlp.{s}.mlp.0.weight"), V(w, $"{b}mlp.{s}.mlp.0.bias"));
                double[][] gelu = Map(up, u => 0.5 * u * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (u + 0.044715 * u * u * u))));
                double[][] down = Linear(gelu, V(w, $"{b}mlp.{s}.mlp.2.weight"), V(w, $"{b}mlp.{s}.mlp.2.bias"));
                streams[s] = GatedAdd(x2, down, gateMlp);
            }
        }

        Dictionary<string, double[][]> result = [];
        foreach (SamMotLatent latent in config.Latents)
        {
            string stream = config.StreamOf(latent.Name);
            int offset = 0;
            if (config.Groups.TryGetValue(stream, out IReadOnlyList<string>? members))
            {
                foreach (string m in members.TakeWhile(m => m != latent.Name))
                    offset += Tokens[m];
            }

            double[][] hidden = streams[stream][offset..(offset + Tokens[latent.Name])];
            string p = $"latent_mapping.{latent.Name}.";
            result[latent.Name] = Linear(LayerNorm(hidden, 1e-5), V(w, p + "out_layer.weight"), V(w, p + "out_layer.bias"));
        }

        return result;
    }

    private static void AssertClose(double[][] expected, Tensor actual, double tolerance = 2e-3)
    {
        ReadOnlySpan<float> got = actual.AsReadOnlySpan<float>();
        int d = expected[0].Length;
        Assert.Equal(expected.Length * d, got.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            for (int j = 0; j < d; j++)
                Assert.True(Math.Abs(expected[i][j] - got[i * d + j]) < tolerance * (1 + Math.Abs(expected[i][j])), $"[{i},{j}] expected {expected[i][j]} got {got[i * d + j]}");
        }
    }

    [Fact]
    public void Forward_MatchesTheNaiveReferenceOfTheUpstreamForward()
    {
        Random rng = new(11);
        Dictionary<string, Tensor> w = Weights(rng, Config);
        Dictionary<string, Tensor> latents = Latents(rng, Config);
        Tensor cond = Filled(rng, 2.0, 1, 5, Config.CondChannels);
        using Owned owned = new(w.Values, latents.Values, [cond]);
        using CpuBackend backend = new();
        SamMotFlow flow = new(Config, w);

        Dictionary<string, Tensor> velocity = flow.Forward(backend, latents, 321f, cond, 0f);
        using Owned results = new(velocity.Values);

        Dictionary<string, double[][]> expected = Reference(Config, w, latents, 321.0, cond, 0.0);
        foreach (SamMotLatent latent in Config.Latents)
        {
            Assert.Equal(new TensorShape(1, Tokens[latent.Name], latent.Channels), velocity[latent.Name].Shape);
            AssertClose(expected[latent.Name], velocity[latent.Name]);
        }
    }

    [Fact]
    public void Forward_MatchesTheReferenceWithSeveralUnprotectedStreamsAndMultiTokenMembers()
    {
        // Streams: shape (protected), c (ungrouped, 3 tokens) and ga (a + b, 3 tokens), so the joint attention concatenates and splits more than one other stream.
        SamMotFlowConfig wide = new(
            ModelChannels: 8,
            Heads: 2,
            NumBlocks: 2,
            CondChannels: 6,
            MlpRatio: 2.0,
            Latents: [new SamMotLatent("a", 2), new SamMotLatent("shape", 3), new SamMotLatent("c", 2), new SamMotLatent("b", 4)],
            Groups: new Dictionary<string, IReadOnlyList<string>> { ["ga"] = ["a", "b"] },
            Protected: ["shape"],
            IsShortcut: false);
        Assert.Equal(["shape", "c", "ga"], wide.StreamNames);
        Random rng = new(19);
        Dictionary<string, Tensor> w = Weights(rng, wide);
        Dictionary<string, Tensor> latents = Latents(rng, wide);
        Tensor cond = Filled(rng, 2.0, 1, 4, wide.CondChannels);
        using Owned owned = new(w.Values, latents.Values, [cond]);
        using CpuBackend backend = new();
        SamMotFlow flow = new(wide, w);

        Dictionary<string, Tensor> velocity = flow.Forward(backend, latents, 640f, cond);
        using Owned results = new(velocity.Values);

        Dictionary<string, double[][]> expected = Reference(wide, w, latents, 640.0, cond, null);
        foreach (SamMotLatent latent in wide.Latents)
            AssertClose(expected[latent.Name], velocity[latent.Name]);
    }

    [Fact]
    public void Forward_StepSizeChangesTheOutputAndMatchesTheReferenceWhenNonZero()
    {
        Random rng = new(12);
        Dictionary<string, Tensor> w = Weights(rng, Config);
        Dictionary<string, Tensor> latents = Latents(rng, Config);
        Tensor cond = Filled(rng, 2.0, 1, 3, Config.CondChannels);
        using Owned owned = new(w.Values, latents.Values, [cond]);
        using CpuBackend backend = new();
        SamMotFlow flow = new(Config, w);

        Dictionary<string, Tensor> atZero = flow.Forward(backend, latents, 500f, cond, 0f);
        Dictionary<string, Tensor> shortStep = flow.Forward(backend, latents, 500f, cond, 0.25f);
        using Owned results = new(atZero.Values, shortStep.Values);

        Dictionary<string, double[][]> expected = Reference(Config, w, latents, 500.0, cond, 0.25);
        foreach (SamMotLatent latent in Config.Latents)
            AssertClose(expected[latent.Name], shortStep[latent.Name]);
        Assert.NotEqual(atZero["shape"].AsReadOnlySpan<float>().ToArray(), shortStep["shape"].AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void Forward_AnOmittedStepSizeMeansZeroOnAShortcutModel()
    {
        Random rng = new(18);
        Dictionary<string, Tensor> w = Weights(rng, Config);
        Dictionary<string, Tensor> latents = Latents(rng, Config);
        Tensor cond = Filled(rng, 2.0, 1, 3, Config.CondChannels);
        using Owned owned = new(w.Values, latents.Values, [cond]);
        using CpuBackend backend = new();
        SamMotFlow flow = new(Config, w);

        Dictionary<string, Tensor> omitted = flow.Forward(backend, latents, 250f, cond);
        Dictionary<string, Tensor> zero = flow.Forward(backend, latents, 250f, cond, 0f);
        using Owned results = new(omitted.Values, zero.Values);

        Assert.Equal(zero["shape"].AsReadOnlySpan<float>().ToArray(), omitted["shape"].AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void Forward_TheProtectedShapeStreamIgnoresThePoseLatentsButNotTheOtherWayAround()
    {
        Random rng = new(13);
        Dictionary<string, Tensor> w = Weights(rng, Config);
        Dictionary<string, Tensor> latents = Latents(rng, Config);
        Dictionary<string, Tensor> changedPose = new(latents) { ["rot"] = Filled(rng, 2.0, 1, 1, 2), ["trans"] = Filled(rng, 2.0, 1, 1, 3) };
        Dictionary<string, Tensor> changedShape = new(latents) { ["shape"] = Filled(rng, 2.0, 1, 8, 3) };
        Tensor cond = Filled(rng, 2.0, 1, 4, Config.CondChannels);
        using Owned owned = new(w.Values, latents.Values, [cond, changedPose["rot"], changedPose["trans"], changedShape["shape"]]);
        using CpuBackend backend = new();
        SamMotFlow flow = new(Config, w);

        Dictionary<string, Tensor> baseline = flow.Forward(backend, latents, 400f, cond, 0f);
        Dictionary<string, Tensor> poseMoved = flow.Forward(backend, changedPose, 400f, cond, 0f);
        Dictionary<string, Tensor> shapeMoved = flow.Forward(backend, changedShape, 400f, cond, 0f);
        using Owned results = new(baseline.Values, poseMoved.Values, shapeMoved.Values);

        Assert.Equal(baseline["shape"].AsReadOnlySpan<float>().ToArray(), poseMoved["shape"].AsReadOnlySpan<float>().ToArray());
        Assert.NotEqual(baseline["rot"].AsReadOnlySpan<float>().ToArray(), shapeMoved["rot"].AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void Forward_AZeroConditionDiffersFromARealOneAndStaysFinite()
    {
        Random rng = new(14);
        Dictionary<string, Tensor> w = Weights(rng, Config);
        Dictionary<string, Tensor> latents = Latents(rng, Config);
        Tensor cond = Filled(rng, 2.0, 1, 4, Config.CondChannels);
        Tensor zero = new(cond.Shape, DType.F32);
        using Owned owned = new(w.Values, latents.Values, [cond, zero]);
        using CpuBackend backend = new();
        SamMotFlow flow = new(Config, w);

        Dictionary<string, Tensor> real = flow.Forward(backend, latents, 100f, cond, 0f);
        Dictionary<string, Tensor> unconditional = flow.Forward(backend, latents, 100f, zero, 0f);
        using Owned results = new(real.Values, unconditional.Values);

        Assert.All(unconditional["shape"].AsReadOnlySpan<float>().ToArray(), v => Assert.True(float.IsFinite(v)));
        Assert.NotEqual(real["shape"].AsReadOnlySpan<float>().ToArray(), unconditional["shape"].AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void Forward_RejectsMissingLatentsBadConditionsAndAStepSizeOnANonShortcutModel()
    {
        Random rng = new(15);
        Dictionary<string, Tensor> w = Weights(rng, Config);
        Dictionary<string, Tensor> latents = Latents(rng, Config);
        Tensor cond = Filled(rng, 2.0, 1, 4, Config.CondChannels);
        Tensor wrongWidth = Filled(rng, 2.0, 1, 4, Config.CondChannels + 1);
        using Owned owned = new(w.Values, latents.Values, [cond, wrongWidth]);
        using CpuBackend backend = new();
        SamMotFlow flow = new(Config, w);

        Dictionary<string, Tensor> missing = new(latents);
        missing.Remove("trans");
        Assert.Throws<ArgumentException>(() => flow.Forward(backend, missing, 1f, cond, 0f));
        Assert.Throws<ArgumentException>(() => flow.Forward(backend, latents, 1f, wrongWidth, 0f));
        using Tensor noTokens = new(new TensorShape(1, 0, Config.CondChannels), DType.F32);
        Assert.Throws<ArgumentException>(() => flow.Forward(backend, latents, 1f, noTokens, 0f));

        SamMotFlowConfig plain = Config with { IsShortcut = false };
        SamMotFlow plainFlow = new(plain, w);
        Assert.Throws<ArgumentException>(() => plainFlow.Forward(backend, latents, 1f, cond, 0f));
    }

    [Fact]
    public void Constructor_RejectsMissingMisshapenAndNonF32Weights()
    {
        Random rng = new(16);
        Dictionary<string, Tensor> w = Weights(rng, Config);
        using Owned owned = new(w.Values);

        Tensor removed = w["blocks.1.mlp.shape.mlp.2.bias"];
        w.Remove("blocks.1.mlp.shape.mlp.2.bias");
        InvalidOperationException missing = Assert.Throws<InvalidOperationException>(() => new SamMotFlow(Config, w));
        Assert.Contains("blocks.1.mlp.shape.mlp.2.bias", missing.Message);
        w["blocks.1.mlp.shape.mlp.2.bias"] = removed;

        Tensor kept = w["latent_mapping.rot.out_layer.weight"];
        using Tensor wrongShape = Filled(rng, 1.0, 3, Config.ModelChannels);
        w["latent_mapping.rot.out_layer.weight"] = wrongShape;
        Assert.Throws<InvalidOperationException>(() => new SamMotFlow(Config, w));
        w["latent_mapping.rot.out_layer.weight"] = kept;

        Tensor f16Source = w["t_embedder.mlp.2.bias"];
        using Tensor f16 = f16Source.CastTo(DType.F16);
        w["t_embedder.mlp.2.bias"] = f16;
        Assert.Throws<InvalidOperationException>(() => new SamMotFlow(Config, w));
        w["t_embedder.mlp.2.bias"] = f16Source;
    }

    [Fact]
    public void Constructor_FallsBackToSuppliedPositionEmbeddingsWhenPosEmbIsMissing()
    {
        Random rng = new(17);
        Dictionary<string, Tensor> w = Weights(rng, Config);
        Tensor grid = w["latent_mapping.shape.pos_emb"];
        w.Remove("latent_mapping.shape.pos_emb");
        using Owned owned = new(w.Values, [grid]);

        Assert.Throws<InvalidOperationException>(() => new SamMotFlow(Config, w));
        SamMotFlow flow = new(Config, w, new Dictionary<string, Tensor> { ["shape"] = grid });
        Assert.Equal(8, flow.TokensOf("shape"));
    }

    [Fact]
    public void Select_RebasesAFlowNestedUnderWrappersAndRejectsAmbiguity()
    {
        using Tensor a = new(new TensorShape(1), DType.F32), b = new(new TensorShape(1), DType.F32), other = new(new TensorShape(1), DType.F32);
        Dictionary<string, Tensor> nested = new()
        {
            ["reverse_fn.backbone.t_embedder.mlp.0.weight"] = a,
            ["reverse_fn.backbone.blocks.0.x"] = b,
            ["unrelated.weight"] = other,
        };

        Dictionary<string, Tensor> flat = SamMotFlow.Select(nested);

        Assert.Equal(["blocks.0.x", "t_embedder.mlp.0.weight"], flat.Keys.OrderBy(k => k).ToArray());
        Assert.Same(a, flat["t_embedder.mlp.0.weight"]);
        Assert.Single(SamMotFlow.Select(new Dictionary<string, Tensor> { ["t_embedder.mlp.0.weight"] = a }));
        Assert.Throws<InvalidOperationException>(() => SamMotFlow.Select(new Dictionary<string, Tensor> { ["x.weight"] = a }));
        Assert.Throws<InvalidOperationException>(() => SamMotFlow.Select(new Dictionary<string, Tensor>
        {
            ["a.t_embedder.mlp.0.weight"] = a,
            ["b.t_embedder.mlp.0.weight"] = b,
        }));
        // A key that merely ends in the anchor text without a separating dot is not the anchor.
        Assert.Throws<InvalidOperationException>(() => SamMotFlow.Select(new Dictionary<string, Tensor> { ["xt_embedder.mlp.0.weight"] = a }));
    }

    [Fact]
    public void Config_ReleasedHasTheShippedStreamsAndTokenLayout()
    {
        SamMotFlowConfig released = SamMotFlowConfig.Released;

        released.Validate();
        Assert.Equal(["shape", "6drotation_normalized"], released.StreamNames);
        Assert.Equal("6drotation_normalized", released.StreamOf("translation_scale"));
        Assert.Equal("shape", released.StreamOf("shape"));
        Assert.Equal(64, released.HeadDim);
        Assert.Equal(4096, released.MlpHidden);
    }

    [Fact]
    public void Config_ValidateRejectsInconsistentGroupsAndProtectedNames()
    {
        Assert.Throws<ArgumentException>(() => (Config with { Groups = new Dictionary<string, IReadOnlyList<string>> { ["rot"] = ["rot", "nope"] } }).Validate());
        Assert.Throws<ArgumentException>(() => (Config with { Groups = new Dictionary<string, IReadOnlyList<string>> { ["rot"] = ["rot", "trans"], ["g2"] = ["trans"] } }).Validate());
        Assert.Throws<ArgumentException>(() => (Config with { Protected = ["missing"] }).Validate());
        Assert.Throws<ArgumentException>(() => (Config with { Heads = 3 }).Validate());
        Assert.Throws<ArgumentException>(() => (Config with { Protected = ["shape", "shape"] }).Validate());
    }

    [Fact]
    public void Narrow_ConvertsOnlyTheBlockLinearsAndLeavesEverythingElseTheSameObject()
    {
        Random rng = new(31);
        Dictionary<string, Tensor> w = Weights(rng, Config);
        using Owned owned = new(w.Values);
        List<Tensor> converted = [];

        Dictionary<string, Tensor> narrowed = SamFlowWeights.Narrow(w, DType.F16, converted);
        using Owned convertedOwned = new(converted);

        Assert.NotEmpty(converted);
        foreach ((string key, Tensor tensor) in w)
        {
            if (SamFlowWeights.IsBlockLinear(key, tensor))
            {
                Assert.Equal(DType.F16, narrowed[key].DType);
                Assert.Contains(narrowed[key], converted);
            }
            else
            {
                Assert.Same(tensor, narrowed[key]);
            }
        }

        List<Tensor> none = [];
        Dictionary<string, Tensor> same = SamFlowWeights.Narrow(w, DType.F32, none);
        Assert.Empty(none);
        Assert.All(w, entry => Assert.Same(entry.Value, same[entry.Key]));
        Assert.Throws<ArgumentOutOfRangeException>(() => SamFlowWeights.Narrow(w, DType.I32, []));
    }

    [Fact]
    public void Forward_WithF16BlockLinearsStaysCloseToTheF32Forward()
    {
        Random rng = new(32);
        Dictionary<string, Tensor> w = Weights(rng, Config);
        Dictionary<string, Tensor> latents = Latents(rng, Config);
        Tensor cond = Filled(rng, 2.0, 1, 5, Config.CondChannels);
        List<Tensor> converted = [];
        Dictionary<string, Tensor> narrowed = SamFlowWeights.Narrow(w, DType.F16, converted);
        using Owned owned = new(w.Values, latents.Values, [cond], converted);
        using CpuBackend backend = new();

        Dictionary<string, Tensor> exact = new SamMotFlow(Config, w).Forward(backend, latents, 321f, cond, 0f);
        Dictionary<string, Tensor> half = new SamMotFlow(Config, narrowed).Forward(backend, latents, 321f, cond, 0f);
        using Owned results = new(exact.Values, half.Values);

        double difference = 0, magnitude = 0;
        foreach (SamMotLatent latent in Config.Latents)
        {
            ReadOnlySpan<float> a = exact[latent.Name].AsReadOnlySpan<float>(), b = half[latent.Name].AsReadOnlySpan<float>();
            for (int i = 0; i < a.Length; i++)
            {
                difference += (a[i] - b[i]) * (double)(a[i] - b[i]);
                magnitude += a[i] * (double)a[i];
            }
        }

        double relative = Math.Sqrt(difference / magnitude);
        Assert.True(relative > 0, "F16 weights must change the result at least a little; identical output means the narrowing did nothing.");
        Assert.True(relative < 5e-3, $"relative L2 {relative}");
    }

    /// <summary>An <see cref="IBackend"/> that records <c>Sync</c> and <c>FreeWeights</c> and, at each, whether a watched tensor is still alive; every other call does nothing.</summary>
    public class RecordingBackend : DispatchProxy
    {
        public List<string> Events { get; } = [];

        public Tensor? Watched { get; set; }

        protected override object? Invoke(MethodInfo? method, object?[]? args)
        {
            if (method?.Name is "Sync" or "FreeWeights")
                Events.Add($"{method.Name}:{(Watched is not null && IsDisposed(Watched) ? "disposed" : "alive")}");
            return method?.ReturnType.IsValueType == true && method.ReturnType != typeof(void) ? Activator.CreateInstance(method.ReturnType) : null;
        }

        public bool IsWatchedDisposed => Watched is not null && IsDisposed(Watched);

        private static bool IsDisposed(Tensor tensor)
        {
            try
            {
                _ = tensor.AsReadOnlySpan<byte>().Length;
                return false;
            }
            catch (ObjectDisposedException)
            {
                return true;
            }
        }
    }

    [Fact]
    public void Release_DrainsFreesAndSweepsBeforeTheConvertedTensorsAreDisposed()
    {
        IBackend proxy = DispatchProxy.Create<IBackend, RecordingBackend>();
        RecordingBackend recording = (RecordingBackend)proxy;
        Tensor converted = new(new TensorShape(4, 4), DType.F16);
        recording.Watched = converted;
        List<Tensor> owned = [converted];

        SamFlowWeights.Release(proxy, [converted], owned, () => recording.Events.Add($"Sweep:{(recording.IsWatchedDisposed ? "disposed" : "alive")}"));

        // Sync, the free and the sweep all run while the converted tensor is alive; only afterwards is it disposed.
        Assert.Equal(["Sync:alive", "FreeWeights:alive", "Sweep:alive"], recording.Events);
        Assert.Throws<ObjectDisposedException>(() => { _ = converted.AsReadOnlySpan<byte>().Length; });
        Assert.Empty(owned);
    }

    [Fact]
    public void Release_DisposesTheConvertedTensorsEvenWhenTheSweepThrows()
    {
        IBackend proxy = DispatchProxy.Create<IBackend, RecordingBackend>();
        Tensor converted = new(new TensorShape(4, 4), DType.F16);
        List<Tensor> owned = [converted];

        Assert.Throws<InvalidOperationException>(() => SamFlowWeights.Release(proxy, [converted], owned, () => throw new InvalidOperationException("sweep")));

        Assert.Throws<ObjectDisposedException>(() => { _ = converted.AsReadOnlySpan<byte>().Length; });
        Assert.Empty(owned);
    }
}
