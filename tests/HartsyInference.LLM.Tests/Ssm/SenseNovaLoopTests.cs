using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Ssm;
using HartsyInference.LLM.Transformer.Mhc;
using Xunit;
using static HartsyInference.LLM.Tests.Ssm.SenseNovaLoopReference;

namespace HartsyInference.LLM.Tests.Ssm;

/// <summary>SenseNova-Loop on the Qwen3.5 trunk: the execution plan, the hyper-connection router, and a tiny looped model (a repeated
/// physical layer with its own state per visit, two-stream mHC around it, YaRN RoPE) against the float64 transcription of the
/// reference. Synthetic weights; <c>SyntheticSmoke</c> in the sense that no checkpoint is involved.</summary>
public sealed class SenseNovaLoopTests
{
    // The real checkpoint's loop and mHC settings (config.json @ 3cc220e), reduced to what the reader needs.
    private const string RealConfig = """
        { "text_config": { "num_hidden_layers": 32 },
          "loop_settings": { "counts": [2], "end": 28, "eval_count": 2, "start": 4 },
          "mhc": { "end": 28, "fixed_identity": false, "iterations": 20, "pre_scale": 1.0, "start": 4, "streams": 2 } }
        """;

    private static string TinyConfig(int streams) => $$"""
        { "loop_settings": { "counts": [2], "end": 3, "eval_count": 2, "start": 1 },
          "mhc": { "end": 3, "fixed_identity": false, "iterations": 20, "pre_scale": 1.0, "start": 1, "streams": {{streams}} },
          "text_config": {
            "hidden_size": 32, "num_hidden_layers": 4, "vocab_size": 40, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "layer_types": ["linear_attention", "linear_attention", "full_attention", "full_attention"],
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 8, "linear_value_head_dim": 8, "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "rms_norm_eps": 1e-6, "max_position_embeddings": 64, "tie_word_embeddings": true,
            "rope_parameters": { "rope_theta": 10000.0, "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0], "mrope_interleaved": true,
              "rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 16 } } }
        """;

    [Fact]
    public void Plan_OfTheRealCheckpoint_RunsTheMiddleTwice()
    {
        SenseNovaLoopConfig loop = SenseNovaLoopConfig.FromJson(RealConfig)!;
        Assert.Equal(56, loop.Slots);
        Assert.Equal(Enumerable.Range(0, 4), loop.Plan.Take(4));
        Assert.Equal(Enumerable.Range(4, 24), loop.Plan.Skip(4).Take(24));
        Assert.Equal(Enumerable.Range(4, 24), loop.Plan.Skip(28).Take(24));
        Assert.Equal(Enumerable.Range(28, 4), loop.Plan.Skip(52));
        Assert.Equal(52, loop.FirstSlot(28));   // the GGUF stores layer 28 at slot 52
        Assert.Equal(4, loop.FirstSlot(4));
        Assert.True(loop.EntersStreams(4) && loop.EntersStreams(28));
        Assert.True(loop.LeavesStreams(27) && loop.LeavesStreams(51));
        Assert.False(loop.IsMhcSlot(3) || loop.IsMhcSlot(52));
        Assert.Null(SenseNovaLoopConfig.FromJson("""{ "text_config": { "num_hidden_layers": 32 } }"""));
    }

    [Theory]
    [InlineData("""{ "text_config": { "num_hidden_layers": 32 }, "loop_settings": { "counts": [2, 3], "end": 28, "eval_count": 2, "start": 4 }, "mhc": { "end": 28, "iterations": 20, "start": 4, "streams": 2 } }""")]
    [InlineData("""{ "text_config": { "num_hidden_layers": 32 }, "loop_settings": { "counts": [2], "end": 28, "eval_count": 2, "start": 4, "use_gate": true }, "mhc": { "end": 28, "iterations": 20, "start": 4, "streams": 2 } }""")]
    [InlineData("""{ "text_config": { "num_hidden_layers": 32 }, "loop_settings": { "counts": [2], "end": 28, "eval_count": 2, "start": 4 }, "mhc": { "end": 30, "iterations": 20, "start": 4, "streams": 2 } }""")]
    [InlineData("""{ "text_config": { "num_hidden_layers": 32 }, "loop_settings": { "counts": [2], "end": 28, "eval_count": 2, "start": 4 }, "mhc": { "end": 28, "iterations": 20, "start": 2, "streams": 2 } }""")]
    [InlineData("""{ "text_config": { "num_hidden_layers": 32 }, "loop_settings": { "counts": [2], "end": 28, "eval_count": 2, "start": 4 }, "mhc": { "end": 28, "iterations": 20, "start": 4, "streams": 3 } }""")]
    public void Plan_RejectsWhatTheReferenceRejects(string json) => Assert.Throws<InvalidDataException>(() => SenseNovaLoopConfig.FromJson(json));

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void Router_MatchesTheTranscription(int streams)
    {
        // Four streams make the mix a 4x4 matrix whose transpose differs from itself, which a two-stream mix hides.
        const int dim = 16, tokens = 3;
        SenseNovaLoopConfig loop = new(1, 3, 2, 4, 1, 3, streams, 20);
        Random rng = new(5);
        int mix = 2 * streams + streams * streams;
        float[] fn = Gaussian(rng, mix * streams * dim, 0.3f), bias = Gaussian(rng, mix, 0.7f), x = Gaussian(rng, tokens * streams * dim, 1f);
        double[] wantBranch = RouteReference(x, fn, bias, tokens, streams, dim);

        using CpuBackend backend = new();
        HyperConnection hc = new(backend, loop.HyperConnection(dim, 1e-6f), fn, [1f, 1f, 1f], bias);
        float[] pre = new float[tokens * streams], post = new float[tokens * streams], comb = new float[tokens * streams * streams];
        hc.Mixes(x, tokens, pre, post, comb);
        float[] branch = new float[tokens * dim];
        hc.Collapse(x, pre, tokens, branch);
        Assert.True(RelRms(wantBranch, branch) < 1e-6, $"collapsed input relRMS {RelRms(wantBranch, branch)}");

        float[] sub = Gaussian(rng, tokens * dim, 1f), got = new float[tokens * streams * dim];
        hc.Expand(sub, x, post, comb, tokens, got);
        double[] expanded = MergeReference(x, sub, fn, bias, tokens, streams, dim);
        Assert.True(RelRms(expanded, got) < 1e-6, $"merged streams relRMS {RelRms(expanded, got)}");
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void TinyLoopedModel_MatchesTheTranscription(int streams)
    {
        (Dictionary<string, float[]> raw, Dictionary<string, Tensor> weights) = TinyWeights(11, streams);
        int[] ids = [3, 17, 5, 29, 8, 1, 22];
        double[][] want = new SenseNovaLoopReference(raw) { Streams = streams }.Forward(ids);

        using CpuBackend backend = new();
        SenseNovaLoopConfig loop = SenseNovaLoopConfig.FromJson(TinyConfig(streams))!;
        Qwen35HfConfig cfg = Qwen35HfConfig.FromJson(TinyConfig(streams));
        using Qwen35Model model = Qwen35Model.FromHuggingFace(weights, cfg, maxSequenceLength: 32, loop: loop);
        Assert.Equal(6, model.NumLayers);
        Assert.False(model.GraphDecodeReady);
        float[] got = model.ForwardHiddenStates(backend, ids);

        double[] flat = want.SelectMany(r => r).ToArray();
        double rel = RelRms(flat, got);
        Assert.True(rel < 1e-5, $"hidden states relRMS {rel}");
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void TinyLoopedModel_StepwiseDecodeEqualsOneShot(int streams)
    {
        (_, Dictionary<string, Tensor> weights) = TinyWeights(23, streams);
        int[] ids = [4, 9, 31, 12, 7, 20];
        SenseNovaLoopConfig loop = SenseNovaLoopConfig.FromJson(TinyConfig(streams))!;
        Qwen35HfConfig cfg = Qwen35HfConfig.FromJson(TinyConfig(streams));
        using CpuBackend backend = new();
        using Qwen35Model model = Qwen35Model.FromHuggingFace(weights, cfg, maxSequenceLength: 32, loop: loop);

        for (int n = 4; n <= ids.Length; n++)
        {
            model.ResetState();
            float[] stepped = model.ForwardLastLogits(backend, ids.Take(3).ToArray());
            for (int k = 3; k < n; k++) stepped = model.ForwardLastLogits(backend, [ids[k]]);
            model.ResetState();
            float[] oneShot = model.ForwardLastLogits(backend, ids.Take(n).ToArray());
            double rel = RelRms(oneShot.Select(v => (double)v).ToArray(), stepped);
            Assert.True(rel < 1e-5, $"{n} tokens: stepwise vs one-shot logits relRMS {rel}");
        }
    }

    [Theory]
    [InlineData(2)]
    public void TinyLoopedModel_VisitsKeepSeparateState(int streams)
    {
        // A second visit that shared the first visit's cache would change the answer; the transcription recomputes every slot from the
        // whole sequence, so agreement above already implies separate state. Here: one more token must not alter earlier positions.
        (_, Dictionary<string, Tensor> weights) = TinyWeights(31, streams);
        SenseNovaLoopConfig loop = SenseNovaLoopConfig.FromJson(TinyConfig(streams))!;
        Qwen35HfConfig cfg = Qwen35HfConfig.FromJson(TinyConfig(streams));
        using CpuBackend backend = new();
        using Qwen35Model model = Qwen35Model.FromHuggingFace(weights, cfg, maxSequenceLength: 32, loop: loop);
        float[] shortRun = model.ForwardHiddenStates(backend, [6, 14, 2, 33]);
        float[] longRun = model.ForwardHiddenStates(backend, [6, 14, 2, 33, 18]);
        Assert.True(RelRms(shortRun.Select(v => (double)v).ToArray(), longRun.Take(shortRun.Length).ToArray()) < 1e-5);
    }

    // ---- synthetic checkpoint ----------------------------------------------------------------------------------------------

    private static (Dictionary<string, float[]> Raw, Dictionary<string, Tensor> Weights) TinyWeights(int seed, int streams)
    {
        Random rng = new(seed);
        Dictionary<string, float[]> w = new();
        const string root = "model.language_model";
        w[$"{root}.embed_tokens.weight"] = Gaussian(rng, Vocab * Hidden, 0.5f);
        w[$"{root}.norm.weight"] = Gaussian(rng, Hidden, 0.2f);
        int mix = 2 * streams + streams * streams;
        for (int l = 0; l < Linear.Length; l++)
        {
            string p = $"{root}.layers.{l}";
            w[$"{p}.input_layernorm.weight"] = Gaussian(rng, Hidden, 0.2f);
            w[$"{p}.post_attention_layernorm.weight"] = Gaussian(rng, Hidden, 0.2f);
            w[$"{p}.mlp.gate_proj.weight"] = Gaussian(rng, Ffn * Hidden, 0.3f);
            w[$"{p}.mlp.up_proj.weight"] = Gaussian(rng, Ffn * Hidden, 0.3f);
            w[$"{p}.mlp.down_proj.weight"] = Gaussian(rng, Hidden * Ffn, 0.3f);
            if (Linear[l])
            {
                int convDim = 2 * KeyHeads * KeyDim + ValueHeads * ValueDim;
                w[$"{p}.linear_attn.in_proj_qkv.weight"] = Gaussian(rng, convDim * Hidden, 0.3f);
                w[$"{p}.linear_attn.in_proj_z.weight"] = Gaussian(rng, ValueHeads * ValueDim * Hidden, 0.3f);
                w[$"{p}.linear_attn.in_proj_a.weight"] = Gaussian(rng, ValueHeads * Hidden, 0.3f);
                w[$"{p}.linear_attn.in_proj_b.weight"] = Gaussian(rng, ValueHeads * Hidden, 0.3f);
                w[$"{p}.linear_attn.conv1d.weight"] = Gaussian(rng, convDim * ConvKernel, 0.4f);
                w[$"{p}.linear_attn.A_log"] = Enumerable.Range(0, ValueHeads).Select(_ => (float)Math.Log(0.5 + 1.5 * rng.NextDouble())).ToArray();
                w[$"{p}.linear_attn.dt_bias"] = Gaussian(rng, ValueHeads, 0.5f);
                w[$"{p}.linear_attn.norm.weight"] = Enumerable.Range(0, ValueDim).Select(_ => 0.8f + 0.4f * (float)rng.NextDouble()).ToArray();
                w[$"{p}.linear_attn.out_proj.weight"] = Gaussian(rng, Hidden * ValueHeads * ValueDim, 0.3f);
            }
            else
            {
                w[$"{p}.self_attn.q_proj.weight"] = Gaussian(rng, Heads * HeadDim * 2 * Hidden, 0.3f);
                w[$"{p}.self_attn.k_proj.weight"] = Gaussian(rng, KvHeads * HeadDim * Hidden, 0.3f);
                w[$"{p}.self_attn.v_proj.weight"] = Gaussian(rng, KvHeads * HeadDim * Hidden, 0.3f);
                w[$"{p}.self_attn.o_proj.weight"] = Gaussian(rng, Hidden * Heads * HeadDim, 0.3f);
                w[$"{p}.self_attn.q_norm.weight"] = Gaussian(rng, HeadDim, 0.2f);
                w[$"{p}.self_attn.k_norm.weight"] = Gaussian(rng, HeadDim, 0.2f);
            }
            if (l is >= 1 and < 3)
            {
                // The shipped routers are zero-initialised; random ones exercise the projection, the gates and Sinkhorn.
                foreach (string router in (string[])["mhc_attn", "mhc_ffn"])
                {
                    w[$"{p}.{router}.weight"] = Gaussian(rng, mix * streams * Hidden, 0.08f);
                    w[$"{p}.{router}.bias"] = Gaussian(rng, mix, 0.6f);
                }
            }
        }
        Dictionary<string, Tensor> tensors = new();
        foreach ((string name, float[] values) in w)
        {
            long[] shape = name.EndsWith("conv1d.weight") ? [values.Length / ConvKernel, 1, ConvKernel] : [values.Length];
            Tensor t = new(new TensorShape(shape), DType.F32);
            values.CopyTo(t.AsSpan<float>());
            tensors[name] = t;
        }
        return (w, ReshapeMatrices(tensors, w, streams));
    }

    /// <summary>Gives the 2-D weights their <c>[out, in]</c> shape (the trunk reads dimensions from the tensors).</summary>
    private static Dictionary<string, Tensor> ReshapeMatrices(Dictionary<string, Tensor> flat, Dictionary<string, float[]> raw, int streams)
    {
        Dictionary<string, Tensor> shaped = new();
        foreach ((string name, Tensor t) in flat)
        {
            bool matrix = name.EndsWith(".weight") && !name.Contains("norm") && !name.EndsWith("conv1d.weight");
            if (name.EndsWith("embed_tokens.weight")) shaped[name] = t.Reshape(new TensorShape(Vocab, Hidden));
            else if (name.Contains("mhc_") && name.EndsWith(".weight")) shaped[name] = t.Reshape(new TensorShape(2 * streams + streams * streams, streams * Hidden));
            else if (matrix) shaped[name] = t.Reshape(new TensorShape(Rows(name), raw[name].Length / Rows(name)));
            else shaped[name] = t;
        }
        return shaped;
    }

    private static int Rows(string name) => name switch
    {
        _ when name.EndsWith("gate_proj.weight") || name.EndsWith("up_proj.weight") => Ffn,
        _ when name.EndsWith("down_proj.weight") || name.EndsWith("o_proj.weight") || name.EndsWith("out_proj.weight") => Hidden,
        _ when name.EndsWith("in_proj_qkv.weight") => 2 * KeyHeads * KeyDim + ValueHeads * ValueDim,
        _ when name.EndsWith("in_proj_z.weight") => ValueHeads * ValueDim,
        _ when name.EndsWith("in_proj_a.weight") || name.EndsWith("in_proj_b.weight") => ValueHeads,
        _ when name.EndsWith("q_proj.weight") => Heads * HeadDim * 2,
        _ when name.EndsWith("k_proj.weight") || name.EndsWith("v_proj.weight") => KvHeads * HeadDim,
        _ => throw new ArgumentException($"No row count for {name}."),
    };

    private static float[] Gaussian(Random rng, int count, float std)
    {
        float[] v = new float[count];
        for (int i = 0; i < count; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            v[i] = (float)(std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        return v;
    }

    private static double RelRms(double[] want, float[] got)
    {
        double num = 0, den = 0;
        for (int i = 0; i < want.Length; i++)
        {
            double d = want[i] - got[i];
            num += d * d;
            den += want[i] * want[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    // ---- router transcription (MHCRouter.forward and .merge, float64) -------------------------------------------------------

    private static (double[] Pre, double[] Post, double[,] Mix) Gates(double[] flat, float[] fn, float[] bias, int streams)
    {
        int mix = 2 * streams + streams * streams;
        double scale = 1.0 / Math.Sqrt(flat.Sum(e => e * e) / flat.Length + 1e-6);
        double[] logits = new double[mix];
        for (int o = 0; o < mix; o++)
        {
            double acc = bias[o];
            for (int i = 0; i < flat.Length; i++) acc += fn[o * flat.Length + i] * flat[i] * scale;
            logits[o] = acc;
        }
        double[] pre = Enumerable.Range(0, streams).Select(s => 1.0 / (1.0 + Math.Exp(-logits[s]))).ToArray();
        double[] post = Enumerable.Range(0, streams).Select(s => 2.0 / (1.0 + Math.Exp(-logits[streams + s]))).ToArray();
        double[,] m = new double[streams, streams];
        for (int i = 0; i < streams; i++)
            for (int j = 0; j < streams; j++) m[i, j] = logits[2 * streams + i * streams + j];
        for (int it = 0; it < 20; it++)
        {
            for (int i = 0; i < streams; i++)
            {
                double lse = Math.Log(Enumerable.Range(0, streams).Sum(j => Math.Exp(m[i, j])));
                for (int j = 0; j < streams; j++) m[i, j] -= lse;
            }
            for (int j = 0; j < streams; j++)
            {
                double lse = Math.Log(Enumerable.Range(0, streams).Sum(i => Math.Exp(m[i, j])));
                for (int i = 0; i < streams; i++) m[i, j] -= lse;
            }
        }
        for (int i = 0; i < streams; i++)
            for (int j = 0; j < streams; j++) m[i, j] = Math.Exp(m[i, j]);
        return (pre, post, m);
    }

    private static double[] RouteReference(float[] x, float[] fn, float[] bias, int tokens, int streams, int dim)
    {
        double[] branch = new double[tokens * dim];
        for (int t = 0; t < tokens; t++)
        {
            double[] flat = Enumerable.Range(0, streams * dim).Select(i => (double)x[t * streams * dim + i]).ToArray();
            (double[] pre, _, _) = Gates(flat, fn, bias, streams);
            for (int c = 0; c < dim; c++)
                branch[t * dim + c] = Enumerable.Range(0, streams).Sum(s => pre[s] * flat[s * dim + c]);
        }
        return branch;
    }

    private static double[] MergeReference(float[] x, float[] sub, float[] fn, float[] bias, int tokens, int streams, int dim)
    {
        double[] result = new double[tokens * streams * dim];
        for (int t = 0; t < tokens; t++)
        {
            double[] flat = Enumerable.Range(0, streams * dim).Select(i => (double)x[t * streams * dim + i]).ToArray();
            (_, double[] post, double[,] mix) = Gates(flat, fn, bias, streams);
            for (int i = 0; i < streams; i++)
                for (int c = 0; c < dim; c++)
                {
                    double acc = 0;
                    for (int j = 0; j < streams; j++) acc += mix[i, j] * flat[j * dim + c];
                    result[(t * streams + i) * dim + c] = acc + post[i] * sub[t * dim + c];
                }
        }
        return result;
    }
}
