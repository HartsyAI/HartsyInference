using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Schedulers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Structural smoke tests for the Boogu-Image DiT and scheduler. These run on the CPU backend with a tiny
/// synthetic config and random weights — they prove the whole forward graph (caption/time embed, context/noise/ref
/// refiners, the 3-axis RoPE tables, the dual-stream blocks, the single-stream stack, the output norm and unpatchify,
/// plus the reference-image edit path) executes end to end and produces a finite velocity of the right shape.
/// Numeric parity against the Python reference is a separate, weights-dependent harness (see docs/Research/BOOGU_IMAGE.md §10).</summary>
public sealed class BooguImageTransformerTests
{
    private readonly ITestOutputHelper _output;
    public BooguImageTransformerTests(ITestOutputHelper output) => _output = output;

    private static BooguImageConfig TinyConfig => new()
    {
        HiddenSize = 24,
        NumLayers = 3,
        NumDoubleStreamLayers = 1,
        NumRefinerLayers = 1,
        NumAttentionHeads = 2,
        NumKvHeads = 1,
        PatchSize = 2,
        InChannels = 4,
        MultipleOf = 16,
        AxesDimRope = [4, 4, 4],
        AxesLens = [64, 64, 64],
        RopeTheta = 10000,
        InstructionFeatDim = 8,
        TimestepScale = 1000.0f,
        MaxRefImages = 5,
    };

    [Fact]
    public void Transformer_T2I_Forward_ProducesFiniteVelocity()
    {
        BooguImageConfig cfg = TinyConfig;
        using CpuBackend backend = new();
        using BooguImageTransformer transformer = new(cfg);
        Dictionary<string, Tensor> weights = BuildSyntheticWeights(cfg);
        transformer.LoadWeights(weights);

        int h = 8, w = 8; // latent grid (4 packed tokens)
        using Tensor latent = RandomTensor(new TensorShape(1, cfg.InChannels, h, w), 1234);
        using Tensor instruction = RandomTensor(new TensorShape(1, 5, cfg.InstructionFeatDim), 5678);

        using Tensor velocity = transformer.Forward(backend, latent, 0.3f, instruction);

        Assert.Equal(4, velocity.Shape.Rank);
        Assert.Equal(1, (int)velocity.Shape[0]);
        Assert.Equal(cfg.InChannels, (int)velocity.Shape[1]);
        Assert.Equal(h, (int)velocity.Shape[2]);
        Assert.Equal(w, (int)velocity.Shape[3]);
        AssertAllFinite(velocity);
        DisposeAll(weights);
        _output.WriteLine($"T2I velocity {velocity.Shape} finite.");
    }

    [Fact]
    public void Transformer_Edit_Forward_WithReferenceLatent_ProducesFiniteVelocity()
    {
        BooguImageConfig cfg = TinyConfig;
        using CpuBackend backend = new();
        using BooguImageTransformer transformer = new(cfg);
        Dictionary<string, Tensor> weights = BuildSyntheticWeights(cfg);
        transformer.LoadWeights(weights);

        using Tensor latent = RandomTensor(new TensorShape(1, cfg.InChannels, 8, 8), 11);
        using Tensor instruction = RandomTensor(new TensorShape(1, 4, cfg.InstructionFeatDim), 22);
        using Tensor refLatent = RandomTensor(new TensorShape(1, cfg.InChannels, 6, 6), 33);

        using Tensor velocity = transformer.Forward(backend, latent, 0.5f, instruction, [refLatent]);

        Assert.Equal(cfg.InChannels, (int)velocity.Shape[1]);
        Assert.Equal(8, (int)velocity.Shape[2]);
        Assert.Equal(8, (int)velocity.Shape[3]);
        AssertAllFinite(velocity);
        DisposeAll(weights);
        _output.WriteLine($"Edit velocity {velocity.Shape} finite.");
    }

    [Fact]
    public void Scheduler_V1Shift_IsAscendingAndMatchesLogisticFormula()
    {
        BooguFlowMatchScheduler scheduler = new(seqLen: 4096);
        scheduler.SetTimesteps(8);
        ReadOnlySpan<float> ts = scheduler.Timesteps;

        // mu from the (256 -> 0.5) .. (4096 -> 1.15) line at seq_len 4096 == max_shift 1.15.
        double m = (1.15 - 0.5) / (4096.0 - 256.0);
        double b = 0.5 - m * 256.0;
        double mu = m * 4096 + b;
        double expMu = Math.Exp(mu);

        Assert.Equal(8, ts.Length);
        float prev = -1f;
        for (int i = 0; i < ts.Length; i++)
        {
            double t = (double)i / 8;
            double t1 = Math.Clamp(1.0 - t, 1e-8, 1.0 - 1e-8);
            double expected = 1.0 - expMu / (expMu + (1.0 / t1 - 1.0));
            Assert.True(Math.Abs(ts[i] - expected) < 1e-5, $"step {i}: {ts[i]} vs {expected}");
            Assert.True(ts[i] >= 0f && ts[i] <= 1f);
            Assert.True(ts[i] > prev, "timesteps must ascend (t=0 noise -> t=1 data)");
            prev = ts[i];
        }
    }

    [Fact]
    public void Scheduler_Step_IntegratesForward()
    {
        BooguFlowMatchScheduler scheduler = new(seqLen: 4096);
        scheduler.SetTimesteps(4);
        using Tensor sample = ConstTensor(new TensorShape(1, 2, 2, 2), 1.0f);
        using Tensor velocity = ConstTensor(new TensorShape(1, 2, 2, 2), 2.0f);
        using Tensor outp = new(new TensorShape(1, 2, 2, 2), DType.F32);
        scheduler.Step(outp, velocity, sample, 0);
        AssertAllFinite(outp);
        // out = sample + v * (t_next - t); dt > 0 so output exceeds sample.
        unsafe
        {
            float* o = (float*)outp.DataPointer;
            Assert.True(o[0] > 1.0f, "forward Euler with positive dt must increase the sample");
        }
    }

    // ── synthetic weight construction ──

    private static Dictionary<string, Tensor> BuildSyntheticWeights(BooguImageConfig c)
    {
        Dictionary<string, Tensor> w = new();
        int hidden = c.HiddenSize;
        int qDim = c.NumAttentionHeads * c.HeadDim;
        int kvDim = c.NumKvHeads * c.HeadDim;
        int ffn = c.FfnInnerDim;
        int cond = c.ConditioningDim;
        int patchVol = c.PatchSize * c.PatchSize * c.InChannels;
        int seed = 1;

        void Lin(string k, int outD, int inD) => w[k] = RandomTensor(new TensorShape(outD, inD), seed++, 0.05f);
        void Vec(string k, int d, float center = 1.0f) => w[k] = ConstishTensor(new TensorShape(d), center, seed++);

        // top-level
        Lin("x_embedder.weight", hidden, patchVol); Vec("x_embedder.bias", hidden, 0f);
        Lin("ref_image_patch_embedder.weight", hidden, patchVol); Vec("ref_image_patch_embedder.bias", hidden, 0f);
        Lin("time_caption_embed.timestep_embedder.linear_1.weight", cond, 256); Vec("time_caption_embed.timestep_embedder.linear_1.bias", cond, 0f);
        Lin("time_caption_embed.timestep_embedder.linear_2.weight", cond, cond); Vec("time_caption_embed.timestep_embedder.linear_2.bias", cond, 0f);
        Vec("time_caption_embed.caption_embedder.0.weight", c.InstructionFeatDim);
        Lin("time_caption_embed.caption_embedder.1.weight", hidden, c.InstructionFeatDim); Vec("time_caption_embed.caption_embedder.1.bias", hidden, 0f);
        Lin("norm_out.linear_1.weight", hidden, cond); Vec("norm_out.linear_1.bias", hidden, 0f);
        Lin("norm_out.linear_2.weight", patchVol, hidden); Vec("norm_out.linear_2.bias", patchVol, 0f);
        w["image_index_embedding"] = RandomTensor(new TensorShape(c.MaxRefImages, hidden), seed++, 0.05f);

        void Block(string p, bool mod)
        {
            Lin($"{p}.attn.to_q.weight", qDim, hidden);
            Lin($"{p}.attn.to_k.weight", kvDim, hidden);
            Lin($"{p}.attn.to_v.weight", kvDim, hidden);
            Lin($"{p}.attn.to_out.0.weight", hidden, hidden);
            Vec($"{p}.attn.norm_q.weight", c.HeadDim);
            Vec($"{p}.attn.norm_k.weight", c.HeadDim);
            Lin($"{p}.feed_forward.linear_1.weight", ffn, hidden);
            Lin($"{p}.feed_forward.linear_2.weight", hidden, ffn);
            Lin($"{p}.feed_forward.linear_3.weight", ffn, hidden);
            if (mod)
            {
                Lin($"{p}.norm1.linear.weight", 4 * hidden, cond); Vec($"{p}.norm1.linear.bias", 4 * hidden, 0f);
                Vec($"{p}.norm1.norm.weight", hidden);
            }
            else { Vec($"{p}.norm1.weight", hidden); }
            Vec($"{p}.norm2.weight", hidden);
            Vec($"{p}.ffn_norm1.weight", hidden);
            Vec($"{p}.ffn_norm2.weight", hidden);
        }

        for (int i = 0; i < c.NumRefinerLayers; i++)
        {
            Block($"context_refiner.{i}", mod: false);
            Block($"noise_refiner.{i}", mod: true);
            Block($"ref_image_refiner.{i}", mod: true);
        }
        for (int i = 0; i < c.NumSingleStreamLayers; i++)
            Block($"single_stream_layers.{i}", mod: true);

        for (int i = 0; i < c.NumDoubleStreamLayers; i++)
        {
            string p = $"double_stream_layers.{i}";
            string ji = $"{p}.img_instruct_attn";
            Lin($"{ji}.processor.img_to_q.weight", qDim, hidden);
            Lin($"{ji}.processor.img_to_k.weight", kvDim, hidden);
            Lin($"{ji}.processor.img_to_v.weight", kvDim, hidden);
            Lin($"{ji}.processor.instruct_to_q.weight", qDim, hidden);
            Lin($"{ji}.processor.instruct_to_k.weight", kvDim, hidden);
            Lin($"{ji}.processor.instruct_to_v.weight", kvDim, hidden);
            Lin($"{ji}.processor.img_out.weight", hidden, hidden);
            Lin($"{ji}.processor.instruct_out.weight", hidden, hidden);
            Lin($"{ji}.to_out.0.weight", hidden, hidden);
            Vec($"{ji}.norm_q.weight", c.HeadDim);
            Vec($"{ji}.norm_k.weight", c.HeadDim);

            string sa = $"{p}.img_self_attn";
            Lin($"{sa}.to_q.weight", qDim, hidden);
            Lin($"{sa}.to_k.weight", kvDim, hidden);
            Lin($"{sa}.to_v.weight", kvDim, hidden);
            Lin($"{sa}.to_out.0.weight", hidden, hidden);
            Vec($"{sa}.norm_q.weight", c.HeadDim);
            Vec($"{sa}.norm_k.weight", c.HeadDim);

            Lin($"{p}.img_feed_forward.linear_1.weight", ffn, hidden);
            Lin($"{p}.img_feed_forward.linear_2.weight", hidden, ffn);
            Lin($"{p}.img_feed_forward.linear_3.weight", ffn, hidden);
            Lin($"{p}.instruct_feed_forward.linear_1.weight", ffn, hidden);
            Lin($"{p}.instruct_feed_forward.linear_2.weight", hidden, ffn);
            Lin($"{p}.instruct_feed_forward.linear_3.weight", ffn, hidden);

            foreach (string n in new[] { "img_norm1", "img_norm2", "img_norm3", "instruct_norm1", "instruct_norm2" })
            {
                Lin($"{p}.{n}.linear.weight", 4 * hidden, cond); Vec($"{p}.{n}.linear.bias", 4 * hidden, 0f);
                Vec($"{p}.{n}.norm.weight", hidden);
            }
            foreach (string n in new[] { "img_attn_norm", "img_self_attn_norm", "img_ffn_norm1", "img_ffn_norm2",
                "instruct_attn_norm", "instruct_ffn_norm1", "instruct_ffn_norm2" })
                Vec($"{p}.{n}.weight", hidden);
        }
        return w;
    }

    private static unsafe Tensor RandomTensor(TensorShape shape, int seed, float scale = 0.1f)
    {
        Tensor t = new(shape, DType.F32);
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        long n = shape.ElementCount;
        for (long i = 0; i < n; i++)
            p[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return t;
    }

    private static unsafe Tensor ConstishTensor(TensorShape shape, float center, int seed)
    {
        Tensor t = new(shape, DType.F32);
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        long n = shape.ElementCount;
        for (long i = 0; i < n; i++)
            p[i] = center + (float)((rng.NextDouble() * 2.0 - 1.0) * 0.02);
        return t;
    }

    private static unsafe Tensor ConstTensor(TensorShape shape, float value)
    {
        Tensor t = new(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        long n = shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] = value;
        return t;
    }

    private static unsafe void AssertAllFinite(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount;
        for (long i = 0; i < n; i++)
            Assert.True(float.IsFinite(p[i]), $"non-finite at {i}: {p[i]}");
    }

    private static void DisposeAll(Dictionary<string, Tensor> w)
    {
        foreach (Tensor t in w.Values) t.Dispose();
    }
}
