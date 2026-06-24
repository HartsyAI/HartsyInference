using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Structural smoke test for the Krea 2 DiT. A tiny synthetic config + random weights on the CPU backend
/// proves the whole forward graph (text-fusion stage with the layerwise→projector→refiner path, time embedding +
/// 6-way `scale_shift_table` modulation, sigmoid-gate GQA attention, 3-axis RoPE, final layer, patchify/unpatchify)
/// runs end to end and produces a finite velocity of the right shape. Numeric parity vs diffusers is a separate
/// harness (docs/Research/KREA2.md § 9).</summary>
public sealed class Krea2TransformerTests
{
    private readonly ITestOutputHelper _output;
    public Krea2TransformerTests(ITestOutputHelper output) => _output = output;

    private static Krea2Config TinyConfig => new()
    {
        VaeChannels = 4,
        PatchSize = 2,
        NumLayers = 2,
        AttentionHeadDim = 8,        // sum(axes) must equal this
        NumAttentionHeads = 2,       // hidden = 16
        NumKvHeads = 1,
        IntermediateSize = 32,
        TimestepEmbedDim = 16,
        TextHiddenDim = 12,
        NumTextLayers = 3,
        TextNumHeads = 2,
        TextNumKvHeads = 2,
        TextIntermediateSize = 24,
        NumLayerwiseTextBlocks = 1,
        NumRefinerTextBlocks = 1,
        AxesDimRope = [2, 4, 2],     // sum 8 = head_dim
        RopeTheta = 1000,
        NormEps = 1e-5f,
    };

    [Fact]
    public void Transformer_Forward_ProducesFiniteVelocity()
    {
        Krea2Config cfg = TinyConfig;
        using CpuBackend backend = new();
        using Krea2Transformer transformer = new(cfg);
        Dictionary<string, Tensor> w = BuildWeights(cfg);
        transformer.LoadWeights(w);

        int h = 8, wd = 8; // latent grid → 4×4 = 16 image tokens
        using Tensor latent = Rand(new TensorShape(1, cfg.VaeChannels, h, wd), 1, 0.5f);
        using Tensor encoderHidden = Rand(new TensorShape(1, 5, cfg.NumTextLayers * cfg.TextHiddenDim), 2, 0.2f);

        using Tensor velocity = transformer.Forward(backend, latent, 0.7f, encoderHidden);

        Assert.Equal(4, velocity.Shape.Rank);
        Assert.Equal(cfg.VaeChannels, (int)velocity.Shape[1]);
        Assert.Equal(h, (int)velocity.Shape[2]);
        Assert.Equal(wd, (int)velocity.Shape[3]);
        AssertFinite(velocity);
        DisposeAll(w);
        _output.WriteLine($"Krea 2 velocity {velocity.Shape} finite.");
    }

    private static Dictionary<string, Tensor> BuildWeights(Krea2Config c)
    {
        Dictionary<string, Tensor> w = new();
        int H = c.HiddenSize, td = c.TextHiddenDim, inCh = c.InChannels;
        int qd = c.NumAttentionHeads * c.AttentionHeadDim, kvd = c.NumKvHeads * c.AttentionHeadDim;
        int tqd = c.TextNumHeads * (td / c.TextNumHeads), tkvd = c.TextNumKvHeads * (td / c.TextNumHeads);
        int seed = 1;
        void Lin(string k, int o, int i) => w[k] = Rand(new TensorShape(o, i), seed++, 0.06f);
        void Vec(string k, int d, float center) => w[k] = Const(new TensorShape(d), center, seed++);

        Lin("img_in.weight", H, inCh); Vec("img_in.bias", H, 0f);
        Lin("time_embed.linear_1.weight", H, c.TimestepEmbedDim); Vec("time_embed.linear_1.bias", H, 0f);
        Lin("time_embed.linear_2.weight", H, H); Vec("time_embed.linear_2.bias", H, 0f);
        Lin("time_mod_proj.weight", 6 * H, H); Vec("time_mod_proj.bias", 6 * H, 0f);
        Vec("txt_in.norm.weight", td, 0f);
        Lin("txt_in.linear_1.weight", H, td); Vec("txt_in.linear_1.bias", H, 0f);
        Lin("txt_in.linear_2.weight", H, H); Vec("txt_in.linear_2.bias", H, 0f);
        w["final_layer.scale_shift_table"] = Rand(new TensorShape(2, H), seed++, 0.02f);
        Vec("final_layer.norm.weight", H, 0f);
        Lin("final_layer.linear.weight", inCh, H); Vec("final_layer.linear.bias", inCh, 0f);

        // text fusion
        void FusionBlock(string p) => Block(p, td, td / c.TextNumHeads, tqd, tkvd, c.TextIntermediateSize);
        for (int i = 0; i < c.NumLayerwiseTextBlocks; i++) FusionBlock($"text_fusion.layerwise_blocks.{i}");
        for (int i = 0; i < c.NumRefinerTextBlocks; i++) FusionBlock($"text_fusion.refiner_blocks.{i}");
        Lin("text_fusion.projector.weight", 1, c.NumTextLayers);

        // main blocks (with scale_shift_table)
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"transformer_blocks.{i}";
            w[$"{p}.scale_shift_table"] = Rand(new TensorShape(6, H), seed++, 0.02f);
            Block(p, H, c.AttentionHeadDim, qd, kvd, c.IntermediateSize);
        }
        return w;

        void Block(string p, int dim, int headDim, int qDim, int kvDim, int inner)
        {
            Vec($"{p}.norm1.weight", dim, 0f);
            Vec($"{p}.norm2.weight", dim, 0f);
            Lin($"{p}.attn.to_q.weight", qDim, dim);
            Lin($"{p}.attn.to_k.weight", kvDim, dim);
            Lin($"{p}.attn.to_v.weight", kvDim, dim);
            Lin($"{p}.attn.to_gate.weight", dim, dim);
            Lin($"{p}.attn.to_out.0.weight", dim, dim);
            Vec($"{p}.attn.norm_q.weight", headDim, 0f);
            Vec($"{p}.attn.norm_k.weight", headDim, 0f);
            Lin($"{p}.ff.gate.weight", inner, dim);
            Lin($"{p}.ff.up.weight", inner, dim);
            Lin($"{p}.ff.down.weight", dim, inner);
        }
    }

    private static unsafe Tensor Rand(TensorShape s, int seed, float scale)
    {
        Tensor t = new(s, DType.F32);
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        long n = s.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private static unsafe Tensor Const(TensorShape s, float center, int seed)
    {
        Tensor t = new(s, DType.F32);
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        long n = s.ElementCount;
        for (long i = 0; i < n; i++) p[i] = center + (float)((rng.NextDouble() * 2 - 1) * 0.02);
        return t;
    }

    private static unsafe void AssertFinite(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }

    private static void DisposeAll(Dictionary<string, Tensor> w)
    {
        foreach (Tensor t in w.Values) t.Dispose();
    }
}
