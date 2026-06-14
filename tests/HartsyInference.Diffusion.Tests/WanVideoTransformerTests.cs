using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tests for the Wan-Video DiT — the per-head interleaved 3D <see cref="WanRope"/> (rotation invariants) and a tiny-config <see cref="WanVideoTransformer"/> end-to-end on CPU. Numerics vs the real checkpoint are validation-pending.</summary>
public unsafe class WanVideoTransformerTests
{
    [Fact]
    public void Rope_PreservesPerHeadNorm()
    {
        WanRope rope = new(headDim: 12, theta: 10000, maxSeqLen: 64);
        (Tensor cos, Tensor sin) = rope.BuildCosSin(2, 2, 2);   // S=8
        Assert.Equal(8, (int)cos.Shape[0]);
        Assert.Equal(12, (int)cos.Shape[1]);

        int heads = 2, hd = 12, dim = heads * hd;
        Tensor x = RandRows(8, dim, seed: 3);
        double before = L2(x);
        rope.ApplyRotary(x, cos, sin, heads);   // [S, heads, hd] contiguous as [S, dim]
        Assert.True(Math.Abs(before - L2(x)) < 1e-3, "rotation must preserve norm");
    }

    [Fact]
    public void Rope_IsPositionSensitive()
    {
        WanRope rope = new(headDim: 12, theta: 10000, maxSeqLen: 64);
        (Tensor cos, Tensor sin) = rope.BuildCosSin(2, 1, 1);   // 2 frames, S=2
        int heads = 1, hd = 12;
        Tensor x = RandRows(2, heads * hd, seed: 7);
        Buffer.MemoryCopy((float*)x.DataPointer, (float*)x.DataPointer + hd, hd * 4, hd * 4);   // identical rows
        rope.ApplyRotary(x, cos, sin, heads);
        float* p = (float*)x.DataPointer;
        float maxDiff = 0;
        for (int d = 0; d < hd; d++) maxDiff = MathF.Max(maxDiff, MathF.Abs(p[d] - p[hd + d]));
        Assert.True(maxDiff > 0.05f, $"different frames should rotate differently: {maxDiff}");
    }

    [Fact]
    public void Transformer_TinyConfig_ProducesLatentShape()
    {
        CpuBackend backend = new();
        WanVideoConfig cfg = new()
        {
            NumHeads = 2, HeadDim = 12, InChannels = 8, OutChannels = 8,
            TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 2, PatchSize = (1, 2, 2),
        };
        WanVideoTransformer transformer = new(cfg);
        transformer.LoadWeights(BuildWan(cfg));

        Tensor latent = RandRows5d(1, cfg.InChannels, 2, 4, 4, seed: 11);   // [1,8,2,4,4]
        Tensor encoder = RandRows(3, cfg.TextDim, seed: 12);

        Tensor outVel = transformer.Forward(backend, latent, encoder, timestep: 0.5f);

        Assert.Equal(5, outVel.Shape.Rank);
        Assert.Equal(cfg.OutChannels, (int)outVel.Shape[1]);
        Assert.Equal(2, (int)outVel.Shape[2]);
        Assert.Equal(4, (int)outVel.Shape[3]);
        Assert.Equal(4, (int)outVel.Shape[4]);
        float* p = (float*)outVel.DataPointer;
        for (long i = 0; i < outVel.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }

    [Fact]
    public void Transformer_PerFrameTimesteps_UniformMatchesScalar()
    {
        CpuBackend backend = new();
        WanVideoConfig cfg = new()
        {
            NumHeads = 2, HeadDim = 12, InChannels = 8, OutChannels = 8,
            TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 2, PatchSize = (1, 2, 2),
        };
        WanVideoTransformer transformer = new(cfg);
        transformer.LoadWeights(BuildWan(cfg));

        Tensor latent = RandRows5d(1, cfg.InChannels, 2, 4, 4, seed: 21);
        Tensor encoder = RandRows(3, cfg.TextDim, seed: 22);

        Tensor scalar = transformer.Forward(backend, latent, encoder, timestep: 0.5f);
        Tensor uniform = transformer.Forward(backend, latent, encoder, [0.5f, 0.5f]);   // one per latent frame

        float* a = (float*)scalar.DataPointer;
        float* b = (float*)uniform.DataPointer;
        for (long i = 0; i < scalar.Shape.ElementCount; i++)
            Assert.True(a[i] == b[i], $"uniform per-frame timesteps must match the scalar path exactly (idx {i}: {a[i]} vs {b[i]})");
    }

    [Fact]
    public void Transformer_PerFrameTimesteps_ConditionedFrameDiffers()
    {
        CpuBackend backend = new();
        WanVideoConfig cfg = new()
        {
            NumHeads = 2, HeadDim = 12, InChannels = 8, OutChannels = 8,
            TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 2, PatchSize = (1, 2, 2),
        };
        WanVideoTransformer transformer = new(cfg);
        transformer.LoadWeights(BuildWan(cfg));

        Tensor latent = RandRows5d(1, cfg.InChannels, 2, 4, 4, seed: 31);
        Tensor encoder = RandRows(3, cfg.TextDim, seed: 32);

        Tensor scalar = transformer.Forward(backend, latent, encoder, timestep: 0.5f);
        Tensor i2v = transformer.Forward(backend, latent, encoder, [0f, 0.5f]);   // TI2V: frame 0 conditioned

        float* a = (float*)scalar.DataPointer;
        float* b = (float*)i2v.DataPointer;
        float maxDiff = 0;
        for (long i = 0; i < i2v.Shape.ElementCount; i++)
        {
            Assert.True(float.IsFinite(b[i]), $"non-finite at {i}");
            maxDiff = MathF.Max(maxDiff, MathF.Abs(a[i] - b[i]));
        }
        Assert.True(maxDiff > 1e-4f, $"per-frame timestep 0 on frame 0 must change the output: maxDiff={maxDiff}");

        Assert.Throws<ArgumentException>(() => transformer.Forward(backend, latent, encoder, [0f, 0.5f, 0.5f]));
    }

    private static Dictionary<string, Tensor> BuildWan(WanVideoConfig c)
    {
        int dim = c.InnerDim, ff = c.FfnDim;
        int patchVec = c.InChannels * c.PatchSize.T * c.PatchSize.H * c.PatchSize.W;
        int outVec = c.OutChannels * c.PatchSize.T * c.PatchSize.H * c.PatchSize.W;
        Dictionary<string, Tensor> w = new()
        {
            ["patch_embedding.weight"] = R([dim, c.InChannels, c.PatchSize.T, c.PatchSize.H, c.PatchSize.W]),
            ["patch_embedding.bias"] = R([dim]),
            ["proj_out.weight"] = R([outVec, dim]), ["proj_out.bias"] = R([outVec]),
            ["scale_shift_table"] = R([1, 2, dim]),
            ["condition_embedder.time_embedder.linear_1.weight"] = R([dim, c.FreqDim]), ["condition_embedder.time_embedder.linear_1.bias"] = R([dim]),
            ["condition_embedder.time_embedder.linear_2.weight"] = R([dim, dim]), ["condition_embedder.time_embedder.linear_2.bias"] = R([dim]),
            ["condition_embedder.time_proj.weight"] = R([6 * dim, dim]), ["condition_embedder.time_proj.bias"] = R([6 * dim]),
            ["condition_embedder.text_embedder.linear_1.weight"] = R([dim, c.TextDim]), ["condition_embedder.text_embedder.linear_1.bias"] = R([dim]),
            ["condition_embedder.text_embedder.linear_2.weight"] = R([dim, dim]), ["condition_embedder.text_embedder.linear_2.bias"] = R([dim]),
        };
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"blocks.{i}";
            w[$"{p}.scale_shift_table"] = R([1, 6, dim]);
            w[$"{p}.norm2.weight"] = R([dim]); w[$"{p}.norm2.bias"] = R([dim]);
            foreach (string a in new[] { "attn1", "attn2" })
            {
                w[$"{p}.{a}.to_q.weight"] = R([dim, dim]); w[$"{p}.{a}.to_q.bias"] = R([dim]);
                w[$"{p}.{a}.to_k.weight"] = R([dim, dim]); w[$"{p}.{a}.to_k.bias"] = R([dim]);
                w[$"{p}.{a}.to_v.weight"] = R([dim, dim]); w[$"{p}.{a}.to_v.bias"] = R([dim]);
                w[$"{p}.{a}.to_out.0.weight"] = R([dim, dim]); w[$"{p}.{a}.to_out.0.bias"] = R([dim]);
                w[$"{p}.{a}.norm_q.weight"] = R([dim]); w[$"{p}.{a}.norm_k.weight"] = R([dim]);
            }
            w[$"{p}.ffn.net.0.proj.weight"] = R([ff, dim]); w[$"{p}.ffn.net.0.proj.bias"] = R([ff]);
            w[$"{p}.ffn.net.2.weight"] = R([dim, ff]); w[$"{p}.ffn.net.2.bias"] = R([dim]);
        }
        return w;
    }

    private static int _seed = 1;
    private static Tensor R(int[] dims)
    {
        long[] d = Array.ConvertAll(dims, x => (long)x);
        Tensor t = new Tensor(new TensorShape(d), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(_seed++);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        return t;
    }

    private static Tensor RandRows(int rows, int cols, int seed)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static Tensor RandRows5d(int b, int c, int t, int h, int w, int seed)
    {
        Tensor x = new Tensor(new TensorShape([(long)b, c, t, h, w]), DType.F32);
        float* p = (float*)x.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < x.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return x;
    }

    private static double L2(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        double sum = 0;
        for (long i = 0; i < t.Shape.ElementCount; i++) sum += (double)p[i] * p[i];
        return Math.Sqrt(sum);
    }
}
