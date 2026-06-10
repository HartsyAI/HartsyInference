using Xunit;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace SharpInference.Diffusion.Tests;

/// <summary>Tests for the LTX-Video DiT pieces — the interleaved 3D <see cref="LtxRope"/> (rotation invariants) and a tiny-config <see cref="LtxVideoTransformer"/> end-to-end on CPU. No GPU/checkpoint; numerics vs the real checkpoint are validation-pending.</summary>
public unsafe class LtxVideoTransformerTests
{
    [Fact]
    public void Rope_PreservesL2Norm()
    {
        LtxRope rope = new(dim: 48, theta: 10000, baseFrames: 20, baseHeight: 2048, baseWidth: 2048);
        (Tensor cos, Tensor sin) = rope.BuildCosSin(2, 2, 2, (1, 1, 1));   // S = 8
        Assert.Equal(8, (int)cos.Shape[0]);
        Assert.Equal(48, (int)cos.Shape[1]);

        Tensor x = RandRows(8, 48, seed: 3);
        double before = L2(x);
        rope.ApplyRotary(x, cos, sin);
        double after = L2(x);
        Assert.True(Math.Abs(before - after) < 1e-3, $"rotation must preserve norm: {before} vs {after}");
    }

    [Fact]
    public void Rope_IdentityPadChannels_Unchanged()
    {
        // dim=128 → 128 % 6 == 2 → the first 2 channels are an identity pad (cos=1, sin=0).
        LtxRope rope = new(dim: 128, theta: 10000, baseFrames: 20, baseHeight: 2048, baseWidth: 2048);
        (Tensor cos, Tensor sin) = rope.BuildCosSin(1, 1, 2, (1, 1, 1));   // S = 2
        Tensor x = RandRows(2, 128, seed: 5);
        float a0 = ((float*)x.DataPointer)[0], a1 = ((float*)x.DataPointer)[1];
        rope.ApplyRotary(x, cos, sin);
        Assert.Equal(a0, ((float*)x.DataPointer)[0], 5);
        Assert.Equal(a1, ((float*)x.DataPointer)[1], 5);
    }

    [Fact]
    public void Rope_IsPositionSensitive()
    {
        LtxRope rope = new(dim: 48, theta: 10000, baseFrames: 20, baseHeight: 2048, baseWidth: 2048);
        (Tensor cos, Tensor sin) = rope.BuildCosSin(2, 1, 1, (1, 1, 1));   // 2 frames, S=2
        Tensor x = RandRows(2, 48, seed: 7);
        // Make rows identical so any post-rotation difference is purely positional.
        Buffer.MemoryCopy((float*)x.DataPointer, (float*)x.DataPointer + 48, 48 * 4, 48 * 4);
        rope.ApplyRotary(x, cos, sin);
        float* p = (float*)x.DataPointer;
        float maxDiff = 0;
        for (int d = 0; d < 48; d++) maxDiff = MathF.Max(maxDiff, MathF.Abs(p[d] - p[48 + d]));
        Assert.True(maxDiff > 0.05f, $"different frames should rotate differently: {maxDiff}");
    }

    [Fact]
    public void Transformer_TinyConfig_ProducesCorrectShape()
    {
        CpuBackend backend = new();
        LtxVideoConfig cfg = new()
        {
            InChannels = 4, OutChannels = 4, NumHeads = 2, HeadDim = 4,
            CrossAttentionDim = 8, NumLayers = 2, CaptionChannels = 16,
        };
        LtxVideoTransformer transformer = new(cfg);
        transformer.LoadWeights(BuildLtx(cfg));

        int f = 2, h = 2, w = 2, s = f * h * w;     // 8 tokens
        Tensor latent = RandRows(s, cfg.InChannels, seed: 11);
        Tensor encoder = RandRows(3, cfg.CaptionChannels, seed: 12);  // L=3 T5 tokens

        Tensor outVel = transformer.Forward(backend, latent, encoder, timestep: 0.5f, (f, h, w), (1, 1, 1), null);

        Assert.Equal(s, (int)outVel.Shape[0]);
        Assert.Equal(cfg.OutChannels, (int)outVel.Shape[1]);
        float* p = (float*)outVel.DataPointer;
        for (long i = 0; i < outVel.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }

    private static Dictionary<string, Tensor> BuildLtx(LtxVideoConfig c)
    {
        int dim = c.InnerDim, ff = 4 * dim;
        Dictionary<string, Tensor> w = new()
        {
            ["proj_in.weight"] = R([dim, c.InChannels]), ["proj_in.bias"] = R([dim]),
            ["proj_out.weight"] = R([c.OutChannels, dim]), ["proj_out.bias"] = R([c.OutChannels]),
            ["scale_shift_table"] = R([2, dim]),
            ["time_embed.emb.timestep_embedder.linear_1.weight"] = R([dim, 256]), ["time_embed.emb.timestep_embedder.linear_1.bias"] = R([dim]),
            ["time_embed.emb.timestep_embedder.linear_2.weight"] = R([dim, dim]), ["time_embed.emb.timestep_embedder.linear_2.bias"] = R([dim]),
            ["time_embed.linear.weight"] = R([6 * dim, dim]), ["time_embed.linear.bias"] = R([6 * dim]),
            ["caption_projection.linear_1.weight"] = R([dim, c.CaptionChannels]), ["caption_projection.linear_1.bias"] = R([dim]),
            ["caption_projection.linear_2.weight"] = R([dim, dim]), ["caption_projection.linear_2.bias"] = R([dim]),
        };
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"transformer_blocks.{i}";
            w[$"{p}.scale_shift_table"] = R([6, dim]);
            foreach (string a in new[] { "attn1", "attn2" })
            {
                w[$"{p}.{a}.to_q.weight"] = R([dim, dim]); w[$"{p}.{a}.to_q.bias"] = R([dim]);
                w[$"{p}.{a}.to_k.weight"] = R([dim, dim]); w[$"{p}.{a}.to_k.bias"] = R([dim]);
                w[$"{p}.{a}.to_v.weight"] = R([dim, dim]); w[$"{p}.{a}.to_v.bias"] = R([dim]);
                w[$"{p}.{a}.to_out.0.weight"] = R([dim, dim]); w[$"{p}.{a}.to_out.0.bias"] = R([dim]);
                w[$"{p}.{a}.norm_q.weight"] = R([dim]); w[$"{p}.{a}.norm_k.weight"] = R([dim]);
            }
            w[$"{p}.ff.net.0.proj.weight"] = R([ff, dim]); w[$"{p}.ff.net.0.proj.bias"] = R([ff]);
            w[$"{p}.ff.net.2.weight"] = R([dim, ff]); w[$"{p}.ff.net.2.bias"] = R([dim]);
        }
        return w;
    }

    private static int s_seed = 1;
    private static Tensor R(int[] dims)
    {
        long[] d = Array.ConvertAll(dims, x => (long)x);
        Tensor t = new Tensor(new TensorShape(d), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(s_seed++);
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

    private static double L2(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        double sum = 0;
        for (long i = 0; i < t.Shape.ElementCount; i++) sum += (double)p[i] * p[i];
        return Math.Sqrt(sum);
    }
}
