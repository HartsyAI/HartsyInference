using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>CPU structural test for <see cref="HunyuanVideoDit"/> with tiny synthetic weights: validates the
/// 33→16-style patchify, 3-axis (T,H,W) RoPE reuse of the HunyuanImage dual/single blocks, and the
/// unpatchify to a velocity volume of the correct shape with finite values.</summary>
public sealed unsafe class HunyuanVideoDitTests
{
    private static HunyuanVideoConfig Tiny => new()
    {
        HiddenSize = 32, NumHeads = 4, NumDoubleBlocks = 2, NumSingleBlocks = 2, MlpDim = 64,
        InChannels = 6, OutChannels = 4, PatchSize = (1, 2, 2), RopeAxesDim = [4, 2, 2],
        TextEmbedDim = 16, PooledEmbedDim = 8,
    };

    [Fact]
    public void Forward_ProducesFiniteVelocityOfLatentShape()
    {
        using IBackend cpu = new CpuBackend();
        HunyuanVideoConfig c = Tiny;
        HunyuanVideoDit dit = new(c);
        // The shared generator tracks the DiT's key layout (the local BuildWeights predates the token
        // refiner: it emits a plain txt_in Linear, missing txt_in.input_embedder/t_embedder/c_embedder).
        dit.LoadWeights(HunyuanVideoSyntheticWeights.BuildDit(c));

        int T = 2, H = 4, W = 4, L = 3;
        using Tensor latent = Filled(0.05f, 1, c.InChannels, T, H, W);
        using Tensor txt = Filled(0.03f, 1, L, c.TextEmbedDim);
        using Tensor pooled = Filled(0.02f, 1, c.PooledEmbedDim);

        using Tensor v = dit.Forward(cpu, latent, txt, pooled, timestep: 0.5f);
        Assert.Equal(5, v.Shape.Rank);
        Assert.Equal(c.OutChannels, (int)v.Shape[1]);
        Assert.Equal(T, (int)v.Shape[2]);
        Assert.Equal(H, (int)v.Shape[3]);
        Assert.Equal(W, (int)v.Shape[4]);
        float* p = (float*)v.DataPointer;
        for (long i = 0; i < v.ElementCount; i++) Assert.True(float.IsFinite(p[i]));
    }

    private static Dictionary<string, Tensor> BuildWeights(HunyuanVideoConfig c)
    {
        int h = c.HiddenSize, hd = c.HeadDim, m = c.MlpDim;
        int patchVec = c.InChannels * c.PatchSize.T * c.PatchSize.H * c.PatchSize.W;
        int outVec = c.OutChannels * c.PatchSize.T * c.PatchSize.H * c.PatchSize.W;
        Random r = new(7);
        Dictionary<string, Tensor> w = new()
        {
            ["img_in.weight"] = T(r, h, patchVec), ["img_in.bias"] = T(r, h),
            ["txt_in.weight"] = T(r, h, c.TextEmbedDim), ["txt_in.bias"] = T(r, h),
            ["time_in.0.weight"] = T(r, h, 256), ["time_in.0.bias"] = T(r, h),
            ["time_in.2.weight"] = T(r, h, h), ["time_in.2.bias"] = T(r, h),
            ["vector_in.0.weight"] = T(r, h, c.PooledEmbedDim), ["vector_in.0.bias"] = T(r, h),
            ["vector_in.2.weight"] = T(r, h, h), ["vector_in.2.bias"] = T(r, h),
            ["final_layer.mod.weight"] = T(r, 2 * h, h), ["final_layer.mod.bias"] = T(r, 2 * h),
            ["final_layer.proj.weight"] = T(r, outVec, h), ["final_layer.proj.bias"] = T(r, outVec),
        };
        for (int i = 0; i < c.NumDoubleBlocks; i++)
        {
            string p = $"double_blocks.{i}";
            w[$"{p}.norm1.linear.weight"] = T(r, 6 * h, h); w[$"{p}.norm1.linear.bias"] = T(r, 6 * h);
            w[$"{p}.norm1_context.linear.weight"] = T(r, 6 * h, h); w[$"{p}.norm1_context.linear.bias"] = T(r, 6 * h);
            foreach (string proj in new[] { "to_q", "to_k", "to_v" }) { w[$"{p}.attn.{proj}.weight"] = T(r, h, h); w[$"{p}.attn.{proj}.bias"] = T(r, h); }
            w[$"{p}.attn.to_out.0.weight"] = T(r, h, h); w[$"{p}.attn.to_out.0.bias"] = T(r, h);
            foreach (string proj in new[] { "add_q_proj", "add_k_proj", "add_v_proj" }) { w[$"{p}.attn.{proj}.weight"] = T(r, h, h); w[$"{p}.attn.{proj}.bias"] = T(r, h); }
            w[$"{p}.attn.to_add_out.weight"] = T(r, h, h); w[$"{p}.attn.to_add_out.bias"] = T(r, h);
            foreach (string n in new[] { "norm_q", "norm_k", "norm_added_q", "norm_added_k" }) w[$"{p}.attn.{n}.weight"] = Ones(hd);
            w[$"{p}.ff.net.0.proj.weight"] = T(r, m, h); w[$"{p}.ff.net.0.proj.bias"] = T(r, m);
            w[$"{p}.ff.net.2.weight"] = T(r, h, m); w[$"{p}.ff.net.2.bias"] = T(r, h);
            w[$"{p}.ff_context.net.0.proj.weight"] = T(r, m, h); w[$"{p}.ff_context.net.0.proj.bias"] = T(r, m);
            w[$"{p}.ff_context.net.2.weight"] = T(r, h, m); w[$"{p}.ff_context.net.2.bias"] = T(r, h);
        }
        for (int i = 0; i < c.NumSingleBlocks; i++)
        {
            string p = $"single_blocks.{i}";
            w[$"{p}.norm.linear.weight"] = T(r, 3 * h, h); w[$"{p}.norm.linear.bias"] = T(r, 3 * h);
            foreach (string proj in new[] { "to_q", "to_k", "to_v" }) { w[$"{p}.attn.{proj}.weight"] = T(r, h, h); w[$"{p}.attn.{proj}.bias"] = T(r, h); }
            foreach (string n in new[] { "norm_q", "norm_k" }) w[$"{p}.attn.{n}.weight"] = Ones(hd);
            w[$"{p}.proj_mlp.weight"] = T(r, m, h); w[$"{p}.proj_mlp.bias"] = T(r, m);
            w[$"{p}.proj_out.weight"] = T(r, h, h + m); w[$"{p}.proj_out.bias"] = T(r, h);
        }
        return w;
    }

    private static Tensor T(Random r, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(r.NextDouble() * 0.2 - 0.1);
        return t;
    }

    private static Tensor Ones(long n)
    {
        Tensor t = new(new TensorShape(n), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < n; i++) p[i] = 1f;
        return t;
    }

    private static Tensor Filled(float v, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = v;
        return t;
    }
}
