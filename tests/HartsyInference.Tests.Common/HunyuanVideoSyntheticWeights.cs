using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Tests.Common;

/// <summary>Tiny deterministic weights for the HunyuanVideo MM-DiT (reused by the Diffusion DiT test and the
/// GameCraft pipeline test). Matches the diffusers key scheme the reused HunyuanImage dual/single blocks expect.</summary>
public static unsafe class HunyuanVideoSyntheticWeights
{
    public static HunyuanVideoConfig TinyConfig => new()
    {
        HiddenSize = 32, NumHeads = 4, NumDoubleBlocks = 2, NumSingleBlocks = 2, MlpDim = 64,
        InChannels = 33, OutChannels = 16, PatchSize = (1, 2, 2), RopeAxesDim = [4, 2, 2],
        TextEmbedDim = 16, PooledEmbedDim = 8,
    };

    public static Dictionary<string, Tensor> BuildDit(HunyuanVideoConfig c, int seed = 7)
    {
        int h = c.HiddenSize, hd = c.HeadDim, m = c.MlpDim;
        int patchVec = c.InChannels * c.PatchSize.T * c.PatchSize.H * c.PatchSize.W;
        int outVec = c.OutChannels * c.PatchSize.T * c.PatchSize.H * c.PatchSize.W;
        Random r = new(seed);
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
}
