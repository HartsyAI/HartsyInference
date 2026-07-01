using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Hunyuan3D;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.ThreeD.Tests;

/// <summary>Tiny deterministic weights for CPU structural tests of the Hunyuan3D model + pipeline — no GPU,
/// no checkpoint. Dimensions are minimal but exercise every code path (multi-head attention, AdaLN, CFG,
/// grid decode, marching cubes).</summary>
internal static unsafe class Hunyuan3DSyntheticWeights
{
    public static Dinov2Preset TinyDino => new()
    {
        Name = "tiny-dino", HiddenSize = 32, NumLayers = 2, NumHeads = 4, IntermediateSize = 64,
        ImageSize = 28, PatchSize = 14, // grid 2 → 4 patches, seq = 5 (CLS + 4)
    };

    public static Hunyuan3DConfig TinyConfig => new()
    {
        LatentTokens = 8, LatentChannels = 8, Width = 32, DepthDouble = 1, DepthSingle = 1, NumHeads = 4, CondDim = 32,
        MlpDim = 64, TimestepEmbedDim = 32, TimeFactor = 1000f,
        VaeWidth = 32, VaeDepth = 2, VaeNumHeads = 4, FourierBands = 2,
        NumInferenceSteps = 2, GuidanceScale = 2f, FlowShift = 1f, GridResolution = 16, IsoLevel = 0f, BoundingBox = 1f,
    };

    public static Dictionary<string, Tensor> BuildDino(Dinov2Preset p)
    {
        int h = p.HiddenSize, inter = p.IntermediateSize, np = p.NumPatches;
        Random r = new(11);
        Dictionary<string, Tensor> w = new()
        {
            ["embeddings.patch_embeddings.projection.weight"] = T(r, h, 3, p.PatchSize, p.PatchSize),
            ["embeddings.patch_embeddings.projection.bias"] = T(r, h),
            ["embeddings.cls_token"] = T(r, 1, 1, h),
            ["embeddings.position_embeddings"] = T(r, 1, 1 + np, h),
            ["layernorm.weight"] = Ones(h),
            ["layernorm.bias"] = T(r, h),
        };
        for (int i = 0; i < p.NumLayers; i++)
        {
            string pre = $"encoder.layer.{i}";
            w[$"{pre}.norm1.weight"] = Ones(h); w[$"{pre}.norm1.bias"] = T(r, h);
            w[$"{pre}.attention.attention.query.weight"] = T(r, h, h); w[$"{pre}.attention.attention.query.bias"] = T(r, h);
            w[$"{pre}.attention.attention.key.weight"] = T(r, h, h); w[$"{pre}.attention.attention.key.bias"] = T(r, h);
            w[$"{pre}.attention.attention.value.weight"] = T(r, h, h); w[$"{pre}.attention.attention.value.bias"] = T(r, h);
            w[$"{pre}.attention.output.dense.weight"] = T(r, h, h); w[$"{pre}.attention.output.dense.bias"] = T(r, h);
            w[$"{pre}.layer_scale1.lambda1"] = Const(h, 0.1f);
            w[$"{pre}.norm2.weight"] = Ones(h); w[$"{pre}.norm2.bias"] = T(r, h);
            w[$"{pre}.mlp.fc1.weight"] = T(r, inter, h); w[$"{pre}.mlp.fc1.bias"] = T(r, inter);
            w[$"{pre}.mlp.fc2.weight"] = T(r, h, inter); w[$"{pre}.mlp.fc2.bias"] = T(r, h);
            w[$"{pre}.layer_scale2.lambda1"] = Const(h, 0.1f);
        }
        return w;
    }

    public static Dictionary<string, Tensor> BuildDit(Hunyuan3DConfig c)
    {
        int w_ = c.Width, cdim = c.CondDim, m = c.MlpDim, hd = c.Width / c.NumHeads;
        Random r = new(22);
        Dictionary<string, Tensor> w = new()
        {
            ["latent_in.weight"] = T(r, w_, c.LatentChannels), ["latent_in.bias"] = T(r, w_),
            ["cond_in.weight"] = T(r, w_, cdim), ["cond_in.bias"] = T(r, w_),
            ["time_in.in_layer.weight"] = T(r, w_, c.TimestepEmbedDim), ["time_in.in_layer.bias"] = T(r, w_),
            ["time_in.out_layer.weight"] = T(r, w_, w_), ["time_in.out_layer.bias"] = T(r, w_),
            ["final_layer.adaLN_modulation.1.weight"] = T(r, 2 * w_, w_), ["final_layer.adaLN_modulation.1.bias"] = T(r, 2 * w_),
            ["final_layer.linear.weight"] = T(r, c.LatentChannels, w_), ["final_layer.linear.bias"] = T(r, c.LatentChannels),
        };
        for (int i = 0; i < c.DepthDouble; i++)
        {
            string p = $"double_blocks.{i}";
            foreach (string s in new[] { "img", "txt" })
            {
                w[$"{p}.{s}_mod.lin.weight"] = T(r, 6 * w_, w_); w[$"{p}.{s}_mod.lin.bias"] = T(r, 6 * w_);
                w[$"{p}.{s}_attn.qkv.weight"] = T(r, 3 * w_, w_); w[$"{p}.{s}_attn.qkv.bias"] = T(r, 3 * w_);
                w[$"{p}.{s}_attn.proj.weight"] = T(r, w_, w_); w[$"{p}.{s}_attn.proj.bias"] = T(r, w_);
                w[$"{p}.{s}_attn.norm.query_norm.scale"] = Ones(hd); w[$"{p}.{s}_attn.norm.key_norm.scale"] = Ones(hd);
                w[$"{p}.{s}_mlp.0.weight"] = T(r, m, w_); w[$"{p}.{s}_mlp.0.bias"] = T(r, m);
                w[$"{p}.{s}_mlp.2.weight"] = T(r, w_, m); w[$"{p}.{s}_mlp.2.bias"] = T(r, w_);
            }
        }
        for (int i = 0; i < c.DepthSingle; i++)
        {
            string p = $"single_blocks.{i}";
            w[$"{p}.modulation.lin.weight"] = T(r, 3 * w_, w_); w[$"{p}.modulation.lin.bias"] = T(r, 3 * w_);
            w[$"{p}.linear1.weight"] = T(r, 3 * w_ + m, w_); w[$"{p}.linear1.bias"] = T(r, 3 * w_ + m);
            w[$"{p}.linear2.weight"] = T(r, w_, w_ + m); w[$"{p}.linear2.bias"] = T(r, w_);
            w[$"{p}.norm.query_norm.scale"] = Ones(hd); w[$"{p}.norm.key_norm.scale"] = Ones(hd);
        }
        return w;
    }

    public static Dictionary<string, Tensor> BuildVae(Hunyuan3DConfig c)
    {
        int vw = c.VaeWidth, fdim = 3 + 6 * c.FourierBands, mlp = 4 * vw;
        Random r = new(33);
        Dictionary<string, Tensor> w = new()
        {
            ["latent_proj.weight"] = T(r, vw, c.LatentChannels), ["latent_proj.bias"] = T(r, vw),
            ["query_proj.weight"] = T(r, vw, fdim), ["query_proj.bias"] = T(r, vw),
            ["out.weight"] = T(r, 1, vw), ["out.bias"] = T(r, 1),
        };
        for (int i = 0; i < c.VaeDepth; i++)
        {
            string p = $"blocks.{i}";
            foreach (string proj in new[] { "q", "k", "v", "o" })
            { w[$"{p}.cross_attn.{proj}.weight"] = T(r, vw, vw); w[$"{p}.cross_attn.{proj}.bias"] = T(r, vw); }
            w[$"{p}.mlp.fc1.weight"] = T(r, mlp, vw); w[$"{p}.mlp.fc1.bias"] = T(r, mlp);
            w[$"{p}.mlp.fc2.weight"] = T(r, vw, mlp); w[$"{p}.mlp.fc2.bias"] = T(r, vw);
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

    private static Tensor Ones(long n) => Const(n, 1f);

    private static Tensor Const(long n, float val)
    {
        Tensor t = new(new TensorShape(n), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < n; i++) p[i] = val;
        return t;
    }
}
