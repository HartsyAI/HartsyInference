using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.Vision.Dinov3.Tests;

/// <summary>Synthetic DINOv3 and DINOv2 checkpoints in the HF key layout, sized from a preset. Values are seeded.</summary>
internal static class Dinov3Fixture
{
    /// <summary>Only the keys and shapes <see cref="Dinov2Preset.FromDinov3Weights"/> reads, for a ViT-L/16 checkpoint.</summary>
    public static Dictionary<string, Tensor> LargeHeaderOnly()
    {
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal)
        {
            ["embeddings.patch_embeddings.weight"] = Alloc(1024, 3, 16, 16),
            ["embeddings.cls_token"] = Alloc(1, 1, 1024),
            ["embeddings.register_tokens"] = Alloc(1, 4, 1024),
            ["layer.0.mlp.up_proj.weight"] = Alloc(4096, 1024),
        };
        for (int i = 0; i < 24; i++)
            w[$"layer.{i}.norm1.weight"] = Alloc(1024);
        return w;
    }

    /// <summary>Only the keys <see cref="Dinov2Preset.FromWeights"/> reads, for a DINOv2 ViT-L/14 checkpoint.</summary>
    public static Dictionary<string, Tensor> Dinov2LargeHeader()
    {
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal)
        {
            ["embeddings.cls_token"] = Alloc(1, 1, 1024),
            ["embeddings.patch_embeddings.projection.weight"] = Alloc(1024, 3, 14, 14),
        };
        for (int i = 0; i < 24; i++)
            w[$"encoder.layer.{i}.norm1.weight"] = Alloc(1024);
        return w;
    }

    /// <summary>A complete HF-layout DINOv3 checkpoint for a small preset (hidden 128, two 64-wide heads, a 4-storage-token
    /// layout). Norm weights are 1 and biases 0, so the final norm's output is a plain per-token standardization.</summary>
    public static Dictionary<string, Tensor> Tiny(Dinov2Preset preset, int seed)
    {
        Random rng = new(seed);
        int h = preset.HiddenSize, inter = preset.IntermediateSize, p = preset.PatchSize;
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal);
        void Fill(string key, float scale, params long[] shape)
        {
            Tensor t = Alloc(shape);
            Span<float> s = t.AsSpan<float>();
            for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
            w[key] = t;
        }
        void Ones(string key, long n)
        {
            Tensor t = Alloc(n);
            t.AsSpan<float>().Fill(1f);
            w[key] = t;
        }
        void Zeros(string key, params long[] shape) => w[key] = Alloc(shape);

        Fill("embeddings.patch_embeddings.weight", 0.05f, h, 3, p, p);
        Fill("embeddings.patch_embeddings.bias", 0.05f, h);
        Fill("embeddings.cls_token", 0.5f, 1, 1, h);
        Fill("embeddings.register_tokens", 0.5f, 1, preset.NumRegisterTokens, h);
        for (int i = 0; i < preset.NumLayers; i++)
        {
            string l = $"layer.{i}.";
            Ones(l + "norm1.weight", h); Zeros(l + "norm1.bias", h);
            Fill(l + "attention.q_proj.weight", 0.05f, h, h); Fill(l + "attention.q_proj.bias", 0.05f, h);
            Fill(l + "attention.k_proj.weight", 0.05f, h, h);
            Fill(l + "attention.v_proj.weight", 0.05f, h, h); Fill(l + "attention.v_proj.bias", 0.05f, h);
            Fill(l + "attention.o_proj.weight", 0.05f, h, h); Fill(l + "attention.o_proj.bias", 0.05f, h);
            Fill(l + "layer_scale1.lambda1", 0.1f, h);
            Ones(l + "norm2.weight", h); Zeros(l + "norm2.bias", h);
            Fill(l + "mlp.up_proj.weight", 0.05f, inter, h); Fill(l + "mlp.up_proj.bias", 0.05f, inter);
            Fill(l + "mlp.down_proj.weight", 0.05f, h, inter); Fill(l + "mlp.down_proj.bias", 0.05f, h);
            Fill(l + "layer_scale2.lambda1", 0.1f, h);
        }
        Ones("norm.weight", h); Zeros("norm.bias", h);
        return w;
    }

    public static Dinov2Preset TinyPreset(int imageSize = 64) => new()
    {
        Name = "tiny-dinov3",
        HiddenSize = 128, NumLayers = 2, NumHeads = 2, IntermediateSize = 256,
        ImageSize = imageSize, PatchSize = 16, NumRegisterTokens = 4, UsesRotaryPositionEmbedding = true, LayerNormEps = 1e-5f,
    };

    public static void Dispose(Dictionary<string, Tensor> weights)
    {
        foreach (Tensor t in weights.Values) t.Dispose();
    }

    private static Tensor Alloc(params long[] shape) => new(new TensorShape(shape), DType.F32);
}
