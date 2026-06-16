using System.Text;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.TextEncoders;
using Xunit;

namespace HartsyInference.ModelHandler.Tests;

/// <summary>Covers the text-encoder quant normalizer that closes the gap where Qwen-style encoders were
/// loaded straight from safetensors without the per-tensor fp8 scale handling every backbone converter does.</summary>
public sealed unsafe class TextEncoderQuantNormalizerTests
{
    private static Tensor ComfyQuantBlob(string format)
    {
        byte[] json = Encoding.UTF8.GetBytes($"{{\"format\": \"{format}\"}}");
        Tensor t = new(new TensorShape(json.Length), DType.U8);
        byte* p = (byte*)t.DataPointer;
        for (int i = 0; i < json.Length; i++) p[i] = json[i];
        return t;
    }

    private static Tensor F32Scalar(float v)
    {
        Tensor t = new(new TensorShape(1), DType.F32);
        ((float*)t.DataPointer)[0] = v;
        return t;
    }

    private static Tensor Fp8Weight(int rows, int cols)
    {
        // Contents are irrelevant for this test; only the dtype/scale plumbing matters.
        return new Tensor(new TensorShape(rows, cols), DType.F8E4M3);
    }

    [Fact]
    public void Normalize_Fp8Scaled_FoldsWeightScaleAndDropsCompanions()
    {
        const string key = "model.layers.0.self_attn.q_proj.weight";
        Dictionary<string, Tensor> weights = new()
        {
            [key] = Fp8Weight(8, 4),
            ["model.layers.0.self_attn.q_proj.weight_scale"] = F32Scalar(0.5f),
            ["model.layers.0.self_attn.q_proj.comfy_quant"] = ComfyQuantBlob("float8_e4m3fn"),
            ["model.embed_tokens.weight"] = new Tensor(new TensorShape(4, 4), DType.BF16),
        };

        Dictionary<string, Tensor> result = TextEncoderQuantNormalizer.Normalize(weights);

        Assert.True(result.ContainsKey(key));
        Assert.Equal(0.5f, result[key].Fp8ScaleFactor);
        Assert.DoesNotContain("model.layers.0.self_attn.q_proj.weight_scale", result.Keys);
        Assert.DoesNotContain("model.layers.0.self_attn.q_proj.comfy_quant", result.Keys);
        Assert.True(result.ContainsKey("model.embed_tokens.weight"));
    }

    [Fact]
    public void Normalize_PlainBf16_PassesThroughUnchanged()
    {
        Dictionary<string, Tensor> weights = new()
        {
            ["model.layers.0.self_attn.q_proj.weight"] = new Tensor(new TensorShape(8, 4), DType.BF16),
            ["model.embed_tokens.weight"] = new Tensor(new TensorShape(4, 4), DType.BF16),
        };

        Dictionary<string, Tensor> result = TextEncoderQuantNormalizer.Normalize(weights);

        Assert.Equal(weights.Count, result.Count);
        Assert.Equal(DType.BF16, result["model.layers.0.self_attn.q_proj.weight"].DType);
    }

    [Fact]
    public void Normalize_U8PackedFp4_ThrowsNamedError()
    {
        Dictionary<string, Tensor> weights = new()
        {
            ["model.layers.0.self_attn.q_proj.weight"] = new Tensor(new TensorShape(8, 2), DType.U8),
            ["model.layers.0.self_attn.q_proj.weight_scale"] = new Tensor(new TensorShape(8, 1), DType.F8E4M3),
            ["model.layers.0.self_attn.q_proj.weight_scale_2"] = F32Scalar(1.0f),
            ["model.layers.0.self_attn.q_proj.comfy_quant"] = ComfyQuantBlob("nvfp4"),
        };

        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => TextEncoderQuantNormalizer.Normalize(weights));
        Assert.Contains("nvfp4", ex.Message);
        Assert.Contains("q_proj.weight", ex.Message);
    }
}
