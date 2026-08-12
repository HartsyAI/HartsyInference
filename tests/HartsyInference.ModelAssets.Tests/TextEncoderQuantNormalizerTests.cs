using System.Text;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.TextEncoders;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Covers the text-encoder quant normalizer that closes the gap where Qwen-style encoders were
/// loaded straight from safetensors without the per-tensor fp8 scale handling every backbone converter does.</summary>
public sealed unsafe class TextEncoderQuantNormalizerTests
{
    private static Tensor ComfyQuantBlob(string format) => ComfyQuantJson($"{{\"format\": \"{format}\"}}");

    private static Tensor ComfyQuantJson(string json)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(json);
        Tensor t = new(new TensorShape(bytes.Length), DType.U8);
        byte* p = (byte*)t.DataPointer;
        for (int i = 0; i < bytes.Length; i++) p[i] = bytes[i];
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

    /// <summary>The U8 guard below never saw an I8 weight, so an int8 encoder (the 15.4 GB Gemma 4 LTX 2.5 build)
    /// used to slip through unscaled. It must come out packed with its row scale on <c>QuantInfo</c>.</summary>
    [Fact]
    public void Normalize_Int8Convrot_AttachesQuantInfoAndKeepsWeightPacked()
    {
        const string key = "model.layers.0.self_attn.q_proj.weight";
        Tensor rowScale = new(new TensorShape(8, 1), DType.F32);
        Dictionary<string, Tensor> weights = new()
        {
            [key] = new Tensor(new TensorShape(8, 256), DType.I8),
            ["model.layers.0.self_attn.q_proj.weight_scale"] = rowScale,
            ["model.layers.0.self_attn.q_proj.comfy_quant"] = ComfyQuantJson(
                "{\"format\": \"int8_tensorwise\", \"convrot\": true, \"convrot_groupsize\": 256}"),
            ["model.embed_tokens.weight"] = new Tensor(new TensorShape(4, 4), DType.BF16),
        };

        Dictionary<string, Tensor> result = TextEncoderQuantNormalizer.Normalize(weights);

        Tensor weight = result[key];
        Assert.Equal(DType.I8, weight.DType);
        Assert.Equal("int8_tensorwise", weight.QuantInfo!.Format);
        Assert.Equal(256, weight.QuantInfo.ConvRotGroupSize);
        Assert.Same(rowScale, weight.QuantInfo.RowScale);
        Assert.DoesNotContain("model.layers.0.self_attn.q_proj.weight_scale", result.Keys);
        Assert.DoesNotContain("model.layers.0.self_attn.q_proj.comfy_quant", result.Keys);
        Assert.Equal(DType.BF16, result["model.embed_tokens.weight"].DType);
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
    public void Normalize_ValidNvfp4_DequantizesToF16()
    {
        // 8 rows × 32 real cols → packed [8, 16] U8, block scales [8, 2] (16 elements per block).
        Dictionary<string, Tensor> weights = new()
        {
            ["model.layers.0.self_attn.q_proj.weight"] = new Tensor(new TensorShape(8, 16), DType.U8),
            ["model.layers.0.self_attn.q_proj.weight_scale"] = new Tensor(new TensorShape(8, 2), DType.F8E4M3),
            ["model.layers.0.self_attn.q_proj.weight_scale_2"] = F32Scalar(1.0f),
            ["model.layers.0.self_attn.q_proj.comfy_quant"] = ComfyQuantBlob("nvfp4"),
        };

        Dictionary<string, Tensor> result = TextEncoderQuantNormalizer.Normalize(weights);
        Tensor weight = result["model.layers.0.self_attn.q_proj.weight"];
        Assert.Equal(DType.F16, weight.DType);
        Assert.Equal(32, weight.Shape[1]);
        Assert.False(result.ContainsKey("model.layers.0.self_attn.q_proj.weight_scale"));
        Assert.False(result.ContainsKey("model.layers.0.self_attn.q_proj.weight_scale_2"));
        Assert.False(result.ContainsKey("model.layers.0.self_attn.q_proj.comfy_quant"));
    }

    [Fact]
    public void Normalize_MalformedNvfp4BlockScales_ThrowsShapeError()
    {
        // Block-scale shape doesn't cover the packed width (2 packed cols = 4 elements, but scale says 1 block
        // of 16) — must fail loudly at load with the shape diagnosis, not silently pass garbage to the GPU.
        Dictionary<string, Tensor> weights = new()
        {
            ["model.layers.0.self_attn.q_proj.weight"] = new Tensor(new TensorShape(8, 2), DType.U8),
            ["model.layers.0.self_attn.q_proj.weight_scale"] = new Tensor(new TensorShape(8, 1), DType.F8E4M3),
            ["model.layers.0.self_attn.q_proj.weight_scale_2"] = F32Scalar(1.0f),
            ["model.layers.0.self_attn.q_proj.comfy_quant"] = ComfyQuantBlob("nvfp4"),
        };

        ArgumentException ex = Assert.Throws<ArgumentException>(() => TextEncoderQuantNormalizer.Normalize(weights));
        Assert.Contains("NVFP4 block-scale shape", ex.Message);
    }

    [Fact]
    public void Normalize_U8PackedUnknownFormat_ThrowsNamedError()
    {
        // A U8-packed format we still don't dequantize (no nvfp4 companion structure) keeps the clear
        // load-time refusal naming the declared format.
        Dictionary<string, Tensor> weights = new()
        {
            ["model.layers.0.self_attn.q_proj.weight"] = new Tensor(new TensorShape(8, 2), DType.U8),
            ["model.layers.0.self_attn.q_proj.comfy_quant"] = ComfyQuantBlob("svdquant"),
        };

        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => TextEncoderQuantNormalizer.Normalize(weights));
        Assert.Contains("svdquant", ex.Message);
        Assert.Contains("q_proj.weight", ex.Message);
    }
}
