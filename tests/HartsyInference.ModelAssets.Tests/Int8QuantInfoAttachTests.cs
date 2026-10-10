using System.Text;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Covers the ComfyUI <c>int8_tensorwise</c> companion fold. The failure this guards is silent: the pass that
/// folds fp8 scales drops <c>.weight_scale</c>/<c>.comfy_quant</c> unconditionally, so an int8 weight whose companions
/// were dropped instead of attached reaches the backend as raw int8 with no scale and no rotation — plausible-looking
/// output, wrong by orders of magnitude.</summary>
public sealed unsafe class Int8QuantInfoAttachTests
{
    private const string ConvRotJson =
        "{\"format\": \"int8_tensorwise\", \"convrot\": true, \"convrot_groupsize\": 256, \"per_row\": true}";

    private static Tensor Blob(string json)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(json);
        Tensor t = new Tensor(new TensorShape(bytes.Length), DType.U8);
        bytes.CopyTo(t.AsSpan<byte>());
        return t;
    }

    private static Tensor RowScale(int rows)
    {
        Tensor t = new Tensor(new TensorShape(rows, 1), DType.F32);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < rows; i++) s[i] = 0.01f * (i + 1);
        return t;
    }

    /// <summary>The shipped LTX 2.5 / H3 layout: I8 <c>[N, K]</c> + F32 <c>[N, 1]</c> row scale + descriptor.</summary>
    private static Dictionary<string, Tensor> ConvRotLayer(string prefix, int rows = 8, int cols = 256,
        string json = ConvRotJson) => new()
        {
            [$"{prefix}.weight"] = new Tensor(new TensorShape(rows, cols), DType.I8),
            [$"{prefix}.weight_scale"] = RowScale(rows),
            [$"{prefix}.comfy_quant"] = Blob(json),
            [$"{prefix}.bias"] = new Tensor(new TensorShape(rows), DType.BF16),
        };

    private static void DisposeAll(Dictionary<string, Tensor> weights)
    {
        foreach (Tensor t in weights.Values) t.Dispose();
    }

    [Fact]
    public void Attach_ConvRotLayer_MovesScaleOntoWeightAndDropsCompanions()
    {
        Dictionary<string, Tensor> weights = ConvRotLayer("blocks.0.attn.qkv_proj");
        try
        {
            Dictionary<string, Tensor> result = CheckpointConvertUtils.AttachInt8QuantInfo(weights);

            Tensor weight = result["blocks.0.attn.qkv_proj.weight"];
            Assert.Equal(DType.I8, weight.DType);
            Assert.NotNull(weight.QuantInfo);
            Assert.Equal("int8_tensorwise", weight.QuantInfo!.Format);
            Assert.Equal(256, weight.QuantInfo.ConvRotGroupSize);
            Assert.False(weight.QuantInfo.FullPrecisionMatMul);
            Assert.Same(weights["blocks.0.attn.qkv_proj.weight_scale"], weight.QuantInfo.RowScale);
            Assert.DoesNotContain("blocks.0.attn.qkv_proj.weight_scale", result.Keys);
            Assert.DoesNotContain("blocks.0.attn.qkv_proj.comfy_quant", result.Keys);
            Assert.True(result.ContainsKey("blocks.0.attn.qkv_proj.bias"));
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    /// <summary>ComfyUI honours <c>full_precision_matrix_mult</c> per layer (H3's <c>mlp.fc2</c>); dropping it costs
    /// accuracy with no error, so it has to survive the fold.</summary>
    [Fact]
    public void Attach_FullPrecisionFlag_ReachesQuantInfo()
    {
        Dictionary<string, Tensor> weights = ConvRotLayer("blocks.0.mlp.fc2",
            json: "{\"format\": \"int8_tensorwise\", \"convrot\": true, \"convrot_groupsize\": 256, "
                + "\"full_precision_matrix_mult\": true}");
        try
        {
            Dictionary<string, Tensor> result = CheckpointConvertUtils.AttachInt8QuantInfo(weights);
            Assert.True(result["blocks.0.mlp.fc2.weight"].QuantInfo!.FullPrecisionMatMul);
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    /// <summary>Without the per-layer descriptor there is no way to know the weight was rotated, and reading a rotated
    /// weight as unrotated is silently wrong — refuse instead of guessing.</summary>
    [Fact]
    public void Attach_MissingDescriptor_Throws()
    {
        Dictionary<string, Tensor> weights = ConvRotLayer("blocks.0.attn.qkv_proj");
        weights["blocks.0.attn.qkv_proj.comfy_quant"].Dispose();
        weights.Remove("blocks.0.attn.qkv_proj.comfy_quant");
        try
        {
            NotSupportedException ex = Assert.Throws<NotSupportedException>(
                () => CheckpointConvertUtils.AttachInt8QuantInfo(weights));
            Assert.Contains("comfy_quant", ex.Message, StringComparison.Ordinal);
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    /// <summary>A checkpoint that quantized only some layers must come out with each weight on its own path — int8
    /// packed with QuantInfo, fp8 folded onto Fp8ScaleFactor, dense untouched — in a single pass.</summary>
    [Fact]
    public void ApplyFp8ScaledDequant_MixedInt8AndFp8_FoldsEachOntoItsOwnPath()
    {
        Dictionary<string, Tensor> weights = ConvRotLayer("blocks.0.attn.qkv_proj");
        Tensor fp8Scale = new Tensor(new TensorShape(1), DType.F32);
        fp8Scale.AsSpan<float>()[0] = 0.5f;
        weights["blocks.0.mlp.fc1.weight"] = new Tensor(new TensorShape(8, 8), DType.F8E4M3);
        weights["blocks.0.mlp.fc1.weight_scale"] = fp8Scale;
        weights["blocks.0.mlp.fc1.comfy_quant"] = Blob("{\"format\": \"float8_e4m3fn\"}");
        weights["blocks.0.norm1.weight"] = new Tensor(new TensorShape(8), DType.BF16);
        try
        {
            Dictionary<string, Tensor> result = CheckpointConvertUtils.ApplyFp8ScaledDequant(weights);

            Tensor int8 = result["blocks.0.attn.qkv_proj.weight"];
            Assert.Equal(DType.I8, int8.DType);
            Assert.Equal(256, int8.QuantInfo!.ConvRotGroupSize);

            Tensor fp8 = result["blocks.0.mlp.fc1.weight"];
            Assert.Equal(DType.F8E4M3, fp8.DType);
            Assert.Null(fp8.QuantInfo);
            Assert.Equal(0.5f, fp8.Fp8ScaleFactor);

            Assert.Equal(DType.BF16, result["blocks.0.norm1.weight"].DType);
            foreach (string key in result.Keys)
            {
                Assert.False(key.EndsWith(".weight_scale", StringComparison.Ordinal));
                Assert.False(key.EndsWith(".comfy_quant", StringComparison.Ordinal));
            }
        }
        finally
        {
            DisposeAll(weights);
        }
    }

}
