using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Guards <see cref="CheckpointConvertUtils.ApplyFp8ScaledDequant"/> against the Wan-Animate checkerboard
/// regression: the per-tensor fp8 <c>.scale_weight</c> companion must be folded into <c>Fp8ScaleFactor</c> whether it
/// ships as F32 (original Wan/Comfy fp8_scaled) or BF16/F16 (Kijai WanVideo_comfy_fp8_scaled v2). The old
/// <c>== DType.F32</c> guard silently dropped BF16 scales → the raw fp8 weights ran hot → block activations exploded
/// and every DiT token collapsed to one direction = the 16 px halftone tile.</summary>
public sealed unsafe class Fp8ScaleCompanionDtypeTests
{
    private static Tensor Fp8Weight(int rows, int cols)
    {
        Tensor w = new(new TensorShape(rows, cols), DType.F8E4M3);
        byte* p = (byte*)w.DataPointer;
        for (long i = 0; i < w.Shape.ElementCount; i++) p[i] = 0x38;   // e4m3 0x38 = +1.0
        return w;
    }

    private static Tensor Bf16Scalar(float value)
    {
        Tensor t = new(new TensorShape(1), DType.BF16);
        uint bits = (uint)BitConverter.SingleToInt32Bits(value);
        ((ushort*)t.DataPointer)[0] = (ushort)(bits >> 16);   // truncate to bf16 (exact for value=0.25)
        return t;
    }

    private static Tensor F32Scalar(float value)
    {
        Tensor t = new(new TensorShape(1), DType.F32);
        ((float*)t.DataPointer)[0] = value;
        return t;
    }

    [Fact]
    public void Bf16ScaleCompanion_IsFoldedIntoFp8ScaleFactor()
    {
        Dictionary<string, Tensor> src = new()
        {
            ["blocks.0.self_attn.q.weight"] = Fp8Weight(4, 4),
            ["blocks.0.self_attn.q.scale_weight"] = Bf16Scalar(0.25f),
        };
        Dictionary<string, Tensor> result = CheckpointConvertUtils.ApplyFp8ScaledDequant(src);

        Assert.True(result.ContainsKey("blocks.0.self_attn.q.weight"));
        Assert.False(result.ContainsKey("blocks.0.self_attn.q.scale_weight"));   // companion dropped
        Assert.Equal(0.25f, result["blocks.0.self_attn.q.weight"].Fp8ScaleFactor, 4);   // NOT the default 1.0
    }

    [Fact]
    public void F32ScaleCompanion_StillFoldedIdentically()
    {
        Dictionary<string, Tensor> src = new()
        {
            ["blocks.0.ffn.0.weight"] = Fp8Weight(4, 4),
            ["blocks.0.ffn.0.scale_weight"] = F32Scalar(0.375f),
        };
        Dictionary<string, Tensor> result = CheckpointConvertUtils.ApplyFp8ScaledDequant(src);

        Assert.Equal(0.375f, result["blocks.0.ffn.0.weight"].Fp8ScaleFactor, 6);
    }
}
