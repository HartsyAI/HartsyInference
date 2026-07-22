using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.BlockScale;

namespace HartsyInference.ModelAssets.Mxfp8;

/// <summary>Dequantizer for MXFP8 (`mxfp8_block32`) weights as packaged by ComfyUI (e.g.
/// <c>Comfy-Org/Lens/diffusion_models/lens_mxfp8.safetensors</c>).
///
/// <para><b>Format</b> (from <c>comfy.float.stochastic_round_quantize_mxfp8_by_block</c>): each quantized
/// linear has three companion safetensors keys —
/// <list type="bullet">
/// <item><c>{name}.weight</c> — FP8 E4M3 element values, shape <c>[out, in]</c>.</item>
/// <item><c>{name}.weight_scale</c> — E8M0 (uint8) per-block scales, group size 32 along the input dim,
/// stored in NVIDIA's swizzled <see cref="BlockScaleSwizzle">blocked layout</see> at shape
/// <c>[128·ceil(out/128), 4·ceil((in/32)/4)]</c>.</item>
/// <item><c>{name}.comfy_quant</c> — a small JSON metadata blob (skipped).</item>
/// </list>
/// Dequant: <c>w[o,i] = decode_e4m3(weight[o,i]) · 2^(scale_e8m0 - 127)</c> where the E8M0 byte decodes
/// via the bit reinterpretation <c>(scale &lt;&lt; 23)</c> as float32 (so 127→1.0, 0→0.0), matching upstream
/// exactly. Output is BF16 to keep the 3.8B DiT inside consumer VRAM (F32 would be ~15 GB). The weight is
/// the standard <c>nn.Linear [out, in]</c> orientation — no transpose. Verified against
/// <c>comfy.float</c> (reconstruction error at FP8 noise level).</summary>
public static unsafe class Mxfp8Codec
{
    /// <summary>Elements per E8M0 block scale.</summary>
    public const int GroupSize = 32;

    /// <summary>Dequantizes one MXFP8 linear weight to BF16 <c>[out, in]</c>.</summary>
    /// <param name="weight">FP8 E4M3 weight, shape <c>[out, in]</c>.</param>
    /// <param name="scale">E8M0 (U8) swizzled block scales.</param>
    public static Tensor DequantLinear(Tensor weight, Tensor scale)
    {
        if (weight.DType != DType.F8E4M3)
            throw new ArgumentException($"MXFP8 weight must be F8E4M3; got {weight.DType}.", nameof(weight));
        if (scale.DType != DType.U8)
            throw new ArgumentException($"MXFP8 weight_scale must be U8 (E8M0); got {scale.DType}.", nameof(scale));
        if (weight.Shape.Rank != 2)
            throw new ArgumentException($"MXFP8 weight must be rank-2 [out, in]; got {weight.Shape}.", nameof(weight));

        long outDim = weight.Shape[0];
        long inDim = weight.Shape[1];
        long paddedCols = scale.Shape[scale.Shape.Rank - 1];

        // Decode FP8 E4M3 element values via the verified CastTo path.
        Tensor wF32 = weight.CastTo(DType.F32);
        Tensor outF32 = new Tensor(new TensorShape(outDim, inDim), DType.F32);
        float* w = (float*)wF32.DataPointer;
        byte* s = (byte*)scale.DataPointer;
        float* o = (float*)outF32.DataPointer;

        for (long r = 0; r < outDim; r++)
        {
            for (long c = 0; c < inDim; c++)
            {
                long blockCol = c / GroupSize;
                byte e8m0 = s[BlockScaleSwizzle.SwizzledIndex(r, blockCol, paddedCols)];
                float blockScale = E8M0ToFloat(e8m0);
                o[r * inDim + c] = w[r * inDim + c] * blockScale;
            }
        }
        wF32.Dispose();

        Tensor outBf16 = outF32.CastTo(DType.BF16);
        outF32.Dispose();
        return outBf16;
    }

    /// <summary>Finds every MXFP8 companion triple (<c>{name}.weight</c> F8E4M3 + <c>{name}.weight_scale</c>
    /// U8 + optional <c>{name}.comfy_quant</c>) in a weight dict, dequantizes the weight to BF16 under the
    /// plain <c>{name}.weight</c> key, and removes the scale + comfy_quant companions. Non-MXFP8 entries
    /// (plain BF16 weights, biases, norms) pass through untouched, so this is a safe no-op on the BF16
    /// checkpoint variant. Returns the number of weights dequantized.</summary>
    public static int DequantInPlace(Dictionary<string, Tensor> weights)
    {
        List<string> scaleKeys = new();
        foreach (string key in weights.Keys)
            if (key.EndsWith(".weight_scale", StringComparison.Ordinal))
                scaleKeys.Add(key);

        int dequanted = 0;
        foreach (string scaleKey in scaleKeys)
        {
            string baseName = scaleKey[..^".weight_scale".Length];   // e.g. "...img_qkv"
            string weightKey = $"{baseName}.weight";
            if (!weights.TryGetValue(weightKey, out Tensor? weight)) continue;
            if (weight.DType != DType.F8E4M3) continue;               // not MXFP8 (e.g. NVFP4 handled elsewhere)
            Tensor scale = weights[scaleKey];

            Tensor dq = DequantLinear(weight, scale);
            weights[weightKey] = dq;
            weights.Remove(scaleKey);
            weights.Remove($"{baseName}.comfy_quant");
            weight.Dispose();
            scale.Dispose();
            dequanted++;
        }
        return dequanted;
    }

    /// <summary>Decodes an E8M0 byte to float via the upstream bit reinterpretation
    /// <c>(byte &lt;&lt; 23)</c> as IEEE-754 single — equals <c>2^(byte-127)</c> for byte ≥ 1 and 0.0 for byte 0.</summary>
    private static float E8M0ToFloat(byte e8m0) => BitConverter.UInt32BitsToSingle((uint)e8m0 << 23);
}
