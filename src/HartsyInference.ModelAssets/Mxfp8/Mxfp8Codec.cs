using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.BlockScale;

namespace HartsyInference.ModelAssets.Mxfp8;

/// <summary>Dequantizer for MXFP8 (`mxfp8_block32`) weights as packaged by ComfyUI (e.g. <c>Comfy-Org/Lens/diffusion_models/lens_mxfp8.safetensors</c>).
///
/// <para><b>Format</b> (from <c>comfy.float.stochastic_round_quantize_mxfp8_by_block</c>): each quantized linear has three companion safetensors keys —
/// <list type="bullet">
/// <item><c>{name}.weight</c> — FP8 E4M3 element values, shape <c>[out, in]</c>.</item>
/// <item><c>{name}.weight_scale</c> — E8M0 (uint8) per-block scales, group size 32 along the input dim, stored in NVIDIA's swizzled <see cref="BlockScaleSwizzle">blocked layout</see> at shape <c>[128·ceil(out/128), 4·ceil((in/32)/4)]</c>.</item>
/// <item><c>{name}.comfy_quant</c> — a small JSON metadata blob (skipped).</item>
/// </list>
/// Dequant: <c>w[o,i] = decode_e4m3(weight[o,i]) · 2^(scale_e8m0 - 127)</c> where the E8M0 byte decodes via the bit reinterpretation <c>(scale &lt;&lt; 23)</c> as float32 (so 127→1.0, 0→0.0), matching upstream exactly (<see cref="Mxfp8ResidentCodec"/>). The weight is the standard <c>nn.Linear [out, in]</c> orientation — no transpose. Verified against <c>comfy.float</c> (reconstruction error at FP8 noise level). A weight stays packed through <see cref="TryAttachResident"/> and is widened per backend by <c>QuantizedWeightPolicy</c>.</summary>
public static unsafe class Mxfp8Codec
{
    /// <summary>Elements per E8M0 block scale.</summary>
    public const int GroupSize = 32;

    /// <summary>Dequantizes one MXFP8 linear weight to BF16 <c>[out, in]</c> on the host — the reference every resident path is measured against.</summary>
    /// <param name="weight">FP8 E4M3 weight, shape <c>[out, in]</c>.</param>
    /// <param name="scale">E8M0 (U8) swizzled block scales.</param>
    public static Tensor DequantLinear(Tensor weight, Tensor scale) => Mxfp8ResidentCodec.DequantToBf16(weight, scale);

    /// <summary>Keeps a packed MXFP8 Linear weight RESIDENT: the F8E4M3 <c>[N, K]</c> tensor is already the shape a backend multiplies, so only its block scales move onto <see cref="Tensor.QuantInfo"/> (<c>Format = "mxfp8"</c>). Returns false, leaving <paramref name="weight"/> untouched, for anything that is not such a weight; the caller then dequantizes eagerly. The scale is <b>borrowed</b> — the dictionary that produced it owns its lifetime.</summary>
    public static bool TryAttachResident(Tensor weight, Tensor scale)
    {
        ArgumentNullException.ThrowIfNull(weight);
        ArgumentNullException.ThrowIfNull(scale);
        if (weight.DType != DType.F8E4M3 || weight.Shape.Rank != 2) return false;
        if (scale.DType != DType.U8 || scale.Shape.Rank != 2) return false;
        long outFeatures = weight.Shape[0], inFeatures = weight.Shape[1];
        if (inFeatures % GroupSize != 0) return false;
        // The blocked layout pads rows to 128 and block columns to 4, and every device kernel (and cuBLASLt) indexes the
        // padded tensor; a smaller one is not this weight's, or not in that layout.
        if (scale.Shape[0] < (outFeatures + 127) / 128 * 128 || scale.Shape[1] < inFeatures / GroupSize
            || scale.Shape[1] % 4 != 0)
            return false;
        weight.QuantInfo = new QuantWeightInfo { Format = "mxfp8", BlockScale = scale };
        return true;
    }

    /// <summary>Finds every MXFP8 companion pair (<c>{name}.weight</c> F8E4M3 + <c>{name}.weight_scale</c> U8, plus the optional <c>{name}.comfy_quant</c>) in a weight dict, attaches the scale to the weight through <see cref="TryAttachResident"/>, and removes the companion keys. Which backends can then hold the weight packed is <c>QuantizedWeightPolicy</c>'s decision, not this one's: a backend without a kernel for it widens to BF16 there. Non-MXFP8 entries pass through untouched, so this is a no-op on the BF16 variant. Returns the number of weights attached.</summary>
    public static int AttachResidentInPlace(Dictionary<string, Tensor> weights)
    {
        List<string> scaleKeys = new();
        foreach (string key in weights.Keys)
            if (key.EndsWith(".weight_scale", StringComparison.Ordinal))
                scaleKeys.Add(key);

        int attached = 0;
        foreach (string scaleKey in scaleKeys)
        {
            string baseName = scaleKey[..^".weight_scale".Length];
            if (!weights.TryGetValue($"{baseName}.weight", out Tensor? weight)) continue;
            Tensor scale = weights[scaleKey];
            if (!TryAttachResident(weight, scale))
            {
                // An F8 weight with a U8 scale is an MXFP8 pair; passed on without its scales it would run as plain fp8.
                if (weight.DType == DType.F8E4M3 && scale.DType == DType.U8)
                    throw new HartsyInferenceException(
                        $"MXFP8 weight '{baseName}' {weight.Shape} has block scales {scale.Shape} that do not cover its padded "
                        + "[128-row, 4-column] tile, so neither the resident path nor a plain fp8 GEMM can read it correctly.");
                continue;
            }
            weights.Remove(scaleKey);
            weights.Remove($"{baseName}.comfy_quant");
            attached++;
        }
        return attached;
    }
}
