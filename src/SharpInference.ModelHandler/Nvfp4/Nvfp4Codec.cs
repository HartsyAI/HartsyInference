using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.BlockScale;

namespace SharpInference.ModelHandler.Nvfp4;

/// <summary>Dequantizer for NVFP4 (4-bit, `nvfp4`) weights as packaged by ComfyUI (e.g. the GPT-OSS text
/// encoder <c>Comfy-Org/Lens/text_encoders/gpt_oss_20b_nvfp4.safetensors</c>).
///
/// <para><b>Format</b> (from <c>comfy.float.stochastic_round_quantize_nvfp4_by_block</c>): two-level
/// scaling over E2M1 4-bit elements.
/// <list type="bullet">
/// <item><c>{name}.weight</c> — U8, two FP4 (E2M1) values per byte, <b>high nibble = even element, low
/// nibble = odd element</b> (opposite of MXFP4). Shape <c>[E, out, in/2]</c>.</item>
/// <item><c>{name}.weight_scale</c> — FP8 E4M3 per-block scales, group size 16 along the input dim,
/// stored per-expert in NVIDIA's swizzled <see cref="BlockScaleSwizzle">blocked layout</see>
/// <c>[E, 128·ceil(out/128), 4·ceil((in/16)/4)]</c>.</item>
/// <item><c>{name}.weight_scale_2</c> — F32 per-expert global scale, shape <c>[E]</c>.</item>
/// <item><c>{name}.comfy_quant</c> — JSON metadata (skipped).</item>
/// </list>
/// Dequant: <c>w[o,i] = e2m1(nibble) · global[e] · decode_e4m3(block_scale)</c>. The on-disk per-expert
/// matrix is <c>[out, in]</c>; the runtime <see cref="SharpInference.Diffusion.Models.TextEncoders.GptOssMoeFfn"/>
/// expects <c>[E, in, out]</c> (gate_up → <c>[E, hidden, 2·intermediate]</c>, down → <c>[E, intermediate,
/// hidden]</c>), so this codec bakes the last-two-axis transpose into the output. Verified against
/// <c>comfy.float</c> (swizzle round-trip exact; reconstruction error at FP4 noise level).</summary>
public static unsafe class Nvfp4Codec
{
    /// <summary>Elements per E4M3 block scale.</summary>
    public const int GroupSize = 16;

    /// <summary>The 16-entry E2M1 FP4 lookup table (same magnitudes as MXFP4; nibble bit-3 = sign).</summary>
    public static readonly float[] E2M1Lut =
    [
        +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    ];

    /// <summary>Dequantizes one NVFP4 expert bank to F32 <c>[E, in, out]</c> (dequant of the on-disk
    /// <c>[E, out, in]</c> per-expert matrices plus the runtime transpose).</summary>
    /// <param name="weight">U8 packed FP4, shape <c>[E, out, in/2]</c>.</param>
    /// <param name="blockScale">FP8 E4M3 swizzled per-block scales, shape <c>[E, paddedRows, paddedCols]</c>.</param>
    /// <param name="globalScale">F32 per-expert global scale, shape <c>[E]</c>.</param>
    public static Tensor DequantExpert(Tensor weight, Tensor blockScale, Tensor globalScale)
    {
        if (weight.DType != DType.U8)
            throw new ArgumentException($"NVFP4 weight must be U8; got {weight.DType}.", nameof(weight));
        if (blockScale.DType != DType.F8E4M3)
            throw new ArgumentException($"NVFP4 weight_scale must be F8E4M3; got {blockScale.DType}.", nameof(blockScale));
        if (globalScale.DType != DType.F32)
            throw new ArgumentException($"NVFP4 weight_scale_2 must be F32; got {globalScale.DType}.", nameof(globalScale));
        if (weight.Shape.Rank != 3)
            throw new ArgumentException($"NVFP4 expert weight must be rank-3 [E, out, in/2]; got {weight.Shape}.", nameof(weight));

        long E = weight.Shape[0];
        long outDim = weight.Shape[1];
        long inHalf = weight.Shape[2];
        long inDim = inHalf * 2;
        long paddedRows = blockScale.Shape[1];
        long paddedCols = blockScale.Shape[2];
        long scalePerExpert = paddedRows * paddedCols;

        // Decode the E4M3 block scales via the verified CastTo path.
        Tensor scaleF32 = blockScale.CastTo(DType.F32);

        Tensor output = new Tensor(new TensorShape(E, inDim, outDim), DType.F32);
        byte* w = (byte*)weight.DataPointer;
        float* bs = (float*)scaleF32.DataPointer;
        float* gs = (float*)globalScale.DataPointer;
        float* o = (float*)output.DataPointer;

        for (long e = 0; e < E; e++)
        {
            float global = gs[e];
            byte* we = w + e * outDim * inHalf;
            float* bse = bs + e * scalePerExpert;
            for (long r = 0; r < outDim; r++)        // on-disk out dim
            {
                byte* wr = we + r * inHalf;
                for (long k = 0; k < inHalf; k++)
                {
                    byte packed = wr[k];
                    int hi = (packed >> 4) & 0x0F;    // even element
                    int lo = packed & 0x0F;           // odd element
                    long c0 = 2 * k;
                    long c1 = 2 * k + 1;
                    float s0 = bse[BlockScaleSwizzle.SwizzledIndex(r, c0 / GroupSize, paddedCols)];
                    float s1 = bse[BlockScaleSwizzle.SwizzledIndex(r, c1 / GroupSize, paddedCols)];
                    // Transposed write: output[e, in, out] at ((e*inDim + c)*outDim + r).
                    o[(e * inDim + c0) * outDim + r] = E2M1Lut[hi] * global * s0;
                    o[(e * inDim + c1) * outDim + r] = E2M1Lut[lo] * global * s1;
                }
            }
        }
        scaleF32.Dispose();
        return output;
    }

    /// <summary>Finds every NVFP4 GPT-OSS expert (<c>…experts.gate_up_proj.weight</c> /
    /// <c>…experts.down_proj.weight</c> U8 with <c>.weight_scale</c> + <c>.weight_scale_2</c> companions),
    /// dequantizes it to the forward-ready transposed F32 layout under the bare
    /// <c>…experts.gate_up_proj</c> / <c>…experts.down_proj</c> key, and removes the companions plus the
    /// <c>.comfy_quant</c> blob. Returns the number of expert banks dequantized.
    ///
    /// <para><b>Memory note:</b> dequant-at-load expands the ~13 GB NVFP4 encoder to a large F32 footprint
    /// (the experts dominate). Acceptable on a big-RAM host running the encoder on CPU; per-layer streaming
    /// dequant is the follow-up for tight-VRAM/RAM operation.</para></summary>
    public static int DequantGptOssExpertsInPlace(Dictionary<string, Tensor> weights)
    {
        List<string> weightKeys = new();
        foreach (string key in weights.Keys)
            if ((key.EndsWith("experts.gate_up_proj.weight", StringComparison.Ordinal) ||
                 key.EndsWith("experts.down_proj.weight", StringComparison.Ordinal)) &&
                weights[key].DType == DType.U8)
                weightKeys.Add(key);

        int dequanted = 0;
        foreach (string weightKey in weightKeys)
        {
            string baseName = weightKey[..^".weight".Length];        // "…experts.gate_up_proj"
            string scaleKey = $"{baseName}.weight_scale";
            string globalKey = $"{baseName}.weight_scale_2";
            if (!weights.TryGetValue(scaleKey, out Tensor? scale) ||
                !weights.TryGetValue(globalKey, out Tensor? global))
                throw new InvalidOperationException(
                    $"NVFP4 weight '{weightKey}' is missing companion '{scaleKey}' or '{globalKey}'.");
            Tensor weight = weights[weightKey];

            Tensor dq = DequantExpert(weight, scale, global);
            weights[baseName] = dq;
            weights.Remove(weightKey);
            weights.Remove(scaleKey);
            weights.Remove(globalKey);
            weights.Remove($"{baseName}.comfy_quant");
            weight.Dispose();
            scale.Dispose();
            global.Dispose();
            dequanted++;
        }
        return dequanted;
    }
}
