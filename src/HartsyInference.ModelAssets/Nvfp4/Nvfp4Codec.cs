using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.BlockScale;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;

namespace HartsyInference.ModelAssets.Nvfp4;

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
/// matrix is <c>[out, in]</c>; the runtime <see cref="HartsyInference.Diffusion.Models.TextEncoders.GptOssMoeFfn"/>
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

    /// <summary>Relabels a packed NVFP4 Linear weight so a backend can keep it RESIDENT — U8 <c>[N, K/2]</c> becomes a
    /// <see cref="DType.F4E2M1"/> <c>[N, K]</c> view over the same bytes with its scale companions attached through
    /// <see cref="Tensor.QuantInfo"/>. Returns false, writing <paramref name="packed"/> straight through, when the
    /// weight is not one this can serve; the caller then falls back to an eager dequant.
    ///
    /// <para>This is the memory lever for nvfp4 DiTs: unpacked at load, the official 18.72 GB LTX-2.5 distilled
    /// transformer is 42 GB and cannot be placed on a 24 GB card at all. Resident it stays at 0.5 byte/param and each
    /// GEMM unpacks one weight transiently. It is <b>not</b> a speed win — no consumer Ada or Ampere part has FP4
    /// tensor cores, so the GEMM still runs in F16/BF16 off the unpacked copy.</para>
    ///
    /// <para><b>Relabelling is not cosmetic.</b> <c>IBackend.Linear</c> derives <c>K</c> from <c>Shape[1]</c>; left as
    /// U8 <c>[N, K/2]</c> the whole GEMM would silently run at half the true inner dimension.</para>
    ///
    /// <para><b>Lifetime.</b> The view borrows the packed tensor's bytes and roots it, and the returned
    /// <see cref="QuantWeightInfo"/> roots the two companions — so a converter may drop the companion KEYS from its
    /// output dictionary, but must not dispose the packed weight or the companions, and must store the view (not the
    /// original) under the weight's key.</para></summary>
    /// <param name="hasPreQuantScale">True when the weight ships an AWQ <c>pre_quant_scale</c> over the input dim.
    /// Those are refused: the scale has to multiply the ACTIVATION (<c>x·Wᵀ = (x⊙s)·(W/s)ᵀ</c>) and no backend Linear
    /// path applies one, so such a layer must take the eager dequant that folds it in.</param>
    public static bool TryAttachResident(Tensor packed, Tensor blockScale, Tensor globalScale, bool hasPreQuantScale,
        out Tensor resident)
    {
        ArgumentNullException.ThrowIfNull(packed);
        ArgumentNullException.ThrowIfNull(blockScale);
        ArgumentNullException.ThrowIfNull(globalScale);
        resident = packed;
        if (hasPreQuantScale) return false;
        if (packed.DType != DType.U8 || packed.Shape.Rank != 2) return false;
        if (blockScale.DType != DType.F8E4M3 || blockScale.Shape.Rank != 2) return false;
        if (globalScale.DType != DType.F32 || globalScale.ElementCount != 1) return false;

        long outFeatures = packed.Shape[0];
        long inFeatures = packed.Shape[1] * 2;
        if (inFeatures % GroupSize != 0) return false;
        // Rows pad up to 128 and block columns up to 4 in the blocked layout, so the stored scale tensor is never
        // smaller than the logical one; smaller means these companions do not describe this weight.
        if (blockScale.Shape[0] < outFeatures || blockScale.Shape[1] < inFeatures / GroupSize
            || blockScale.Shape[1] % 4 != 0)
        {
            return false;
        }

        resident = packed.ReinterpretAs(DType.F4E2M1, new TensorShape(outFeatures, inFeatures));
        resident.QuantInfo = new QuantWeightInfo
        {
            Format = "nvfp4",
            BlockScale = blockScale,
            GlobalScale = globalScale
        };
        return true;
    }

    /// <summary>Dequantizes one NVFP4 expert bank to F32 <c>[E, in, out]</c> (dequant of the on-disk
    /// <c>[E, out, in]</c> per-expert matrices plus the runtime transpose).
    /// <para><b>Memory note:</b> materializes the WHOLE bank at F32 — for the 20B GPT-OSS encoder that is
    /// ~76 GB across all layers and OOM-kills 64 GB hosts. Production loads keep the bank packed and use
    /// <see cref="DequantExpertSlice"/> one expert at a time instead.</para></summary>
    /// <param name="weight">U8 packed FP4, shape <c>[E, out, in/2]</c>.</param>
    /// <param name="blockScale">FP8 E4M3 swizzled per-block scales, shape <c>[E, paddedRows, paddedCols]</c>.</param>
    /// <param name="globalScale">F32 per-expert global scale, shape <c>[E]</c>.</param>
    public static Tensor DequantExpert(Tensor weight, Tensor blockScale, Tensor globalScale)
    {
        ValidateBank(weight, blockScale, globalScale);

        long E = weight.Shape[0];
        long outDim = weight.Shape[1];
        long inHalf = weight.Shape[2];
        long inDim = inHalf * 2;
        long paddedCols = blockScale.Shape[2];
        long scalePerExpert = blockScale.Shape[1] * paddedCols;

        Tensor output = new Tensor(new TensorShape(E, inDim, outDim), DType.F32);
        byte* w = (byte*)weight.DataPointer;
        byte* bs = (byte*)blockScale.DataPointer;
        float* gs = (float*)globalScale.DataPointer;
        float* o = (float*)output.DataPointer;

        for (long e = 0; e < E; e++)
            DequantExpertCore(w + e * outDim * inHalf, bs + e * scalePerExpert,
                blockScale.Fp8ScaleFactor, gs[e], outDim, inHalf, paddedCols, o + e * inDim * outDim);
        return output;
    }

    /// <summary>Dequantizes ONE expert of an NVFP4 bank into a caller-provided F32 <c>[out, in]</c> tensor
    /// — the on-disk row-major orientation, which is exactly the <c>[N, K]</c> weight layout
    /// <c>IBackend.Linear</c> consumes, so no transpose is needed. This is the memory-bounded
    /// forward-time path: the packed bank stays mmap-backed and only one expert (~100 MB for
    /// GPT-OSS-20B) is ever resident at F32.</summary>
    /// <param name="weight">U8 packed FP4, shape <c>[E, out, in/2]</c>.</param>
    /// <param name="blockScale">FP8 E4M3 swizzled per-block scales, shape <c>[E, paddedRows, paddedCols]</c>.</param>
    /// <param name="globalScale">F32 per-expert global scale, shape <c>[E]</c>.</param>
    /// <param name="expertIdx">Expert to dequantize, in <c>[0, E)</c>.</param>
    /// <param name="destination">Preallocated F32 tensor of shape <c>[out, in]</c>; overwritten fully.</param>
    public static void DequantExpertSlice(Tensor weight, Tensor blockScale, Tensor globalScale,
        int expertIdx, Tensor destination)
    {
        ValidateBank(weight, blockScale, globalScale);

        long E = weight.Shape[0];
        long outDim = weight.Shape[1];
        long inHalf = weight.Shape[2];
        long inDim = inHalf * 2;
        if ((uint)expertIdx >= (uint)E)
            throw new ArgumentOutOfRangeException(nameof(expertIdx), $"Expert index {expertIdx} out of range [0, {E}).");
        if (destination.DType != DType.F32 || destination.Shape.Rank != 2 ||
            destination.Shape[0] != outDim || destination.Shape[1] != inDim)
            throw new ArgumentException(
                $"Destination must be F32 [{outDim}, {inDim}]; got {destination.DType} {destination.Shape}.",
                nameof(destination));

        long paddedCols = blockScale.Shape[2];
        long scalePerExpert = blockScale.Shape[1] * paddedCols;
        DequantExpertSliceRowMajorCore(
            (byte*)weight.DataPointer + (long)expertIdx * outDim * inHalf,
            (byte*)blockScale.DataPointer + (long)expertIdx * scalePerExpert,
            blockScale.Fp8ScaleFactor,
            ((float*)globalScale.DataPointer)[expertIdx],
            outDim, inHalf, paddedCols,
            (float*)destination.DataPointer);
    }

    /// <summary>Row-major slice dequant: on-disk <c>[out, in]</c> → F32 <c>[out, in]</c>. Both the
    /// packed reads and the F32 writes are sequential; the per-16-element block scale is decoded once
    /// per group. Parallel over on-disk rows.</summary>
    private static void DequantExpertSliceRowMajorCore(byte* w, byte* bs, float scaleFactor, float global,
        long outDim, long inHalf, long paddedCols, float* dst)
    {
        float[] e4m3 = CheckpointConvertUtils.E4M3Table;
        int bytesPerGroup = GroupSize / 2;
        long inDim = inHalf * 2;

        Parallel.For(0, (int)outDim, r =>
        {
            byte* wr = w + (long)r * inHalf;
            float* dr = dst + (long)r * inDim;
            float scale = 0f;
            for (long k = 0; k < inHalf; k++)
            {
                if (k % bytesPerGroup == 0)
                    scale = e4m3[bs[BlockScaleSwizzle.SwizzledIndex(r, k / bytesPerGroup, paddedCols)]] * scaleFactor * global;
                byte packed = wr[k];
                dr[2 * k] = E2M1Lut[(packed >> 4) & 0x0F] * scale;   // high nibble = even element
                dr[2 * k + 1] = E2M1Lut[packed & 0x0F] * scale;      // low nibble = odd element
            }
        });
    }

    /// <summary>Rows per dequant tile: 64 rows × 1440 packed bytes (GPT-OSS dims) ≈ 90 KB stays
    /// L2-resident while the tile's column-major walk makes the transposed writes sequential.</summary>
    private const int RowTile = 64;

    /// <summary>Dequantizes one expert matrix (on-disk <c>[out, in]</c>) into <paramref name="dst"/> as
    /// transposed <c>[in, out]</c>. E4M3 block scales are LUT-decoded on the fly (× the tensor-level
    /// <paramref name="scaleFactor"/>, matching the <c>CastTo(F32)</c> semantics) — no transient
    /// full-bank F32 scale copy. Parallel over <see cref="RowTile"/>-row tiles; within a tile the
    /// per-(row, group) scales are decoded once (each covers 16 elements) and the walk is column-major
    /// so both the packed reads (tile stays in L2) and the transposed stores (sequential runs of
    /// <c>tileRows</c> floats) are cache-friendly.</summary>
    private static void DequantExpertCore(byte* w, byte* bs, float scaleFactor, float global,
        long outDim, long inHalf, long paddedCols, float* dst)
    {
        float[] e4m3 = CheckpointConvertUtils.E4M3Table;
        long groups = (inHalf * 2 + GroupSize - 1) / GroupSize;      // scale groups per row (16 elems = 8 bytes)
        int bytesPerGroup = GroupSize / 2;
        int tiles = (int)((outDim + RowTile - 1) / RowTile);

        Parallel.For(0, tiles,
            () => new float[RowTile * groups],
            (tile, _, scaleScratch) =>
            {
                long r0 = (long)tile * RowTile;
                int tileRows = (int)(Math.Min(r0 + RowTile, outDim) - r0);

                for (int tr = 0; tr < tileRows; tr++)
                    for (long g = 0; g < groups; g++)
                        scaleScratch[tr * groups + g] =
                            e4m3[bs[BlockScaleSwizzle.SwizzledIndex(r0 + tr, g, paddedCols)]] * scaleFactor * global;

                for (long k = 0; k < inHalf; k++)
                {
                    long g = k / bytesPerGroup;
                    float* d0 = dst + 2 * k * outDim + r0;           // even element column
                    float* d1 = d0 + outDim;                         // odd element column
                    byte* wk = w + r0 * inHalf + k;
                    for (int tr = 0; tr < tileRows; tr++)
                    {
                        byte packed = wk[(long)tr * inHalf];
                        float s = scaleScratch[tr * groups + g];
                        d0[tr] = E2M1Lut[(packed >> 4) & 0x0F] * s;  // high nibble = even element
                        d1[tr] = E2M1Lut[packed & 0x0F] * s;         // low nibble = odd element
                    }
                }
                return scaleScratch;
            },
            _ => { });
    }

    private static void ValidateBank(Tensor weight, Tensor blockScale, Tensor globalScale)
    {
        if (weight.DType != DType.U8)
            throw new ArgumentException($"NVFP4 weight must be U8; got {weight.DType}.", nameof(weight));
        if (blockScale.DType != DType.F8E4M3)
            throw new ArgumentException($"NVFP4 weight_scale must be F8E4M3; got {blockScale.DType}.", nameof(blockScale));
        if (globalScale.DType != DType.F32)
            throw new ArgumentException($"NVFP4 weight_scale_2 must be F32; got {globalScale.DType}.", nameof(globalScale));
        if (weight.Shape.Rank != 3)
            throw new ArgumentException($"NVFP4 expert weight must be rank-3 [E, out, in/2]; got {weight.Shape}.", nameof(weight));
        if (blockScale.Shape.Rank != 3 || blockScale.Shape[0] != weight.Shape[0])
            throw new ArgumentException(
                $"NVFP4 weight_scale must be rank-3 [E, paddedRows, paddedCols] with E={weight.Shape[0]}; got {blockScale.Shape}.",
                nameof(blockScale));
    }

    /// <summary>Finds every NVFP4 GPT-OSS expert (<c>…experts.gate_up_proj.weight</c> /
    /// <c>…experts.down_proj.weight</c> U8 with <c>.weight_scale</c> + <c>.weight_scale_2</c> companions),
    /// dequantizes it to the forward-ready transposed F32 layout under the bare
    /// <c>…experts.gate_up_proj</c> / <c>…experts.down_proj</c> key, and removes the companions plus the
    /// <c>.comfy_quant</c> blob. Returns the number of expert banks dequantized.
    ///
    /// <para><b>Memory note:</b> dequant-at-load expands the ~13 GB NVFP4 encoder to ~76 GB of F32 —
    /// that OOM-kills 64 GB hosts, so the production Lens load path no longer calls this. It keeps the
    /// packed bank mmap-backed instead and <c>GptOssMoeFfn</c> streams experts through
    /// <see cref="DequantExpertSlice"/> at forward time. Kept for tooling/tests that want an eager
    /// full-bank dequant of a small bank.</para></summary>
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
