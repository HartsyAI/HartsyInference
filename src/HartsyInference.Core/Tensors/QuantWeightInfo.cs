using HartsyInference.Core.Tensors.Quant;

namespace HartsyInference.Core.Tensors;

/// <summary>Quantization companions a ComfyUI checkpoint ships alongside a weight, carried on the weight itself so a backend can consume it <b>packed</b> instead of materializing a dequantized copy.</summary>
/// <remarks><para>The scalar <see cref="Tensor.Fp8ScaleFactor"/> covers <c>fp8_scaled</c>, where one float folds into
/// the GEMM's alpha. Formats whose companions are tensors (<c>int8_tensorwise</c>'s per-output-row scale, nvfp4's
/// block scales) need this instead. Attaching it to the weight rather than to a per-model linear wrapper is what lets
/// one backend branch serve every model that loads such a checkpoint.</para>
/// <para>The companion tensors are <b>borrowed</b>: the weight dictionary that produced them owns their lifetime, so
/// this type is not <see cref="IDisposable"/> and must not outlive that dictionary.</para></remarks>
public sealed record QuantWeightInfo
{
    /// <summary>ComfyUI's own format name from the <c>.comfy_quant</c> descriptor (<c>int8_tensorwise</c>, <c>nvfp4</c>, …).</summary>
    public required string Format { get; init; }

    /// <summary>F32 per-output-row dequant scale, one per weight row; a per-tensor scale arrives here as a single element.</summary>
    public Tensor? RowScale { get; init; }

    /// <summary>NVFP4 per-16-element block scales (E4M3, NVIDIA's blocked/swizzled layout).</summary>
    public Tensor? BlockScale { get; init; }

    /// <summary>NVFP4 per-tensor F32 scalar multiplying every block scale.</summary>
    public Tensor? GlobalScale { get; init; }

    /// <summary>Hadamard rotation group size along the input dim; 0 when the weight was quantized unrotated.</summary>
    /// <remarks>ConvRot stores <c>W @ Hᵀ</c> and expects the activation to be rotated by <c>H</c> per group at runtime.
    /// The quantizer only rotates a layer when <c>in_features % 256 == 0</c>, so 0 is common and not an error.</remarks>
    public int ConvRotGroupSize { get; init; }

    /// <summary>The block-scaled recipe for checkpoints whose companions are tensors on both sides of a 2-D block (DeepSeek-V4.1); null for every Comfy format above.</summary>
    /// <remarks>Recipes use their own <see cref="Format"/> strings, so <c>BlockScaleFormats.FromQuantFormat</c> never mistakes them for the Blackwell cuBLASLt formats.</remarks>
    public QuantRecipe? Recipe { get; init; }

    /// <summary>The descriptor's <c>full_precision_matrix_mult</c>: this layer must dequantize and run a normal GEMM.</summary>
    public bool FullPrecisionMatMul { get; init; }

    /// <summary>Whether a block-scaled GEMM may consume this weight as stored; false where the layer's policy asks for a dense GEMM.</summary>
    public bool NativeGemmEligible => !FullPrecisionMatMul;

    /// <summary>The dense dtype the eager load would have unpacked this weight to, when it stays packed in that copy's place.</summary>
    /// <remarks>A backend that unpacks the weight for a GEMM computes as that dense weight would — same unpacked values,
    /// same GEMM dtype, same small-m route — so keeping it packed changes the footprint and nothing else. Null leaves the
    /// unpack's precision to the backend.</remarks>
    public DType? DenseDType { get; init; }

    /// <summary>The F32 block-scale grid of a DeepSeek-style FP8 E4M3 weight (<c>weight_scale_inv</c>), <c>[ceil(N / Fp8BlockSize), ceil(K / Fp8BlockSize)]</c>,
    /// one scale per square block of the weight. Null for every other format. A linear on such a weight runs <c>IBackend.Fp8BlockLinear</c>
    /// (see <c>Fp8BlockDispatch</c>), so any projection that reaches <c>IBackend.Linear</c> with it needs no model-specific branch.</summary>
    public Tensor? Fp8BlockScaleInv { get; init; }

    /// <summary>Edge length of one <see cref="Fp8BlockScaleInv"/> block in weight elements (128 for DeepSeek-V3/V4 and ERNIE-5.5).</summary>
    public int Fp8BlockSize { get; init; } = 128;

    /// <summary>Returns these companions narrowed to a contiguous run of weight rows, for a caller splitting a fused weight (QKV) or consuming one in windows.</summary>
    /// <remarks><para><see cref="RowScale"/> is indexed by output row, so a row slice of the weight needs the matching
    /// slice of the scale; a per-tensor scale (one element) covers every row and is shared as-is. <see cref="ConvRotGroupSize"/>
    /// divides the <b>input</b> dimension, which a row split leaves alone.</para>
    /// <para>A block-scaled companion (NVFP4, MXFP8) is padded to 128 rows and swizzled in 128-row tiles laid out
    /// tile-row by tile-row, so a range that starts and ends on a tile boundary is a contiguous run of whole tile-rows
    /// and slices as a view — a fused QKV split at a multiple of 128 rows. A leading run of rows keeps the whole
    /// companion, whose tiles it indexes in place (a weight whose rows the quantizer padded). Any other range would pair
    /// rows with another row's block scales — plausible output, silently wrong — and refuses.</para></remarks>
    /// <param name="weightKey">The weight's checkpoint key, named in the refusal so the caller need not re-wrap it.</param>
    public QuantWeightInfo SliceRows(long rowOffset, long rowCount, string weightKey)
    {
        QuantRecipe? recipe = Recipe?.SliceRows(rowOffset, rowCount, weightKey);
        Tensor? blockScale = BlockScale;
        // A leading run that ends inside a tile keeps the whole companion, whose tiles it indexes in place.
        if (blockScale is not null && (rowOffset != 0 || rowCount % BlockScaleTileRows == 0))
        {
            if (rowOffset % BlockScaleTileRows != 0 || rowCount % BlockScaleTileRows != 0 || blockScale.Shape.Rank != 2)
                throw new NotSupportedException(
                    $"'{weightKey}' is {Format}, whose block scales are swizzled in {BlockScaleTileRows}-row tiles, and rows "
                    + $"[{rowOffset}..{rowOffset + rowCount}) do not start and end on one. Dequantize it before splitting or windowing the weight.");
            blockScale = blockScale.SliceRows(rowOffset, rowCount);
        }
        Tensor? rowScale = RowScale is null || RowScale.ElementCount == 1 ? RowScale : RowScale.SliceRows(rowOffset, rowCount);
        // An FP8 block grid is indexed by block row, so a window must start and end on a block edge to keep its scales.
        Tensor? fp8Scale = Fp8BlockScaleInv;
        if (fp8Scale is not null)
        {
            int block = Fp8BlockSize;
            if (rowOffset % block != 0 || rowCount % block != 0)
                throw new NotSupportedException(
                    $"'{weightKey}' is an FP8 block-scaled weight with {block}-row blocks, and rows [{rowOffset}..{rowOffset + rowCount}) "
                    + "do not start and end on a block edge. Dequantize it before splitting or windowing the weight.");
            fp8Scale = fp8Scale.SliceRows(rowOffset / block, rowCount / block);
        }
        return ReferenceEquals(rowScale, RowScale) && ReferenceEquals(blockScale, BlockScale) && ReferenceEquals(recipe, Recipe) && ReferenceEquals(fp8Scale, Fp8BlockScaleInv)
            ? this : this with { RowScale = rowScale, BlockScale = blockScale, Recipe = recipe, Fp8BlockScaleInv = fp8Scale };
    }

    /// <summary>Rows per swizzle tile of a block-scale companion (cuBLASLt's blocked layout: 128 rows × 4 block columns, tiles ordered by tile-row).</summary>
    public const int BlockScaleTileRows = 128;
}
