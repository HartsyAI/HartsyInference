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

    /// <summary>The descriptor's <c>full_precision_matrix_mult</c>: this layer must dequantize and run a normal GEMM.</summary>
    public bool FullPrecisionMatMul { get; init; }

    /// <summary>Returns these companions narrowed to a contiguous run of weight rows, for a caller splitting a fused weight (QKV) or consuming one in windows.</summary>
    /// <remarks><para><see cref="RowScale"/> is indexed by output row, so a row slice of the weight needs the matching
    /// slice of the scale; a per-tensor scale (one element) covers every row and is shared as-is. <see cref="ConvRotGroupSize"/>
    /// divides the <b>input</b> dimension, which a row split leaves alone.</para>
    /// <para>NVFP4 refuses: <see cref="BlockScale"/> is padded to 128 rows and swizzled, so a row range of the weight is
    /// not a row range of the scales, and slicing it would pair each row with another row's block scales — plausible
    /// output, silently wrong.</para></remarks>
    /// <param name="weightKey">The weight's checkpoint key, named in the refusal so the caller need not re-wrap it.</param>
    public QuantWeightInfo SliceRows(long rowOffset, long rowCount, string weightKey)
    {
        if (BlockScale is not null)
            throw new NotSupportedException(
                $"'{weightKey}' is {Format}, whose block scales use a padded swizzled layout that does not slice by row. "
                + "Dequantize it before splitting or windowing the weight.");
        if (RowScale is null || RowScale.ElementCount == 1)
            return this;
        return this with { RowScale = RowScale.SliceRows(rowOffset, rowCount) };
    }
}
