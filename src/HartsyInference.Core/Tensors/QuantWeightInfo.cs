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
}
