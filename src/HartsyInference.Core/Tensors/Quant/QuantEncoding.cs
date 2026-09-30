namespace HartsyInference.Core.Tensors.Quant;

/// <summary>The numerical encoding of a quantized weight, independent of the file container that stores it.</summary>
public enum QuantEncoding
{
    /// <summary>FP8 E4M3 elements with one scale per 2-D block (32x32 in the official checkpoint, 1x32 for Engram).</summary>
    Fp8E4M3BlockE8M0,

    /// <summary>FP8 E4M3 elements with one scale per output row.</summary>
    Fp8E4M3RowE8M0,

    /// <summary>Two E2M1 values per byte (low nibble = even index) with one scale per 32 input elements.</summary>
    Mxfp4E8M0,

    /// <summary>Two E2M1 values per byte (low nibble = even index, as ModelOpt writes it) with E4M3 scales per 16 elements plus a per-tensor F32 global scale.</summary>
    Nvfp4,

    /// <summary>MLX affine 4-bit: eight nibbles per U32 (lowest first), one scale and one bias per group.</summary>
    AffineInt4,

    /// <summary>MLX affine 8-bit: four bytes per U32 (lowest first), one scale and one bias per group; the mixed-precision MLX checkpoints keep attention and shared experts here.</summary>
    AffineInt8,

    /// <summary>EXL3 trellis-coded weights with Hadamard sign vectors and an MCG codebook.</summary>
    Exl3Trellis,

    /// <summary>A GGUF block-quantized type, described by the tensor's own DType.</summary>
    Gguf,
}
