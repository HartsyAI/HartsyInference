namespace HartsyInference.Core.Tensors.Quant;

/// <summary>The EXL3 (exllamav3 trellis) layout facts the host codec, the CUDA kernel and the checkpoint binder share, confirmed against exllamav3 d3739fd and the real sfxnz/DeepSeek-V4.1-Flash-EXL3 headers.</summary>
/// <remarks><para>A weight <c>W[in, out]</c> is stored as <c>W = diag(suh) · H · W_hat · H · diag(svh)</c>. <c>W_hat</c> is a trellis-coded
/// matrix in 16x16 tiles, <c>H</c> is a block-diagonal 128-point Sylvester (natural order) Hadamard scaled by
/// <c>1/sqrt(128)</c> per side, and the trellis tensor is <c>[in/16, out/16, 16*bits]</c> int16. Only 2 bits per weight and the
/// MCG codebook are accepted: <c>state * 0xCBAC1FED</c>, then <c>(x &amp; 0x8fff8fff) ^ 0x3b603b60</c>, then the sum of the two halves as fp16.</para>
/// <para>Recipe orientation matches the other encodings: <see cref="QuantRecipe.LogicalRows"/> is the output width (the <c>svh</c> length),
/// <see cref="QuantRecipe.LogicalCols"/> is the input width (the <c>suh</c> length), and the decoded matrix is <c>M[o, i] = W[i, o]</c>.</para></remarks>
public static class Exl3Format
{
    /// <summary>Side of one trellis tile.</summary>
    public const int TileSize = 16;

    /// <summary>Weights in one tile.</summary>
    public const int TileWeights = TileSize * TileSize;

    /// <summary>Width of one Hadamard block; both matrix dimensions must be a multiple of it.</summary>
    public const int HadamardBlock = 128;

    /// <summary>The only supported bits per weight.</summary>
    public const int SupportedBits = 2;

    /// <summary>The MCG multiplier, stored in every weight's <c>mcg</c> tensor.</summary>
    public const uint McgMultiplier = 0xCBAC1FED;

    /// <summary>Per-side Hadamard scale, <c>1/sqrt(128)</c> rounded to F32 exactly as exllamav3 writes it.</summary>
    public const float HadamardScale = 0.08838834764831845f;

    /// <summary>Bytes of the trellis for an <paramref name="inDim"/> x <paramref name="outDim"/> weight.</summary>
    public static long PackedBytes(long inDim, long outDim, int bits) => inDim * outDim * bits / 8;

    /// <summary>Checks that <paramref name="recipe"/> is an EXL3 recipe this engine can decode, naming <paramref name="weightKey"/> in every failure.</summary>
    /// <exception cref="NotSupportedException">The recipe is not EXL3, lacks a companion, or is outside the supported bits, MCG value or geometry.</exception>
    public static void ValidateRecipe(QuantRecipe recipe, string weightKey)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        if (recipe.Encoding != QuantEncoding.Exl3Trellis)
            throw new NotSupportedException($"'{weightKey}' is {recipe.Encoding}, not EXL3.");
        Exl3Companions exl3 = recipe.Exl3 ?? throw new NotSupportedException($"'{weightKey}' is EXL3 but the recipe carries no suh/svh/mcg companions.");
        if (exl3.Bits != SupportedBits)
            throw new NotSupportedException(
                $"'{weightKey}' is EXL3 at {exl3.Bits} bits per weight; only {SupportedBits} bits (K=2, MCG) is decoded.");

        long inDim = recipe.LogicalCols, outDim = recipe.LogicalRows;
        if (inDim <= 0 || outDim <= 0 || inDim % HadamardBlock != 0 || outDim % HadamardBlock != 0)
            throw new NotSupportedException(
                $"'{weightKey}' is EXL3 with in={inDim} and out={outDim}; both must be positive multiples of {HadamardBlock} (block-diagonal Hadamard).");
        if (exl3.Suh.DType != DType.F16 || exl3.Suh.Shape.Rank != 1 || exl3.Suh.ElementCount != inDim)
            throw new NotSupportedException(
                $"'{weightKey}' suh is {exl3.Suh.DType} {exl3.Suh.Shape}; EXL3 needs F16 [{inDim}] (the input width).");
        if (exl3.Svh.DType != DType.F16 || exl3.Svh.Shape.Rank != 1 || exl3.Svh.ElementCount != outDim)
            throw new NotSupportedException(
                $"'{weightKey}' svh is {exl3.Svh.DType} {exl3.Svh.Shape}; EXL3 needs F16 [{outDim}] (the output width).");
        if (exl3.Mcg.DType != DType.I32 || exl3.Mcg.ElementCount != 1)
            throw new NotSupportedException($"'{weightKey}' mcg is {exl3.Mcg.DType} {exl3.Mcg.Shape}; EXL3 needs one I32 scalar.");
        uint mcg = unchecked((uint)exl3.Mcg.AsSpan<int>()[0]);
        if (mcg != McgMultiplier)
            throw new NotSupportedException($"'{weightKey}' mcg is 0x{mcg:X8}; only the MCG codebook multiplier 0x{McgMultiplier:X8} is decoded.");
    }
}
