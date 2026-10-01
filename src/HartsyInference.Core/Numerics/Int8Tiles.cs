namespace HartsyInference.Core.Numerics;

/// <summary>The weight layout and the activation code of <see cref="Backends.IBackend.LinearI8U8"/>: int8 weights in
/// tiles of 8 output rows by 4 inputs, and uint8 activations centred on 127.
///
/// <para>This is the layout Opus's DNN code, and RNNoise with it, compile their int8 tables in (<c>cgemv8x4</c> in
/// <c>vec_avx.h</c>). A weight of <c>outputs × inputs</c> is cut into 8-row blocks, each block into 4-column groups,
/// and each 8 × 4 tile is stored row by row: 32 bytes, row 0's four weights first. Blocks follow each other, and
/// within a block the tiles run left to right. One tile then holds the four weights each of eight output rows applies
/// to the same four inputs, so a kernel can broadcast four activations and multiply-accumulate all eight rows at
/// once.</para></summary>
public static class Int8Tiles
{
    /// <summary>Output rows per tile.</summary>
    public const int Rows = 8;

    /// <summary>Inputs per tile.</summary>
    public const int Cols = 4;

    /// <summary>Bytes per tile.</summary>
    public const int Bytes = Rows * Cols;

    /// <summary>Offset, in a tiled <c>outputs × inputs</c> matrix, of the weight output <paramref name="row"/> applies
    /// to input <paramref name="col"/>.</summary>
    public static int Offset(int row, int col, int inputs) =>
        (row / Rows * (inputs / Cols) + col / Cols) * Bytes + row % Rows * Cols + col % Cols;

    /// <summary>Tiles a row-major <c>outputs × inputs</c> matrix. Both dimensions must be whole tiles.</summary>
    public static void Pack(ReadOnlySpan<sbyte> rowMajor, int outputs, int inputs, Span<sbyte> tiles)
    {
        CheckShape(rowMajor.Length, outputs, inputs, tiles.Length);
        for (int row = 0; row < outputs; row++)
        {
            for (int col = 0; col < inputs; col++) tiles[Offset(row, col, inputs)] = rowMajor[row * inputs + col];
        }
    }

    /// <summary>Reads a tiled <c>outputs × inputs</c> matrix back into row-major order.</summary>
    public static void Unpack(ReadOnlySpan<sbyte> tiles, int outputs, int inputs, Span<sbyte> rowMajor)
    {
        CheckShape(rowMajor.Length, outputs, inputs, tiles.Length);
        for (int row = 0; row < outputs; row++)
        {
            for (int col = 0; col < inputs; col++) rowMajor[row * inputs + col] = tiles[Offset(row, col, inputs)];
        }
    }

    /// <summary>The uint8 code of activation <paramref name="x"/>: <c>127 + floor(0.5 + fl(127·x))</c>, clamped to
    /// [0, 255].</summary>
    /// <remarks><para>It is the expression RNNoise's default build compiles (<c>vector_ps_to_epi8</c> in
    /// <c>vec_avx.h</c> without AVX2): the product is rounded to float, the half is added in double, which is exact,
    /// and the result is floored, so a tie rounds up. The product, the sum and the floor are done the same way here, so
    /// every in-range value gets the same code. Upstream's AVX2 build instead rounds <c>fma(x, 127, 127)</c> to nearest
    /// even, which differs by one step at an exact tie.</para>
    /// <para>Activations in [-1, 1] map to [0, 254]. Outside that, upstream stores the int into an unsigned char and
    /// wraps; this clamps instead, and NaN maps to 0.</para></remarks>
    public static byte QuantizeActivation(float x)
    {
        double rounded = Math.Floor(0.5 + (double)(127f * x));
        if (!(rounded >= -127.0)) return 0;
        if (rounded >= 128.0) return 255;
        return (byte)(127 + (int)rounded);
    }

    private static void CheckShape(int rowMajorLength, int outputs, int inputs, int tilesLength)
    {
        if (outputs <= 0 || inputs <= 0 || outputs % Rows != 0 || inputs % Cols != 0)
            throw new ArgumentException(
                $"A tiled matrix needs outputs a multiple of {Rows} and inputs a multiple of {Cols}; got {outputs} × {inputs}.");
        long count = (long)outputs * inputs;
        if (rowMajorLength != count || tilesLength != count)
            throw new ArgumentException($"Expected {count} weights on both sides, got {rowMajorLength} and {tilesLength}.");
    }
}
