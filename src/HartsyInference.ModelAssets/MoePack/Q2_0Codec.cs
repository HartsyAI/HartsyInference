using System.Buffers.Binary;

namespace HartsyInference.ModelAssets.MoePack;

/// <summary>
/// Pack-local 2-bit block codec. Each block holds 64 weights as an fp16 scale <c>d</c> and 64 two-bit codes packed four
/// per byte (code <c>i</c> sits in byte <c>2 + i / 4</c> at bit <c>2 * (i % 4)</c>), 18 bytes in all. A code selects
/// <c>grid[code] * d</c> with the grid <c>{-1, 0, 1, 2}</c>. A row of <c>cols</c> values is <c>cols / 64</c> blocks, so
/// <c>cols</c> must be a multiple of 64.
/// </summary>
/// <remarks>
/// Encoding picks the scale by alternating least squares: start at <c>max|x| / 2</c>, assign each value its nearest grid
/// point at the fp16-rounded scale, then refit the scale in closed form to those codes. It repeats up to four times and
/// keeps the candidate with the least squared error, so the stored scale is the best one seen, not the last one fitted.
/// Blocks that are all zero store scale 0. This format is not a ggml type and no GGUF path reads or writes it.
/// </remarks>
public static class Q2_0Codec
{
    /// <summary>Weights per block.</summary>
    public const int BlockElements = 64;

    /// <summary>Stored bytes per block: a 2-byte fp16 scale and 16 bytes of codes.</summary>
    public const int BlockBytes = 18;

    /// <summary>Alternating refits after the initial max-abs scale.</summary>
    private const int MaxRefits = 4;

    /// <summary>Largest finite fp16 value.</summary>
    private const float MaxFp16 = 65504f;

    private static readonly float[] Grid = [-1f, 0f, 1f, 2f];

    /// <summary>Quantizes a row-major <c>[rows, cols]</c> matrix of finite values.</summary>
    /// <exception cref="ArgumentException"><paramref name="cols"/> is not a multiple of 64, the value count does not match,
    /// or a value is not finite.</exception>
    public static byte[] Encode(ReadOnlySpan<float> values, int rows, int cols)
    {
        ValidateShape(values.Length, rows, cols);
        int blocks = rows * (cols / BlockElements);
        byte[] packed = new byte[(long)blocks * BlockBytes];
        for (int b = 0; b < blocks; b++)
        {
            EncodeBlock(values.Slice(b * BlockElements, BlockElements), packed.AsSpan(b * BlockBytes, BlockBytes));
        }
        return packed;
    }

    /// <summary>Dequantizes <paramref name="packed"/> back to a row-major <c>[rows, cols]</c> matrix.</summary>
    /// <exception cref="ArgumentException"><paramref name="cols"/> is not a multiple of 64 or the byte count is wrong.</exception>
    public static float[] Decode(ReadOnlySpan<byte> packed, int rows, int cols)
    {
        ValidateShape(rows * cols, rows, cols);
        int blocks = rows * (cols / BlockElements);
        if (packed.Length != (long)blocks * BlockBytes)
        {
            throw new ArgumentException(
                $"Q2_0 data for {rows} x {cols} needs {(long)blocks * BlockBytes} bytes; it holds {packed.Length}.", nameof(packed));
        }
        float[] values = new float[rows * cols];
        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<byte> block = packed.Slice(b * BlockBytes, BlockBytes);
            float scale = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(block));
            for (int i = 0; i < BlockElements; i++)
            {
                int code = (block[2 + (i >> 2)] >> (2 * (i & 3))) & 3;
                values[b * BlockElements + i] = Grid[code] * scale;
            }
        }
        return values;
    }

    /// <summary>The stored byte count of a <c>[rows, cols]</c> matrix.</summary>
    public static long ByteCount(int rows, int cols)
    {
        ValidateShape(rows * cols, rows, cols);
        return (long)rows * (cols / BlockElements) * BlockBytes;
    }

    private static void ValidateShape(int valueCount, int rows, int cols)
    {
        if (rows <= 0 || cols <= 0)
        {
            throw new ArgumentException($"Q2_0 needs a positive shape; got {rows} x {cols}.");
        }
        if (cols % BlockElements != 0)
        {
            throw new ArgumentException($"Q2_0 blocks hold {BlockElements} values, so {cols} columns do not split into whole blocks.", nameof(cols));
        }
        if ((long)rows * cols != valueCount)
        {
            throw new ArgumentException($"A {rows} x {cols} matrix holds {(long)rows * cols} values; got {valueCount}.");
        }
    }

    private static void EncodeBlock(ReadOnlySpan<float> x, Span<byte> block)
    {
        float maxAbs = 0f;
        for (int i = 0; i < BlockElements; i++)
        {
            if (!float.IsFinite(x[i]))
            {
                throw new ArgumentException("Q2_0 cannot encode a non-finite value.");
            }
            maxAbs = Math.Max(maxAbs, Math.Abs(x[i]));
        }
        block.Clear();
        if (maxAbs == 0f)
        {
            return;
        }
        if (maxAbs / 2f > MaxFp16)
        {
            throw new ArgumentException($"Q2_0 cannot store a block whose largest magnitude is {maxAbs}; the fp16 scale overflows.");
        }

        Span<byte> trial = stackalloc byte[BlockElements];
        Span<byte> best = stackalloc byte[BlockElements];
        Half bestScale = (Half)0f;
        double bestError = double.PositiveInfinity;
        float scale = maxAbs / 2f;
        for (int refit = 0; refit <= MaxRefits; refit++)
        {
            Half rounded = (Half)scale;
            float stored = (float)rounded;
            if (stored == 0f || !float.IsFinite(stored))
            {
                break;
            }
            double error = AssignCodes(x, stored, trial);
            if (error < bestError)
            {
                bestError = error;
                bestScale = rounded;
                trial.CopyTo(best);
            }
            // Closed-form least-squares scale for the codes just chosen: minimize sum (x - g * s)^2 over s.
            double numerator = 0, denominator = 0;
            for (int i = 0; i < BlockElements; i++)
            {
                float g = Grid[trial[i]];
                numerator += x[i] * g;
                denominator += g * g;
            }
            if (denominator == 0 || (float)(numerator / denominator) == scale)
            {
                break;
            }
            scale = (float)(numerator / denominator);
        }
        // No candidate survived the fp16 range check: the block is smaller than the smallest fp16 scale, so it stores zero.
        if (double.IsPositiveInfinity(bestError))
        {
            return;
        }

        BinaryPrimitives.WriteUInt16LittleEndian(block, BitConverter.HalfToUInt16Bits(bestScale));
        for (int i = 0; i < BlockElements; i++)
        {
            block[2 + (i >> 2)] |= (byte)(best[i] << (2 * (i & 3)));
        }
    }

    /// <summary>Sets each code to its nearest grid point at <paramref name="scale"/> and returns the squared error.</summary>
    private static double AssignCodes(ReadOnlySpan<float> x, float scale, Span<byte> codes)
    {
        double error = 0;
        for (int i = 0; i < BlockElements; i++)
        {
            int bestCode = 0;
            double bestDistance = double.PositiveInfinity;
            for (int c = 0; c < Grid.Length; c++)
            {
                double distance = x[i] - Grid[c] * scale;
                distance *= distance;
                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    bestCode = c;
                }
            }
            codes[i] = (byte)bestCode;
            error += bestDistance;
        }
        return error;
    }
}
