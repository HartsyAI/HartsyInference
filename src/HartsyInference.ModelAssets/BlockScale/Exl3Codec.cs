using System.Numerics;
using System.Runtime.CompilerServices;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors.Quant;

namespace HartsyInference.ModelAssets.BlockScale;

/// <summary>Host decoder for EXL3 (exllamav3) 2-bit trellis weights: tile decode, block-diagonal Hadamard, then <c>suh</c>/<c>svh</c> sign-scales. See <see cref="Exl3Format"/> for the layout.</summary>
/// <remarks><para>The trellis stage is bit-exact against exllamav3's own <c>reconstruct_tile</c> kernel (checked by the fixture tests). The rotation runs in
/// F32 with a fixed operation order (in-axis butterfly, scale, <c>suh</c>, out-axis butterfly, scale, <c>svh</c>) that the CUDA twin repeats operation for
/// operation, so host F32 and device F32 agree exactly before the BF16 rounding. exllamav3's fused kernel runs its butterflies in fp16, so the result
/// differs from it by fp16 rounding; the tests pin that gap to a stated bound instead of hiding it.</para>
/// <para>Row windows are output windows and must be multiples of 128 (the Hadamard block). The packed span is always the whole trellis.</para></remarks>
public static class Exl3Codec
{
    private const int Tile = Exl3Format.TileSize;
    private const int Block = Exl3Format.HadamardBlock;
    private const int TilesPerBlock = Block / Tile;
    private const int TileBytes = Tile * Exl3Format.SupportedBits * 2;

    /// <summary>Dequantizes <paramref name="rowCount"/> output rows from <paramref name="rowOffset"/> to F32, as <c>M[o, i] = W[i, o]</c>.</summary>
    /// <param name="packed">The recipe's whole trellis, <c>[in/16, out/16, 32]</c> int16 as raw bytes.</param>
    /// <param name="recipe">An EXL3 recipe with <c>suh</c>, <c>svh</c> and <c>mcg</c>.</param>
    /// <param name="rowOffset">First output row; a multiple of 128.</param>
    /// <param name="rowCount">Rows to decode; a multiple of 128.</param>
    /// <param name="dest">Receives <c>rowCount * LogicalCols</c> floats.</param>
    /// <exception cref="NotSupportedException">The recipe is not a supported EXL3 layout, or the window is not on 128-row boundaries.</exception>
    public static unsafe void DequantRows(ReadOnlySpan<byte> packed, QuantRecipe recipe, long rowOffset, long rowCount, Span<float> dest)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        Exl3Format.ValidateRecipe(recipe, "(EXL3 recipe)");
        long inDim = recipe.LogicalCols, outDim = recipe.LogicalRows;
        long expected = Exl3Format.PackedBytes(inDim, outDim, Exl3Format.SupportedBits);
        if (packed.Length != expected)
            throw new ArgumentException($"Packed trellis is {packed.Length} bytes; an {inDim}x{outDim} 2-bit EXL3 weight is {expected}.", nameof(packed));
        if (rowOffset < 0 || rowCount < 0 || rowOffset + rowCount > outDim)
            throw new ArgumentOutOfRangeException(nameof(rowOffset), $"Rows [{rowOffset}..{rowOffset + rowCount}) are outside [0..{outDim}).");
        if (rowOffset % Block != 0 || rowCount % Block != 0)
            throw new NotSupportedException(
                $"EXL3 decodes whole {Block}-row Hadamard blocks; rows [{rowOffset}..{rowOffset + rowCount}) are not on {Block}-row boundaries.");
        if (dest.Length != rowCount * inDim)
            throw new ArgumentException($"Destination holds {dest.Length} floats; {rowCount} rows of {inDim} need {rowCount * inDim}.", nameof(dest));
        if (rowCount == 0) return;

        int inBlocks = (int)(inDim / Block), rowBlocks = (int)(rowCount / Block), firstOutBlock = (int)(rowOffset / Block);
        int outTiles = (int)(outDim / Tile);
        ushort* suh = (ushort*)recipe.Exl3!.Suh.DataPointer;
        ushort* svh = (ushort*)recipe.Exl3.Svh.DataPointer;
        nint suhAddr = (nint)suh, svhAddr = (nint)svh;
        long cols = inDim;

        fixed (byte* src = packed)
        fixed (float* dst = dest)
        {
            nint srcAddr = (nint)src, dstAddr = (nint)dst;
            CpuParallel.For(rowBlocks * inBlocks, (long)rowBlocks * inBlocks * Block * Block * 8, unit =>
            {
                float[] buf = ArrayPool<float>.Shared.Rent(Block * Block);
                try
                {
                    int rb = unit / inBlocks, ib = unit % inBlocks, ob = firstOutBlock + rb;
                    DecodeBlock((byte*)srcAddr, outTiles, ib, ob, buf);
                    RotateBlock(buf, (ushort*)suhAddr + ib * Block, (ushort*)svhAddr + ob * Block);
                    float* rows = (float*)dstAddr + (long)rb * Block * cols + (long)ib * Block;
                    for (int ol = 0; ol < Block; ol++)
                        buf.AsSpan(ol * Block, Block).CopyTo(new Span<float>(rows + ol * cols, Block));
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(buf);
                }
            });
        }
    }

    /// <summary>Decodes only the trellis stage into <paramref name="whatHat"/> as fp16, laid out <c>[in, out]</c> row-major: exllamav3's <c>reconstruct_tile</c> output before any rotation.</summary>
    /// <param name="packed">The whole trellis, <c>[in/16, out/16, 32]</c> int16 as raw bytes.</param>
    /// <param name="inDim">Input width, a multiple of 16.</param>
    /// <param name="outDim">Output width, a multiple of 16.</param>
    /// <param name="whatHat">Receives <c>inDim * outDim</c> halves.</param>
    public static unsafe void DecodeTrellis(ReadOnlySpan<byte> packed, int inDim, int outDim, Span<Half> whatHat)
    {
        if (inDim <= 0 || outDim <= 0 || inDim % Tile != 0 || outDim % Tile != 0)
            throw new ArgumentException($"EXL3 dimensions must be positive multiples of {Tile}; got {inDim}x{outDim}.");
        long expected = Exl3Format.PackedBytes(inDim, outDim, Exl3Format.SupportedBits);
        if (packed.Length != expected)
            throw new ArgumentException($"Packed trellis is {packed.Length} bytes; an {inDim}x{outDim} 2-bit EXL3 weight is {expected}.", nameof(packed));
        if (whatHat.Length != (long)inDim * outDim)
            throw new ArgumentException($"Destination holds {whatHat.Length} halves; {inDim}x{outDim} needs {(long)inDim * outDim}.", nameof(whatHat));

        int inTiles = inDim / Tile, outTiles = outDim / Tile;
        Span<Half> tile = stackalloc Half[Tile * Tile];
        for (int ia = 0; ia < inTiles; ia++)
        {
            for (int oa = 0; oa < outTiles; oa++)
            {
                DecodeTile(packed.Slice(checked((int)(((long)ia * outTiles + oa) * TileBytes)), TileBytes), tile);
                for (int p = 0; p < Tile * Tile; p++)
                {
                    (int r, int c) = TilePosition(p);
                    whatHat[(ia * Tile + r) * outDim + oa * Tile + c] = tile[p];
                }
            }
        }
    }

    /// <summary>The (input row, output column) inside a 16x16 tile of the p-th decoded value, the mma fragment order exllamav3 stores tiles in.</summary>
    internal static (int Row, int Col) TilePosition(int p)
    {
        int lane = p >> 3, e = p & 7;
        return ((lane & 3) * 2 + (e & 1) + ((e >> 1) & 1) * 8, (lane >> 2) + (e >> 2) * 8);
    }

    /// <summary>Decodes one tile's 64 bytes to its 256 MCG values in stored order, each already rounded to fp16.</summary>
    internal static void DecodeTile(ReadOnlySpan<byte> tileBytes, Span<Half> values)
    {
        Span<uint> w = stackalloc uint[16];
        for (int j = 0; j < 16; j++)
            w[j] = (uint)(tileBytes[4 * j] | (tileBytes[4 * j + 1] << 8) | (tileBytes[4 * j + 2] << 16) | (tileBytes[4 * j + 3] << 24));
        for (int p = 0; p < Tile * Tile; p++)
        {
            // The 16-bit state of position p is the ring window ending at bit (p+1)*2 of the MSB-first stream (tail-biting, 512 bits).
            int start = (p * Exl3Format.SupportedBits + Exl3Format.SupportedBits - 16 + 512) & 511;
            int wi = start >> 5, o = start & 31;
            ulong two = ((ulong)w[wi] << 32) | w[(wi + 1) & 15];
            uint state = (uint)((two >> (48 - o)) & 0xFFFF);
            uint x = unchecked(state * Exl3Format.McgMultiplier);
            x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
            float sum = (float)BitConverter.UInt16BitsToHalf((ushort)x) + (float)BitConverter.UInt16BitsToHalf((ushort)(x >> 16));
            values[p] = (Half)sum;
        }
    }

    // Fills buf[ol * 128 + il] with the trellis values of the 128x128 block (input block ib, output block ob) as F32.
    private static unsafe void DecodeBlock(byte* trellis, int outTiles, int ib, int ob, float[] buf)
    {
        Span<Half> tile = stackalloc Half[Tile * Tile];
        for (int it = 0; it < TilesPerBlock; it++)
        {
            for (int ot = 0; ot < TilesPerBlock; ot++)
            {
                long tileIndex = (long)(ib * TilesPerBlock + it) * outTiles + ob * TilesPerBlock + ot;
                DecodeTile(new ReadOnlySpan<byte>(trellis + tileIndex * TileBytes, TileBytes), tile);
                for (int p = 0; p < Tile * Tile; p++)
                {
                    (int r, int c) = TilePosition(p);
                    buf[(ot * Tile + c) * Block + it * Tile + r] = (float)tile[p];
                }
            }
        }
    }

    // In-place W = diag(suh) H (block) diag(svh) on buf[ol * 128 + il]; the operation order is the contract the CUDA kernel repeats.
    private static unsafe void RotateBlock(float[] buf, ushort* suh, ushort* svh)
    {
        for (int ol = 0; ol < Block; ol++)
        {
            Span<float> row = buf.AsSpan(ol * Block, Block);
            ButterflyContiguous(row);
            for (int il = 0; il < Block; il++)
                row[il] = row[il] * Exl3Format.HadamardScale * (float)BitConverter.UInt16BitsToHalf(suh[il]);
        }
        for (int h = 1; h < Block; h <<= 1)
        {
            for (int j = 0; j < Block; j++)
            {
                if ((j & h) != 0) continue;
                Span<float> a = buf.AsSpan(j * Block, Block), b = buf.AsSpan((j + h) * Block, Block);
                for (int i = 0; i < Block; i += Vector<float>.Count)
                {
                    Vector<float> va = new(a[i..]), vb = new(b[i..]);
                    (va + vb).CopyTo(a[i..]);
                    (va - vb).CopyTo(b[i..]);
                }
            }
        }
        for (int ol = 0; ol < Block; ol++)
        {
            float sv = (float)BitConverter.UInt16BitsToHalf(svh[ol]);
            Span<float> row = buf.AsSpan(ol * Block, Block);
            for (int il = 0; il < Block; il++)
                row[il] = row[il] * Exl3Format.HadamardScale * sv;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ButterflyContiguous(Span<float> x)
    {
        for (int h = 1; h < Block; h <<= 1)
        {
            for (int j = 0; j < Block; j++)
            {
                if ((j & h) != 0) continue;
                float a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
}
