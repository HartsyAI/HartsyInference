using System.Numerics;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.ThreeD.Models.Utonia;

/// <summary>Space-filling-curve keys and the sort permutations that order voxels for serialized attention.</summary>
public static class UtoniaSerialization
{
    /// <summary>Deepest coordinate grid the keys support, matching Pointcept's 16-bit limit.</summary>
    public const int MaxDepth = 16;

    /// <summary>Parses a checkpoint curve name.</summary>
    public static UtoniaSerialOrder ParseOrder(string name) => name switch
    {
        "z" => UtoniaSerialOrder.ZOrder,
        "z-trans" => UtoniaSerialOrder.ZOrderTrans,
        "hilbert" => UtoniaSerialOrder.Hilbert,
        "hilbert-trans" => UtoniaSerialOrder.HilbertTrans,
        _ => throw new HartsyInferenceException($"Unknown Utonia serialization order '{name}'.")
    };

    /// <summary>Morton key: bit i of x, y and z lands at positions 3i+2, 3i+1 and 3i.</summary>
    public static long ZOrderKey(int x, int y, int z, int depth)
    {
        long key = 0;
        for (int i = 0; i < depth; i++)
        {
            key |= (long)((x >> i) & 1) << (3 * i + 2);
            key |= (long)((y >> i) & 1) << (3 * i + 1);
            key |= (long)((z >> i) & 1) << (3 * i);
        }

        return key;
    }

    /// <summary>
    /// Hilbert key over <paramref name="numBits"/> bits per axis. Ported from Pointcept's bit-array transform so the
    /// curve orientation and bit order match its output; a Skilling-form reimplementation would differ.
    /// </summary>
    public static long HilbertKey(int x, int y, int z, int numBits)
    {
        Span<int> g = stackalloc int[3 * MaxDepth];
        int nb = numBits;
        for (int b = 0; b < nb; b++)
        {
            int shift = nb - 1 - b;
            g[b] = (x >> shift) & 1;
            g[nb + b] = (y >> shift) & 1;
            g[2 * nb + b] = (z >> shift) & 1;
        }

        for (int bit = 0; bit < nb; bit++)
        {
            for (int dim = 0; dim < 3; dim++)
            {
                int mask = g[dim * nb + bit];
                for (int k = bit + 1; k < nb; k++)
                {
                    g[k] ^= mask;
                }

                for (int k = bit + 1; k < nb; k++)
                {
                    int flip = mask == 0 && g[k] != g[dim * nb + k] ? 1 : 0;
                    g[dim * nb + k] ^= flip;
                    g[k] ^= flip;
                }
            }
        }

        // Gray-to-binary is a prefix XOR over the bit-major sequence; bit j of that sequence sits at 3nb-1-j.
        long code = 0;
        int acc = 0;
        int total = 3 * nb;
        for (int bit = 0; bit < nb; bit++)
        {
            for (int dim = 0; dim < 3; dim++)
            {
                int j = bit * 3 + dim;
                acc ^= g[dim * nb + bit];
                code |= (long)acc << (total - 1 - j);
            }
        }

        return code;
    }

    /// <summary>Key of one voxel under a curve, without the batch bits.</summary>
    public static long Key(UtoniaSerialOrder order, int x, int y, int z, int depth) => order switch
    {
        UtoniaSerialOrder.ZOrder => ZOrderKey(x, y, z, depth),
        UtoniaSerialOrder.ZOrderTrans => ZOrderKey(y, x, z, depth),
        UtoniaSerialOrder.Hilbert => HilbertKey(x, y, z, depth),
        _ => HilbertKey(y, x, z, depth)
    };

    /// <summary>Bit length of a non-negative value, as Python's int.bit_length.</summary>
    public static int BitLength(int value) => 32 - BitOperations.LeadingZeroCount((uint)value);

    /// <summary>
    /// Serializes a point set. The depth is the bit length of the largest grid coordinate plus one, and the batch index
    /// sits above the coordinate bits so each sample forms one contiguous run. Ties keep input order.
    /// </summary>
    public static UtoniaSerialized Serialize(ReadOnlySpan<int> gridCoord, ReadOnlySpan<int> batch, UtoniaSerialOrder[] orders)
    {
        int n = batch.Length;
        if (gridCoord.Length != n * 3)
        {
            throw new HartsyInferenceException($"Utonia serialization needs {n * 3} grid values, got {gridCoord.Length}.");
        }

        int max = 0, maxBatch = 0;
        for (int i = 0; i < n; i++)
        {
            for (int a = 0; a < 3; a++)
            {
                int v = gridCoord[i * 3 + a];
                if (v < 0)
                {
                    throw new HartsyInferenceException("Utonia grid coordinates must be non-negative.");
                }

                max = Math.Max(max, v);
            }

            if (batch[i] < 0)
            {
                throw new HartsyInferenceException("Utonia batch indices must be non-negative.");
            }

            maxBatch = Math.Max(maxBatch, batch[i]);
        }

        int depth = BitLength(max + 1);
        if (depth > MaxDepth || 3 * depth + BitLength(maxBatch) > 63)
        {
            throw new HartsyInferenceException($"Utonia grid depth {depth} exceeds the serialization key width.");
        }

        int[][] order = new int[orders.Length][];
        int[][] inverse = new int[orders.Length][];
        for (int k = 0; k < orders.Length; k++)
        {
            long[] keys = new long[n];
            for (int i = 0; i < n; i++)
            {
                keys[i] = Key(orders[k], gridCoord[i * 3], gridCoord[i * 3 + 1], gridCoord[i * 3 + 2], depth)
                    | ((long)batch[i] << (3 * depth));
            }

            int[] perm = StableArgSort(keys);
            int[] inv = new int[n];
            for (int j = 0; j < n; j++)
            {
                inv[perm[j]] = j;
            }

            order[k] = perm;
            inverse[k] = inv;
        }

        return new UtoniaSerialized(depth, order, inverse);
    }

    /// <summary>Ascending argsort that keeps the input order among equal keys.</summary>
    internal static int[] StableArgSort(long[] keys)
    {
        int n = keys.Length;
        long[] sorted = (long[])keys.Clone();
        int[] perm = new int[n];
        for (int i = 0; i < n; i++)
        {
            perm[i] = i;
        }

        Array.Sort(sorted, perm);
        for (int start = 0; start < n;)
        {
            int end = start + 1;
            while (end < n && sorted[end] == sorted[start])
            {
                end++;
            }

            if (end - start > 1)
            {
                Array.Sort(perm, start, end - start);
            }

            start = end;
        }

        return perm;
    }
}
