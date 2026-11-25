using HartsyInference.Core.Numerics;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>The odd multipliers an Engram layer mixes its n-gram token ids with, drawn as the reference does (<c>draw_hash_multipliers</c>):
/// NumPy's default generator seeded with <c>engram_seed + 10007 * layer_id</c>, bounded so a product with any compressed id stays within int64.</summary>
public static class Hy4FlashEngramMultipliers
{
    /// <summary>Draws <paramref name="count"/> multipliers for Engram layer id <paramref name="layerId"/> (the config id, the decoder index plus one).</summary>
    /// <param name="seed"><c>engram_seed</c>.</param>
    /// <param name="layerId">Config layer id.</param>
    /// <param name="compressedVocab"><c>max(lookup_table) + 1</c>: the size of the compressed vocabulary the ids are mapped into.</param>
    /// <param name="count"><c>engram_max_ngram_size</c>.</param>
    public static long[] Draw(int seed, int layerId, int compressedVocab, int count)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(compressedVocab, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(count, 1);
        long halfBound = Math.Max(1L, long.MaxValue / compressedVocab / 2);
        NumpyPcg64 bits = new(checked((uint)(seed + 10007L * layerId)));
        long[] multipliers = new long[count];
        for (int i = 0; i < count; i++) multipliers[i] = bits.NextInt64(0, halfBound) * 2 + 1;
        return multipliers;
    }
}
