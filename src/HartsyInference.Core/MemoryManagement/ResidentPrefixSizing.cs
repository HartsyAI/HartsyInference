namespace HartsyInference.Core.MemoryManagement;

/// <summary>The resident-prefix sizing arithmetic, isolated so it can be swept on CPU against the hand-rolled formula it replaces.</summary>
public static class ResidentPrefixSizing
{
    /// <summary>Whole blocks that fit once <paramref name="headroomBytes"/> is set aside, clamped to <c>[0, blockCount]</c>.</summary>
    /// <param name="freeBytes">Free device memory, read AFTER a pool trim — untrimmed slack counts as used and under-sizes the prefix.</param>
    public static int Size(long freeBytes, long headroomBytes, long blockBytes, int blockCount)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        long spendable = freeBytes - headroomBytes;
        return (int)Math.Clamp(spendable / Math.Max(blockBytes, 1), 0, blockCount);
    }

    /// <summary>The leading blocks whose summed footprint fits once <paramref name="headroomBytes"/> is set aside. Each block
    /// counts at its own size, so a denoiser with wide and narrow blocks (HunyuanVideo's double-stream blocks beside its
    /// single-stream tail) is not sized as if every block were the widest one. The prefix is contiguous: it stops at the
    /// first block that does not fit.</summary>
    /// <param name="freeBytes">Free device memory, read AFTER a pool trim — untrimmed slack counts as used and under-sizes the prefix.</param>
    public static int Size(long freeBytes, long headroomBytes, ReadOnlySpan<long> blockBytes)
    {
        long spendable = freeBytes - headroomBytes;
        long used = 0;
        int prefix = 0;
        while (prefix < blockBytes.Length && used + blockBytes[prefix] <= spendable)
        {
            used += blockBytes[prefix];
            prefix++;
        }
        return prefix;
    }
}
