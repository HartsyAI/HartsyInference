namespace HartsyInference.Core.Memory.Vram.Streaming;

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

    /// <summary>The prefix that fits once the streaming window is set aside as well as <paramref name="headroomBytes"/>. The window is
    /// <c>prefetchAhead + 1</c> streamed blocks: <see cref="BlockStreamingController.Prime"/> uploads the first of them right after the
    /// placement, so a prefix sized without it leaves the device short by the window (the FastFrames L run: 4171 MiB free, 4996 MiB
    /// needed). The window is sized at the largest block, so it is an upper bound; when every block fits there is no window and the
    /// prefix is the whole denoiser.</summary>
    /// <param name="freeBytes">Free device memory, read AFTER a pool trim.</param>
    public static int SizeWithStreamingWindow(long freeBytes, long headroomBytes, ReadOnlySpan<long> blockBytes, int prefetchAhead)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(prefetchAhead);
        int all = Size(freeBytes, headroomBytes, blockBytes);
        if (all == blockBytes.Length)
            return all;
        long widest = 0;
        foreach (long bytes in blockBytes)
            widest = Math.Max(widest, bytes);
        long window = (prefetchAhead + 1) * widest;
        return Size(freeBytes, headroomBytes + window, blockBytes);
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
