using System.Numerics;
using HartsyInference.Core.Numerics;

namespace HartsyInference.Audio.Preprocessing;

/// <summary>A fixed partition of a frame axis into blocks for <see cref="CpuParallel.For"/>. The block size is a
/// function of the per-frame cost only — never of the core count or <c>numerics.cpuThreads</c> — so which frames
/// share a block, and therefore every output bit, is the same on any machine, at any cap and inside
/// <see cref="CpuParallel.EnterInline"/>.</summary>
internal static class FramePartition
{
    /// <summary>Per-block budget for per-frame transforms, in transform points: a few hundred small frames or a
    /// handful of large ones per block, enough blocks to balance and few enough that dispatch stays negligible.</summary>
    public const int TransformBudget = 8192;

    /// <summary>Frames per block when each frame costs <paramref name="perFrame"/> units of <paramref name="budget"/>.</summary>
    public static int FramesPerBlock(int perFrame, int budget) => Math.Max(1, budget / Math.Max(1, perFrame));

    /// <summary>Number of blocks of <paramref name="framesPerBlock"/> covering <paramref name="frames"/>.</summary>
    public static int BlockCount(int frames, int framesPerBlock) => (frames + framesPerBlock - 1) / framesPerBlock;

    /// <summary>Scalar-operation estimate of one <paramref name="nFft"/>-point transform (direct DFT below 64 points,
    /// radix-2 or Bluestein above), used only against <see cref="CpuParallel.MinWorkForParallel"/>.</summary>
    public static long TransformWork(int nFft) =>
        nFft < 64 ? 4L * nFft * nFft : 5L * nFft * (BitOperations.Log2((uint)nFft) + 1);
}
