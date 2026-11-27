using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Streaming;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary><see cref="FrameTailCache"/> and <see cref="CausalStreamCache"/> bookkeeping. Every frame carries its global index,
/// so the frames a tail holds are readable. Upstream's own cached ops are checked in <see cref="SanaWmStreamingCacheTests"/>.</summary>
public sealed class FrameTailCacheTests
{
    [Theory]
    [InlineData(new[] { 4, 3, 3 }, new[] { 7, 8, 9 })]
    [InlineData(new[] { 2, 1 }, new[] { 0, 1, 2 })]
    [InlineData(new[] { 1, 1, 1, 1, 1 }, new[] { 2, 3, 4 })]
    [InlineData(new[] { 1 }, new[] { 0 })]
    [InlineData(new[] { 5, 1 }, new[] { 3, 4, 5 })]
    public void Tail_HoldsTheNewestFramesOfTheStream(int[] chunks, int[] expected)
    {
        using CpuBackend cpu = new();
        // Frames run along axis 0 as rows of two tokens, so a tail cut on a token boundary would show up as a split index.
        using FrameTailCache tail = new(cpu, tailFrames: 3, axis: 0, positionsPerFrame: 2);
        int next = 0;
        foreach (int frames in chunks)
        {
            using Tensor chunk = Frames(next, frames, tokensPerFrame: 2);
            tail.Stage(chunk);
            tail.Commit();
            next += frames;
        }

        Assert.Equal(expected.Length, tail.HeldFrames);
        Assert.Equal(expected, FrameIndices(tail.Current!, tokensPerFrame: 2));
    }

    [Fact]
    public void Tail_JoinsOnlyAtCommit_AndDiscardKeepsTheCommittedTail()
    {
        using CpuBackend cpu = new();
        using FrameTailCache tail = new(cpu, tailFrames: 2, axis: 0);
        Assert.Null(tail.Current);
        using (Tensor first = Frames(0, 3, 1))
            tail.Stage(first);
        Assert.Null(tail.Current);
        Assert.Equal(0, tail.HeldFrames);
        tail.Commit();
        Assert.Equal([1, 2], FrameIndices(tail.Current!, 1));

        using (Tensor second = Frames(3, 3, 1))
            tail.Stage(second);
        tail.Discard();
        tail.Commit();
        Assert.Equal([1, 2], FrameIndices(tail.Current!, 1));

        tail.Reset();
        Assert.Null(tail.Current);
        Assert.Equal(0, tail.HeldFrames);
    }

    [Fact]
    public void ReplicatePadded_RepeatsTheOldestFrameInFront()
    {
        using CpuBackend cpu = new();
        using FrameTailCache tail = new(cpu, tailFrames: 3, axis: 0);
        Assert.Null(tail.ReplicatePadded(2));
        using (Tensor first = Frames(0, 2, 1))
            tail.Stage(first);
        tail.Commit();

        using Tensor padded = tail.ReplicatePadded(4)!;
        Assert.Equal([0, 0, 0, 1], FrameIndices(padded, 1));
        using Tensor trimmed = tail.ReplicatePadded(1)!;
        Assert.Equal([1], FrameIndices(trimmed, 1));
    }

    [Fact]
    public void Stage_RefusesPartialFrames()
    {
        using CpuBackend cpu = new();
        using FrameTailCache tail = new(cpu, tailFrames: 3, axis: 0, positionsPerFrame: 2);
        using Tensor odd = new(new TensorShape(3, 1), DType.F32);
        Assert.Throws<ArgumentException>(() => tail.Stage(odd));
        using Tensor wrongAxis = new(new TensorShape(4), DType.F32);
        using FrameTailCache onAxis1 = new(cpu, tailFrames: 3, axis: 1);
        Assert.Throws<ArgumentException>(() => onAxis1.Stage(wrongAxis));
    }

    [Fact]
    public void CausalStreamCache_HandsOutTailsInTraversalOrder()
    {
        using CpuBackend cpu = new();
        using CausalStreamCache cache = new(cpu, frameAxis: 0);
        cache.BeginPass();
        FrameTailCache a = cache.NextTail(2), b = cache.NextTail(1);
        Assert.True(cache.NextFirstUse());
        using (Tensor chunk = Frames(0, 3, 1))
        {
            a.Stage(chunk);
            b.Stage(chunk);
        }

        // A second pass over the same chunk (not committed yet) still sees an empty cache and a first use.
        cache.BeginPass();
        Assert.Same(a, cache.NextTail(2));
        Assert.Same(b, cache.NextTail(1));
        Assert.True(cache.NextFirstUse());
        Assert.Null(a.Current);
        cache.Commit();
        Assert.Equal(1, cache.CommittedPasses);
        Assert.Equal([1, 2], FrameIndices(a.Current!, 1));
        Assert.Equal([2], FrameIndices(b.Current!, 1));

        cache.BeginPass();
        cache.NextTail(2);
        Assert.Throws<InvalidOperationException>(() => cache.NextTail(3));
        Assert.False(cache.NextFirstUse());
    }

    [Fact]
    public void CausalStreamCache_DiscardAndReset()
    {
        using CpuBackend cpu = new();
        using CausalStreamCache cache = new(cpu, frameAxis: 0);
        cache.BeginPass();
        FrameTailCache tail = cache.NextTail(2);
        Assert.True(cache.NextFirstUse());
        using (Tensor chunk = Frames(0, 2, 1))
            tail.Stage(chunk);
        cache.Discard();
        cache.Commit();
        Assert.Null(tail.Current);
        cache.BeginPass();
        cache.NextTail(2);
        Assert.True(cache.NextFirstUse());   // the discarded use did not count
        cache.Commit();

        cache.Reset();
        Assert.Equal(0, cache.CommittedPasses);
        cache.BeginPass();
        Assert.NotSame(tail, cache.NextTail(2));
        Assert.True(cache.NextFirstUse());
    }

    // [frames * tokensPerFrame, 1] rows, every token of frame f holding start + f.
    private static Tensor Frames(int start, int frames, int tokensPerFrame)
    {
        Tensor tensor = new(new TensorShape((long)frames * tokensPerFrame, 1), DType.F32, DeviceKind.Cpu);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = start + i / tokensPerFrame;
        return tensor;
    }

    private static int[] FrameIndices(Tensor tensor, int tokensPerFrame)
    {
        ReadOnlySpan<float> values = tensor.AsReadOnlySpan<float>();
        int[] frames = new int[values.Length / tokensPerFrame];
        for (int f = 0; f < frames.Length; f++)
        {
            for (int t = 1; t < tokensPerFrame; t++)
                Assert.Equal(values[f * tokensPerFrame], values[f * tokensPerFrame + t]);
            frames[f] = (int)values[f * tokensPerFrame];
        }

        return frames;
    }
}
