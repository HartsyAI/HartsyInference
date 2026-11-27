using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Models.MossVl;
using HartsyInference.LLM.Multimodal;

namespace HartsyInference.LLM.Tests;

/// <summary>Segment planning and the appendable cross-attention cache that let a MOSS-VL stream grow one segment at a time
/// (video frames, live prompts, decode steps) with the same positions and visibility as one whole prompt.</summary>
public sealed class MossVlSegmentTests
{
    private const int Img = 151655;

    [Fact]
    public void SegmentsPlannedOneAfterAnotherMatchTheWholePrompt()
    {
        // [a, img(2x3), b, c, img(2x2), d] as one prompt, and as two segments split after c.
        int[] whole = [10, Img, 11, 12, Img, 13];
        MossVlImage[] images = [new(2, 3), new(2, 2)];
        MossVlPositionPlan all = MossVlPositions.Plan(whole, Img, images);

        MossVlPositionPlan first = MossVlPositions.PlanSegment(whole.AsSpan(0, 4), Img, images.AsSpan(0, 1), 0, 0);
        MossVlPositionPlan second = MossVlPositions.PlanSegment(whole.AsSpan(4), Img, images.AsSpan(1), first.NextPosition, first.VisionLength);

        int[] Axis(MossVlPositionPlan p, int axis, int count) => p.Text.Skip(axis * count).Take(count).ToArray();
        for (int axis = 0; axis < 3; axis++)
            Assert.Equal(Axis(all, axis, 6), Axis(first, axis, 4).Concat(Axis(second, axis, 2)).ToArray());
        // The second segment's vision columns are the whole prompt's tail.
        int len1 = first.VisionLength, len2 = second.VisionLength;
        Assert.Equal(all.VisionLength, len1 + len2);
        for (int axis = 0; axis < 3; axis++)
            Assert.Equal(all.Vision.Skip(axis * all.VisionLength + len1).Take(len2), second.Vision.Skip(axis * len2).Take(len2));
        Assert.Equal(all.NextPosition, second.NextPosition);
    }

    [Fact]
    public void NextPositionFollowsTheLastToken()
    {
        // Text ends the prompt: next = last + 1. An image ends it: next = its placeholder position + 1.
        Assert.Equal(3, MossVlPositions.Plan([1, 2, 3], Img, []).NextPosition);
        MossVlPositionPlan endsOnImage = MossVlPositions.Plan([1, Img], Img, [new MossVlImage(2, 4)]);
        Assert.Equal(1 + 4, endsOnImage.Text[1]);
        Assert.Equal(1 + 4 + 1, endsOnImage.NextPosition);
    }

    [Fact]
    public void NewFramesAfterPriorVisionAreGroupedByWhatEachRowSees()
    {
        // 12 vision tokens already cached; the segment is [x, img(1x1), y, img(1x1), z]: each block is 1 token + separator.
        MossVlPositionPlan plan = MossVlPositions.PlanSegment([7, Img, 8, Img, 9], Img, [new(1, 1), new(1, 1)], startPosition: 40,
            priorVisionLength: 12);

        Assert.Equal(4, plan.VisionLength);
        Assert.Null(plan.RowMask);   // every row sees the prior vision
        IReadOnlyList<MossVlCrossGroup> groups = plan.CrossGroups!;
        Assert.Equal(new[] { 12, 14, 16 }, groups.Select(g => g.KvLength).ToArray());
        Assert.Equal(new float[] { 1, 0, 0, 0, 0 }, groups[0].RowSelect);
        Assert.Equal(new float[] { 0, 1, 1, 0, 0 }, groups[1].RowSelect);
        Assert.Equal(new float[] { 0, 0, 0, 1, 1 }, groups[2].RowSelect);
        Assert.Equal(40, plan.Text[0]);
    }

    [Fact]
    public void APromptOrStepAfterVisionAttendsAllOfItWithoutMasksOrGroups()
    {
        MossVlPositionPlan prompt = MossVlPositions.PlanSegment([1, 2, 3], Img, [], startPosition: 50, priorVisionLength: 20);
        Assert.Null(prompt.RowMask);
        Assert.Null(prompt.CrossGroups);
        Assert.Equal(0, prompt.VisionLength);
        Assert.Equal(53, prompt.NextPosition);

        MossVlPositionPlan step = MossVlPositions.PlanSegment([4], Img, [], startPosition: 53, priorVisionLength: 20);
        Assert.Equal(new[] { 53, 53, 53 }, step.Text);
    }

    [Fact]
    public void TextBeforeAnyVisionIsMaskedAndSeesNothing()
    {
        MossVlPositionPlan plan = MossVlPositions.PlanSegment([1, 2], Img, [], 0, 0);
        Assert.Null(plan.RowMask);   // no vision anywhere: the cross layers are skipped, not masked
        Assert.Null(plan.CrossGroups);

        MossVlPositionPlan firstImage = MossVlPositions.PlanSegment([1, Img, 2], Img, [new(1, 1)], 0, 0);
        Assert.Equal(new float[] { 0, 1, 1 }, firstImage.RowMask);
        Assert.Null(firstImage.CrossGroups);   // the rows that see anything all see the whole sequence
    }

    [Fact]
    public void CrossCacheAppendsGrowsAndOnlyCommitsWhatACallFinished()
    {
        using CpuBackend backend = new();
        using CrossKvCache cache = new(layers: 4, initialCapacity: 2);

        using (Tensor k = Rows(heads: 2, n: 2, headDim: 3, start: 0)) cache.Append(backend, 1, k, k);
        cache.Commit(2);
        using (Tensor k = Rows(heads: 2, n: 3, headDim: 3, start: 100)) cache.Append(backend, 1, k, k);   // past capacity 2: grows
        cache.Commit(3);
        Assert.Equal(5, cache.Length);

        (Tensor K, Tensor V) buffers = cache.Get(1)!.Value;
        Assert.True(buffers.K.Shape[2] >= 5);
        // Head 1, row 3 is the second appended segment's row 1: value 100 + 1*3*... see Rows.
        Assert.Equal(Value(start: 100, head: 1, row: 1, col: 2, n: 3), At(buffers.K, head: 1, row: 3, col: 2));
        Assert.Equal(Value(start: 0, head: 0, row: 1, col: 0, n: 2), At(buffers.K, head: 0, row: 1, col: 0));
        Assert.Null(cache.Get(0));

        // A call that appends but is not committed leaves Length alone; the next append overwrites its rows.
        using (Tensor k = Rows(heads: 2, n: 1, headDim: 3, start: 500)) cache.Append(backend, 1, k, k);
        Assert.Equal(5, cache.Length);
        using (Tensor k = Rows(heads: 2, n: 1, headDim: 3, start: 900)) cache.Append(backend, 1, k, k);
        cache.Commit(1);
        Assert.Equal(Value(start: 900, head: 0, row: 0, col: 0, n: 1), At(cache.Get(1)!.Value.K, head: 0, row: 5, col: 0));
    }

    private static float Value(int start, int head, int row, int col, int n) => start + (head * n + row) * 10 + col;

    /// <summary>A head-major <c>[1, heads, n, headDim]</c> F32 tensor whose element (h, r, c) is <see cref="Value"/>.</summary>
    private static Tensor Rows(int heads, int n, int headDim, int start)
    {
        Tensor t = new(new TensorShape(1, heads, n, headDim), DType.F32);
        Span<float> s = t.AsSpan<float>();
        for (int h = 0; h < heads; h++)
            for (int r = 0; r < n; r++)
                for (int c = 0; c < headDim; c++)
                    s[(h * n + r) * headDim + c] = Value(start, h, r, c, n);
        return t;
    }

    private static float At(Tensor buffer, int head, int row, int col)
    {
        int capacity = (int)buffer.Shape[2], headDim = (int)buffer.Shape[3];
        return buffer.AsSpan<float>()[(head * capacity + row) * headDim + col];
    }
}
