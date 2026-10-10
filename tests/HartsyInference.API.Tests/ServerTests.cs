using HartsyInference.API;
using HartsyInference.Engine;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>Tests for the server's self-contained pieces: the inference queue's capacity/429 behaviour and the
/// KV-pool byte-budget sizing.</summary>
public sealed class ServerTests
{
    [Fact]
    public async Task InferenceQueue_RejectsWhenFull()
    {
        using InferenceQueue queue = new InferenceQueue(maxConcurrency: 1, maxQueueDepth: 0);

        // Occupy the single slot with a task that blocks until released.
        TaskCompletionSource gate = new TaskCompletionSource();
        Task<bool> running = queue.EnqueueAsync(async () => { await gate.Task; return true; }, CancellationToken.None);

        // Give the first task time to acquire the slot.
        await Task.Delay(50);

        // Capacity is concurrency(1) + depth(0) = 1; a second enqueue must be rejected.
        await Assert.ThrowsAsync<QueueFullException>(() =>
            queue.EnqueueAsync(() => Task.FromResult(true), CancellationToken.None));

        gate.SetResult();
        Assert.True(await running);
    }

    /// <summary>Regression test for a real incident: a FIXED page-count KV pool default (1024) comfortably
    /// fit a small model but eagerly pre-allocated ~3.4GB for a model with wider KV heads/more layers,
    /// OOM-ing on a GPU that had ~4GB free. <see cref="PagedKvPool.PageCountForBudget"/> replaced the
    /// fixed count with a byte-budget-aware calculation — this locks in that the resulting page count
    /// actually respects the budget regardless of model shape, and that a larger-KV-dimension model
    /// correctly gets FEWER pages for the same budget (not the same fixed count that caused the OOM).</summary>
    [Theory]
    [InlineData(4, 26, 256, 512L * 1024 * 1024)]  // gemma-3-1b-ish: wider heads, fewer layers — the actual OOM case
    public void ComputeKvPoolPageCount_RespectsByteBudget(int numKvHeads, int numLayers, int headDim, long budget)
    {
        int[] headDimPerLayer = new int[numLayers];
        Array.Fill(headDimPerLayer, headDim);
        const int pageSize = 16;

        int pages = PagedKvPool.PageCountForBudget(numKvHeads, headDimPerLayer, pageSize, budget);

        long actualBytes = (long)pages * numLayers * numKvHeads * headDim * pageSize * sizeof(float) * 2;
        Assert.True(actualBytes <= budget, $"pool would use {actualBytes} bytes, over the {budget} budget");
        Assert.True(pages >= 8, "should never return fewer than the floor, even for a tiny budget");
    }

    [Fact]
    public void ComputeKvPoolPageCount_WiderModel_GetsFewerPagesForSameBudget()
    {
        const long budget = 512L * 1024 * 1024;
        int[] narrowHeadDims = Enumerable.Repeat(64, 24).ToArray();   // small model
        int[] wideHeadDims = Enumerable.Repeat(256, 26).ToArray();    // large-KV-dim model (the OOM case)

        int narrowPages = PagedKvPool.PageCountForBudget(2, narrowHeadDims, 16, budget);
        int widePages = PagedKvPool.PageCountForBudget(4, wideHeadDims, 16, budget);

        // The whole point of the fix: a fixed page count gave the SAME pages to both, over-allocating VRAM
        // for the wide model. The budget-aware version must give the wide model meaningfully fewer pages.
        Assert.True(widePages < narrowPages, $"wide model got {widePages} pages, narrow got {narrowPages} — expected fewer for the wider model");
    }
}
