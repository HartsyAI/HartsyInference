using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Draining a scheduler (what unloading a model does first): the requests waiting for admission fail at once, nothing new is admitted, and the sequence
/// already decoding runs on to its end.</summary>
public sealed class DynamicBatchSchedulerDrainTests
{
    [Fact]
    public async Task CancelQueued_Fails_The_Waiting_Requests_And_Lets_The_Admitted_One_Finish()
    {
        TransformerConfig cfg = DynamicBatchSchedulerTests.Cfg();
        Dictionary<string, Tensor> w = DynamicBatchSchedulerTests.Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        using ManualResetEventSlim entered = new(false);
        using ManualResetEventSlim release = new(false);
        DynamicBatchSchedulerShutdownTests.HeldPrefillModel held = new(new GenericTransformerModel(model, backend), entered, release);
        using PagedKvPool pool = new(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, pageSize: 4, maxPages: 64);
        using DynamicBatchScheduler scheduler = new(held, new DynamicBatchSchedulerTests.StubTokenizer(), pool);

        Task<GenerationResult> admitted = scheduler.SubmitAsync(DynamicBatchSchedulerTests.Req([1, 2, 3], 4, seed: 0), null, CancellationToken.None);
        Assert.True(entered.Wait(TimeSpan.FromSeconds(30)), "the first prefill never started");
        Task<GenerationResult> waitingB = scheduler.SubmitAsync(DynamicBatchSchedulerTests.Req([4, 5], 4, seed: 0), null, CancellationToken.None);
        Task<GenerationResult> waitingC = scheduler.SubmitAsync(DynamicBatchSchedulerTests.Req([6, 7], 4, seed: 0), null, CancellationToken.None);
        Assert.Equal(2, scheduler.QueuedCount);

        scheduler.CancelQueued();

        await Assert.ThrowsAsync<SchedulerStoppedException>(() => waitingB).WaitAsync(TimeSpan.FromSeconds(30));
        await Assert.ThrowsAsync<SchedulerStoppedException>(() => waitingC).WaitAsync(TimeSpan.FromSeconds(30));
        Assert.Equal(0, scheduler.QueuedCount);
        Assert.False(admitted.IsCompleted, "the admitted request should still be inside its prefill");

        // Once draining has begun, nothing new is taken.
        await Assert.ThrowsAsync<SchedulerStoppedException>(() =>
            scheduler.SubmitAsync(DynamicBatchSchedulerTests.Req([8], 4, seed: 0), null, CancellationToken.None));

        release.Set();
        GenerationResult result = await admitted.WaitAsync(TimeSpan.FromSeconds(60));
        Assert.True(result.TokenIds.Count <= 4);
        foreach (Tensor t in w.Values) t.Dispose();
    }
}
