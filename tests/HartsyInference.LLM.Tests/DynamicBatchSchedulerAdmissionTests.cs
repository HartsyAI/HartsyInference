using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Admission on <see cref="DynamicBatchScheduler"/>: requests that do not fit the KV pool together wait their turn in FIFO order instead of failing; a full waiting
/// queue refuses the next submission; a waiting request is told its place in the queue. Each request that runs is checked against its solo pipeline run, so waiting
/// cannot change an answer.</summary>
public sealed class DynamicBatchSchedulerAdmissionTests
{
    /// <summary>The solo pipeline run of one prompt on <paramref name="model"/>: the answer waiting and batching must reproduce.</summary>
    private static string Solo(GenericTransformer model, IBackend backend, int[] prompt, int maxTokens)
    {
        GenerationResult result = new TextGenerationPipeline(model, new DynamicBatchSchedulerTests.StubTokenizer(), backend)
            .Generate(DynamicBatchSchedulerTests.Req(prompt, maxTokens, seed: 0));
        return string.Join(",", result.TokenIds);
    }

    [Fact]
    public async Task Requests_That_Do_Not_Fit_Together_Wait_Their_Turn_And_Match_Their_Solo_Runs()
    {
        TransformerConfig cfg = DynamicBatchSchedulerTests.Cfg();
        Dictionary<string, Tensor> w = DynamicBatchSchedulerTests.Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");

        // Each request needs ceil((3 + 6 + 1) / 4) = 3 pages of a 4-page pool, so only one can hold its reservation at a time.
        using PagedKvPool pool = new(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, pageSize: 4, maxPages: 4);
        using DynamicBatchScheduler scheduler = new(model, new DynamicBatchSchedulerTests.StubTokenizer(), backend, pool);
        int[][] prompts = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        string[] expected = [.. prompts.Select(p => Solo(model, backend, p, maxTokens: 6))];

        Task<GenerationResult>[] tasks = [.. prompts.Select(p => scheduler.SubmitAsync(DynamicBatchSchedulerTests.Req(p, 6, seed: 0), null, CancellationToken.None))];
        GenerationResult[] results = await Task.WhenAll(tasks).WaitAsync(TimeSpan.FromSeconds(60));

        for (int i = 0; i < prompts.Length; i++)
            Assert.Equal(expected[i], string.Join(",", results[i].TokenIds));
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public async Task A_Full_Waiting_Queue_Refuses_The_Next_Submission_With_SchedulerQueueFull()
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
        using DynamicBatchScheduler scheduler = new(held, new DynamicBatchSchedulerTests.StubTokenizer(), pool, maxQueued: 1);

        Task<GenerationResult> admitted = scheduler.SubmitAsync(DynamicBatchSchedulerTests.Req([1, 2, 3], 4, seed: 0), null, CancellationToken.None);
        Assert.True(entered.Wait(TimeSpan.FromSeconds(30)), "the first prefill never started");
        // The admitted request has left the waiting queue, so one more may wait; the one after that is refused at once.
        Task<GenerationResult> waiting = scheduler.SubmitAsync(DynamicBatchSchedulerTests.Req([4, 5], 4, seed: 0), null, CancellationToken.None);
        Task<GenerationResult> refused = scheduler.SubmitAsync(DynamicBatchSchedulerTests.Req([6], 4, seed: 0), null, CancellationToken.None);

        await Assert.ThrowsAsync<SchedulerQueueFullException>(() => refused).WaitAsync(TimeSpan.FromSeconds(30));
        release.Set();
        await admitted.WaitAsync(TimeSpan.FromSeconds(60));
        await waiting.WaitAsync(TimeSpan.FromSeconds(60));
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public async Task A_Waiting_Request_Is_Told_Its_Place_In_The_Queue()
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

        List<int> places = [];
        GenerationRequest second = DynamicBatchSchedulerTests.Req([4, 5], 4, seed: 0) with { OnQueued = place => places.Add(place) };
        GenerationRequest third = DynamicBatchSchedulerTests.Req([6, 7], 4, seed: 0) with { OnQueued = place => places.Add(place) };
        Task<GenerationResult> b = scheduler.SubmitAsync(second, null, CancellationToken.None);
        Task<GenerationResult> c = scheduler.SubmitAsync(third, null, CancellationToken.None);

        release.Set();
        await Task.WhenAll(admitted, b, c).WaitAsync(TimeSpan.FromSeconds(60));
        Assert.Equal([1, 2], places);
        foreach (Tensor t in w.Values) t.Dispose();
    }
}
