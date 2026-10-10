using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Shutdown of <see cref="DynamicBatchScheduler"/>: a request still waiting in the queue, and one admitted but not finished, both complete with
/// <see cref="SchedulerStoppedException"/> when the scheduler is disposed, so no caller waits on a loop that will not run. The first prefill is held open, so
/// the scheduler is inside admission with two requests queued behind it when Dispose runs.</summary>
public sealed class DynamicBatchSchedulerShutdownTests
{
    /// <summary>Forwards to a real model, but holds the first prefill until released and signals once it has entered.</summary>
    internal sealed class HeldPrefillModel(IGenerationModel inner, ManualResetEventSlim entered, ManualResetEventSlim release) : IGenerationModel
    {
        public GenerationModelInfo Info => inner.Info;
        public GenerationCapabilities Capabilities => inner.Capabilities;
        public IBackend OutputBackend => inner.OutputBackend;
        public ISequenceState CreateSequenceState(SequenceStateOptions options) => inner.CreateSequenceState(options);

        public Tensor Prefill(in PrefillChunk chunk, ISequenceState state)
        {
            entered.Set();
            release.Wait();
            return inner.Prefill(chunk, state);
        }

        public Tensor Prefill(in PrefillChunk chunk, ISequenceState state, CancellationToken cancel)
        {
            entered.Set();
            release.Wait();
            return inner.Prefill(chunk, state, cancel);
        }

        public Tensor DecodeBatch(ReadOnlySpan<int> tokenIds, ISequenceState[] states) => inner.DecodeBatch(tokenIds, states);
        public Tensor ProjectLogits(Tensor hidden, int rows) => inner.ProjectLogits(hidden, rows);
        public IEnumerable<Tensor> EnumerateWeights(bool includeRedundantSplits) => inner.EnumerateWeights(includeRedundantSplits);
        public long EstimateSequenceBytes(int contextTokens) => inner.EstimateSequenceBytes(contextTokens);
        public CapacitySnapshot Capacity() => inner.Capacity();
        public void Dispose() => inner.Dispose();
    }

    [Fact]
    public async Task Dispose_FailsQueuedAndAdmittedRequests_WithSchedulerStopped()
    {
        TransformerConfig cfg = DynamicBatchSchedulerTests.Cfg();
        Dictionary<string, Tensor> w = DynamicBatchSchedulerTests.Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        using ManualResetEventSlim entered = new(false);
        using ManualResetEventSlim release = new(false);
        HeldPrefillModel held = new(new GenericTransformerModel(model, backend), entered, release);
        using PagedKvPool pool = new(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, pageSize: 4, maxPages: 64);
        DynamicBatchScheduler scheduler = new(held, new DynamicBatchSchedulerTests.StubTokenizer(), pool);

        Task<GenerationResult> admitted = scheduler.SubmitAsync(DynamicBatchSchedulerTests.Req([1, 2, 3], 6, seed: 0), null, CancellationToken.None);
        Assert.True(entered.Wait(TimeSpan.FromSeconds(30)), "the first prefill never started");
        Task<GenerationResult> queuedB = scheduler.SubmitAsync(DynamicBatchSchedulerTests.Req([4], 6, seed: 0), null, CancellationToken.None);
        Task<GenerationResult> queuedC = scheduler.SubmitAsync(DynamicBatchSchedulerTests.Req([5, 6], 6, seed: 0), null, CancellationToken.None);

        Task disposing = Task.Run(scheduler.Dispose);
        // The queued requests fail as soon as shutdown starts, while the loop is still inside the held prefill.
        await AssertStopped(queuedB);
        await AssertStopped(queuedC);
        Assert.False(admitted.IsCompleted, "the admitted request should still be inside its prefill");

        release.Set();
        await AssertStopped(admitted);
        await disposing.WaitAsync(TimeSpan.FromSeconds(30));
        foreach (Tensor t in w.Values) t.Dispose();
    }

    private static async Task AssertStopped(Task<GenerationResult> task)
    {
        Task finished = await Task.WhenAny(task, Task.Delay(TimeSpan.FromSeconds(30)));
        Assert.Same(task, finished);
        await Assert.ThrowsAsync<SchedulerStoppedException>(() => task);
    }
}
