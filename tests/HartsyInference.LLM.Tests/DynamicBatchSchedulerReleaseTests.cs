using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>A request on <see cref="DynamicBatchScheduler"/> completes only after its sequence has been released, however it ends. TextService ends a slot lease when
/// the request completes and may then free the model under the device gate; a release still pending at that point would wait on that gate. Each sequence state here holds
/// its release open, so a completion that came first is seen.</summary>
public sealed class DynamicBatchSchedulerReleaseTests
{
    /// <summary>How the request ends.</summary>
    public enum Ending
    {
        /// <summary>It reaches its token budget.</summary>
        Finished,

        /// <summary>Its caller cancels it after the first token.</summary>
        Cancelled,

        /// <summary>Its decode round throws.</summary>
        RoundFailed,
    }

    [Theory]
    [InlineData(Ending.Finished)]
    [InlineData(Ending.Cancelled)]
    [InlineData(Ending.RoundFailed)]
    public async Task A_Request_Completes_Only_After_Its_Sequence_Is_Released(Ending ending)
    {
        TransformerConfig cfg = DynamicBatchSchedulerTests.Cfg();
        Dictionary<string, Tensor> w = DynamicBatchSchedulerTests.Weights(cfg);
        try
        {
            using CpuBackend backend = new();
            using GenericTransformer model = new(cfg);
            model.LoadWeights(w, "model");
            using ManualResetEventSlim releasing = new(false);
            using ManualResetEventSlim release = new(false);
            HeldReleaseModel held = new(new GenericTransformerModel(model, backend), releasing, release);
            using PagedKvPool pool = new(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, pageSize: 4, maxPages: 16);
            using DynamicBatchScheduler scheduler = new(held, new NoStopTokenizer(), pool);
            if (ending == Ending.RoundFailed) scheduler.TestFaultInjector = _ => new InvalidOperationException("injected round failure");
            using CancellationTokenSource cancel = new();
            Action<int>? onToken = ending == Ending.Cancelled ? _ => cancel.Cancel() : null;

            Task<GenerationResult> request = scheduler.SubmitAsync(DynamicBatchSchedulerTests.Req([1, 2, 3], ending == Ending.Finished ? 3 : 12, seed: 0), onToken, cancel.Token);

            try
            {
                Assert.True(releasing.Wait(TimeSpan.FromSeconds(30)), "the sequence was never released");
                Assert.False(request.IsCompleted, "the request completed while its sequence was still being released");
            }
            finally
            {
                release.Set();
            }
            Task finished = await Task.WhenAny(request, Task.Delay(TimeSpan.FromSeconds(30)));
            Assert.Same(request, finished);
            switch (ending)
            {
                case Ending.Finished:
                    Assert.Equal(3, (await request).TokenIds.Count);
                    break;
                case Ending.Cancelled:
                    await Assert.ThrowsAnyAsync<OperationCanceledException>(() => request);
                    break;
                case Ending.RoundFailed:
                    await Assert.ThrowsAsync<InvalidOperationException>(() => request);
                    break;
            }
            // The pages came back with the release, before the caller saw the request complete.
            Assert.Equal(pool.MaxPages, pool.FreePageCount);
        }
        finally
        {
            foreach (Tensor t in w.Values) t.Dispose();
        }
    }

    /// <summary>A tokenizer with no stop ids, so a request runs to its budget unless it is cancelled or fails.</summary>
    private sealed class NoStopTokenizer : ILlmTokenizer
    {
        public int[] Encode(string text, bool addSpecial) => throw new NotSupportedException();
        public int[] EncodeOrdinary(string text) => throw new NotSupportedException();
        public string Decode(IReadOnlyList<int> ids) => string.Join(",", ids);
        public int? SpecialId(string token) => null;
        public int? BosId => null;
        public int? EosId => null;
        public IReadOnlyList<int> StopIds => [];
        public string? BosToken => null;
        public string? EosToken => null;
    }

    /// <summary>Forwards to a real model, but every sequence state it creates signals when its release begins and holds the release until <paramref name="release"/> is set.</summary>
    private sealed class HeldReleaseModel(IGenerationModel inner, ManualResetEventSlim releasing, ManualResetEventSlim release) : IGenerationModel
    {
        public GenerationModelInfo Info => inner.Info;
        public GenerationCapabilities Capabilities => inner.Capabilities;
        public IBackend OutputBackend => inner.OutputBackend;
        public ISequenceState CreateSequenceState(SequenceStateOptions options) => new HeldState(inner.CreateSequenceState(options), releasing, release);
        public Tensor Prefill(in PrefillChunk chunk, ISequenceState state) => inner.Prefill(chunk, Inner(state));
        public Tensor Prefill(in PrefillChunk chunk, ISequenceState state, CancellationToken cancel) => inner.Prefill(chunk, Inner(state), cancel);
        public Tensor DecodeBatch(ReadOnlySpan<int> tokenIds, ISequenceState[] states) => inner.DecodeBatch(tokenIds, [.. states.Select(Inner)]);
        public Tensor ProjectLogits(Tensor hidden, int rows) => inner.ProjectLogits(hidden, rows);
        public IEnumerable<Tensor> EnumerateWeights(bool includeRedundantSplits) => inner.EnumerateWeights(includeRedundantSplits);
        public long EstimateSequenceBytes(int contextTokens) => inner.EstimateSequenceBytes(contextTokens);
        public CapacitySnapshot Capacity() => inner.Capacity();
        public void Dispose() => inner.Dispose();

        private static ISequenceState Inner(ISequenceState state) => ((HeldState)state).Inner;
    }

    /// <summary>A sequence state whose release waits for the test.</summary>
    private sealed class HeldState(ISequenceState inner, ManualResetEventSlim releasing, ManualResetEventSlim release) : ISequenceState
    {
        public ISequenceState Inner => inner;
        public int Length => inner.Length;
        public int Capacity => inner.Capacity;
        public int MaxRollback => inner.MaxRollback;
        public void Truncate(int newLength) => inner.Truncate(newLength);
        public void Reset() => inner.Reset();

        public void Dispose()
        {
            releasing.Set();
            release.Wait();
            inner.Dispose();
        }
    }
}
