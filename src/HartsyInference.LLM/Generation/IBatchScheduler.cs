namespace HartsyInference.LLM.Generation;

/// <summary>Admits requests dynamically and batches active sequences' decode steps together for throughput, evicting each sequence as soon as it finishes/stops/is cancelled rather than waiting for the whole cohort; see <see cref="DynamicBatchScheduler"/> for the implementation.</summary>
public interface IBatchScheduler
{
    /// <summary>Queues <paramref name="request"/> and returns immediately; the task completes on generation finishing, cancellation via <paramref name="ct"/>, or admission/generation failure (e.g. <see cref="Transformer.KvPoolExhaustedException"/>). <paramref name="onToken"/>, if supplied, fires once per token from the scheduler's internal loop (not the calling thread) — keep it fast and non-blocking.</summary>
    Task<GenerationResult> SubmitAsync(GenerationRequest request, Action<int>? onToken, CancellationToken ct);
}
