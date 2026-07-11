namespace HartsyInference.LLM.Generation;

/// <summary>A generation scheduler that admits requests dynamically (at any time, not just up front) and
/// batches active sequences' decode steps together for throughput, evicting each sequence as soon as it
/// finishes/stops/is cancelled rather than waiting for the whole cohort to complete. See
/// <see cref="DynamicBatchScheduler"/> for the implementation.</summary>
public interface IBatchScheduler
{
    /// <summary>Submits one request for generation. Returns as soon as the request is queued; the returned
    /// task completes when generation finishes (result), the request is cancelled via
    /// <paramref name="ct"/> (<see cref="OperationCanceledException"/>), or admission/generation fails
    /// (faulted — e.g. <see cref="Transformer.KvPoolExhaustedException"/> when the scheduler is out of KV
    /// capacity). <paramref name="onToken"/>, if supplied, is invoked once per generated token as it is
    /// produced, from the scheduler's internal loop (not the calling thread) — keep it fast and
    /// non-blocking.</summary>
    Task<GenerationResult> SubmitAsync(GenerationRequest request, Action<int>? onToken, CancellationToken ct);
}
