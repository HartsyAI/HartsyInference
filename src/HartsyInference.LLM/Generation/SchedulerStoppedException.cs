namespace HartsyInference.LLM.Generation;

/// <summary>A request the <see cref="DynamicBatchScheduler"/> could not finish because the scheduler was stopped (its model was unloaded or reloaded) while the request was queued or running. Retryable: the caller may resubmit once the model is loaded again.</summary>
/// <remarks>Derives from <see cref="ObjectDisposedException"/>, so handlers that already catch the disposal exception still see it.</remarks>
public sealed class SchedulerStoppedException()
    : ObjectDisposedException(nameof(DynamicBatchScheduler), "The batch scheduler stopped before this request finished (its model was unloaded or reloaded). Retry the request.");
