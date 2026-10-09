namespace HartsyInference.LLM.Generation;

/// <summary>A request the <see cref="DynamicBatchScheduler"/> refused because its waiting queue is full. Transient: the caller may retry after a short wait, and the API answers it with 429 and a Retry-After header.</summary>
public sealed class SchedulerQueueFullException(int capacity)
    : Exception($"The batch scheduler's queue is full ({capacity} requests waiting). Retry later.");
