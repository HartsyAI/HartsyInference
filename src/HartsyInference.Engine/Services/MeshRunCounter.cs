namespace HartsyInference.Engine.Services;

/// <summary>Counts the runs on each pipeline that has no run gate of its own, so a free waits only for the runs on the pipelines it frees. A free holds new runs on those pipelines off until it is done; runs on any other pipeline carry on.</summary>
internal sealed class MeshRunCounter
{
    private readonly object _lock = new();
    private readonly Dictionary<string, Slot> _slots = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>The key a pipeline's runs are counted under: its kind and the key it is cached under.</summary>
    public static string KeyOf(string kind, string cacheKey) => $"{kind}:{cacheKey}";

    /// <summary>The keys that have a run in progress or a free in progress.</summary>
    public IReadOnlyList<string> ActiveKeys()
    {
        lock (_lock)
            return [.. _slots.Keys];
    }

    /// <summary>Runs <paramref name="run"/> counted under <paramref name="key"/>; a null key (nothing resident to protect) runs uncounted. While a free of the key waits or runs, the run waits without holding a thread, and gives up when <paramref name="cancel"/> is cancelled.</summary>
    public async Task<T> RunAsync<T>(string? key, Func<Task<T>> run, CancellationToken cancel = default)
    {
        if (key is null)
            return await run().ConfigureAwait(false);
        while (true)
        {
            Task reopened;
            lock (_lock)
            {
                Slot slot = SlotOf(key);
                if (!slot.Freeing)
                {
                    slot.Runs++;
                    break;
                }
                reopened = slot.Reopened.Task;
            }
            await reopened.WaitAsync(cancel).ConfigureAwait(false);
        }
        try
        {
            return await run().ConfigureAwait(false);
        }
        finally
        {
            lock (_lock)
            {
                Slot slot = _slots[key];
                if (--slot.Runs == 0)
                {
                    slot.Drained?.TrySetResult();
                    if (!slot.Freeing)
                        _slots.Remove(key);
                }
            }
        }
    }

    /// <summary>Runs <paramref name="free"/> once no run is in progress under any of <paramref name="keys"/>, holding new runs under them off until it finishes. The caller's thread waits: the lifecycle calls it serves are synchronous.</summary>
    public void WhenIdle(IEnumerable<string> keys, Action free)
    {
        List<(string Key, Slot Slot, TaskCompletionSource Drained, TaskCompletionSource Reopened)> held = [];
        try
        {
            // One order for every free, so two frees that share keys cannot each hold what the other waits for.
            foreach (string key in keys.Distinct(StringComparer.OrdinalIgnoreCase).Order(StringComparer.OrdinalIgnoreCase))
            {
                TaskCompletionSource drained = new(TaskCreationOptions.RunContinuationsAsynchronously);
                TaskCompletionSource reopened = new(TaskCreationOptions.RunContinuationsAsynchronously);
                while (true)
                {
                    Task? other = null;
                    lock (_lock)
                    {
                        Slot slot = SlotOf(key);
                        if (slot.Freeing)
                            other = slot.Reopened.Task;
                        else
                        {
                            slot.Freeing = true;
                            slot.Reopened = reopened;
                            slot.Drained = drained;
                            if (slot.Runs == 0)
                                drained.TrySetResult();
                            held.Add((key, slot, drained, reopened));
                            break;
                        }
                    }
                    other.Wait();
                }
            }
            foreach ((string _, Slot _, TaskCompletionSource drained, TaskCompletionSource _) in held)
                drained.Task.Wait();
            free();
        }
        finally
        {
            lock (_lock)
            {
                foreach ((string key, Slot slot, TaskCompletionSource _, TaskCompletionSource _) in held)
                {
                    slot.Freeing = false;
                    slot.Drained = null;
                    if (slot.Runs == 0 && _slots.TryGetValue(key, out Slot? current) && ReferenceEquals(current, slot))
                        _slots.Remove(key);
                }
            }
            foreach ((string _, Slot _, TaskCompletionSource _, TaskCompletionSource reopened) in held)
                reopened.TrySetResult();
        }
    }

    private Slot SlotOf(string key)
    {
        if (!_slots.TryGetValue(key, out Slot? slot))
            _slots[key] = slot = new Slot();
        return slot;
    }

    private sealed class Slot
    {
        public int Runs;
        public bool Freeing;
        public TaskCompletionSource Reopened = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource? Drained;
    }
}
