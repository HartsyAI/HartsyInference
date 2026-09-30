namespace HartsyInference.LLM.Transformer;

/// <summary>Per-sequence decode state a generation model owns for one request: committed length, rollback and reset. <see cref="IKvCache"/> implements it for the dense transformer; other architectures bring their own state type.</summary>
public interface ISequenceState : IDisposable
{
    /// <summary>Tokens currently committed to this sequence.</summary>
    int Length { get; }

    /// <summary>Most tokens this sequence can hold.</summary>
    int Capacity { get; }

    /// <summary>Furthest back <see cref="Truncate"/> can reach from the current length; 0 when rollback is unsupported.</summary>
    int MaxRollback { get; }

    /// <summary>Rolls the committed length back to <paramref name="newLength"/> (at most <see cref="Length"/>), discarding everything beyond it.</summary>
    void Truncate(int newLength);

    /// <summary>Drops all state and returns to an empty sequence.</summary>
    void Reset();

    /// <summary>Records the current cursor so a later <see cref="Rollback"/> can return to it.</summary>
    SequenceCheckpoint Checkpoint() => new(Length);

    /// <summary>Rolls back to <paramref name="checkpoint"/>; equivalent to <c>Truncate(checkpoint.Length)</c>.</summary>
    void Rollback(in SequenceCheckpoint checkpoint) => Truncate(checkpoint.Length);
}
