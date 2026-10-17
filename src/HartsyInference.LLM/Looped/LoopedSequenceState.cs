using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Looped;

/// <summary>A looped decoder's per-sequence state: the key/value cache (one slot per attention layer and loop, see <see cref="LoopedSchedule"/>)
/// plus the loop count the sequence was created for, which every forward on it must keep using because the recurrent slots hold
/// loop-specific keys.</summary>
public sealed class LoopedSequenceState : IKvCache
{
    private int _disposed;

    /// <summary>The cache holding every slot.</summary>
    public IKvCache Inner { get; }

    /// <summary>Passes of the recurrent block each forward on this sequence runs.</summary>
    public int Loops { get; }

    /// <summary>Wraps <paramref name="inner"/>, which this state then owns, for <paramref name="loops"/> passes.</summary>
    public LoopedSequenceState(IKvCache inner, int loops)
    {
        ArgumentNullException.ThrowIfNull(inner);
        if (loops < 1) throw new ArgumentOutOfRangeException(nameof(loops), loops, "Loops must be at least 1.");
        Inner = inner;
        Loops = loops;
    }

    /// <inheritdoc/>
    public int NumLayers => Inner.NumLayers;

    /// <inheritdoc/>
    public int CurrentLength => Inner.CurrentLength;

    /// <inheritdoc/>
    public void AppendStep(IBackend backend, int layer, Tensor newK, Tensor newV) => Inner.AppendStep(backend, layer, newK, newV);

    /// <inheritdoc/>
    public Tensor KeyPrefix(int layer) => Inner.KeyPrefix(layer);

    /// <inheritdoc/>
    public Tensor ValuePrefix(int layer) => Inner.ValuePrefix(layer);

    /// <inheritdoc/>
    public void AdvanceLength(int by) => Inner.AdvanceLength(by);

    /// <inheritdoc/>
    public int Length => Inner.Length;

    /// <inheritdoc/>
    public int Capacity => Inner.Capacity;

    /// <inheritdoc/>
    public int MaxRollback => Inner.MaxRollback;

    /// <inheritdoc/>
    public void Truncate(int newLength) => Inner.Truncate(newLength);

    /// <inheritdoc/>
    public void Reset() => Inner.Reset();

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        Inner.Dispose();
    }
}
