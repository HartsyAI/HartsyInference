using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>The engine's view of one V4.1 sequence: the host model's state plus the committed token ids, so the sequence can be rolled back.</summary>
/// <remarks>The compressor's partial group and the sliding-window ring cannot be truncated in place, so <see cref="Truncate"/> resets and replays the kept
/// prefix. That is correct but costs a prefill of that prefix; it is for prefix reuse and rare rollbacks, not a per-token operation.</remarks>
public sealed class DeepSeekV41GenerationState : ISequenceState
{
    private readonly DeepSeekV41HostModel _model;
    private readonly DeepSeekV41SequenceState _state;
    private readonly List<int> _tokens = [];

    internal DeepSeekV41GenerationState(DeepSeekV41HostModel model, int capacity)
    {
        _model = model;
        _state = model.CreateState(capacity);
    }

    /// <inheritdoc />
    public int Length => _state.Length;

    /// <inheritdoc />
    public int Capacity => _state.Capacity;

    /// <inheritdoc />
    public int MaxRollback => Length;

    /// <summary>Runs <paramref name="ids"/> at the end of the sequence and writes every position's final hidden state to <paramref name="hidden"/>.</summary>
    internal void Append(ReadOnlySpan<int> ids, Span<float> hidden)
    {
        _model.Forward(ids, _state, hidden);
        _tokens.AddRange(ids.ToArray());
    }

    /// <inheritdoc />
    public void Truncate(int newLength)
    {
        if ((uint)newLength > (uint)Length) throw new ArgumentOutOfRangeException(nameof(newLength), newLength, $"The sequence holds {Length} tokens.");
        if (newLength == Length) return;
        int[] kept = _tokens.GetRange(0, newLength).ToArray();
        Reset();
        if (newLength == 0) return;
        try
        {
            Append(kept, new float[newLength * _model.Dim]);
        }
        catch
        {
            Reset();
            throw;
        }
    }

    /// <inheritdoc />
    public void Reset()
    {
        _state.Reset();
        _tokens.Clear();
    }

    /// <inheritdoc />
    public void Dispose()
    {
    }
}
