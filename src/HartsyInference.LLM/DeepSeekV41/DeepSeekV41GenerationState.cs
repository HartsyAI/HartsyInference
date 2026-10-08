using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>The engine's view of one V4.1 sequence: the host model's state plus the committed token ids, so the sequence can be rolled back.</summary>
/// <remarks>The compressor's partial group and the sliding-window ring cannot be truncated in place, so <see cref="Truncate"/> resets and replays the kept
/// prefix. The replay follows how the history was built: the first append runs as one prefill chunk and every later token runs one at a time. Those two
/// modes are not arithmetically equivalent (the sparse attention's selection can differ between them), so a single-chunk replay of a decoded tail would
/// not restore the state that decoding built. A rollback into the first chunk is the exception: it replays a shorter chunk, which is not the original chunk's
/// arithmetic. The speculative scorer never does that: a context that diverges inside the prompt is prefilled afresh instead.</remarks>
public sealed class DeepSeekV41GenerationState : ISequenceState
{
    private readonly DeepSeekV41HostModel _model;
    private readonly DeepSeekV41SequenceState _state;
    private readonly List<int> _tokens = [];

    // tokens in the first append, which the host ran as one prefill chunk; each later token ran on its own
    private int _chunkLength;

    internal DeepSeekV41GenerationState(DeepSeekV41HostModel model, int capacity)
    {
        _model = model;
        _state = model.CreateState(capacity);
    }

    /// <inheritdoc />
    public int Length => _state.Length;

    /// <summary>The committed token ids, oldest first; a speculative scorer compares them with its context to find how far back to roll.</summary>
    internal IReadOnlyList<int> Tokens => _tokens;

    /// <summary>Length of the first append, which the host ran as one prefill chunk; 0 for an empty sequence. Rolling back below it replays a shorter chunk.</summary>
    internal int PrefillLength => _chunkLength;

    /// <inheritdoc />
    public int Capacity => _state.Capacity;

    /// <inheritdoc />
    public int MaxRollback => Length;

    /// <summary>Runs <paramref name="ids"/> at the end of the sequence and writes every position's final hidden state to <paramref name="hidden"/>. A failure
    /// resets the sequence: the host can be left partly advanced, which the token list cannot describe.</summary>
    internal void Append(ReadOnlySpan<int> ids, Span<float> hidden)
    {
        if (_tokens.Count == 0) _chunkLength = ids.Length;
        try
        {
            _model.Forward(ids, _state, hidden);
        }
        catch
        {
            Reset();
            throw;
        }
        _tokens.AddRange(ids.ToArray());
    }

    /// <inheritdoc />
    public void Truncate(int newLength)
    {
        if ((uint)newLength > (uint)Length) throw new ArgumentOutOfRangeException(nameof(newLength), newLength, $"The sequence holds {Length} tokens.");
        if (newLength == Length) return;
        int[] kept = _tokens.GetRange(0, newLength).ToArray();
        int chunk = _chunkLength;
        Reset();
        if (newLength == 0) return;
        try
        {
            // the prefix the first chunk covered is replayed as one chunk, and the rest one token at a time, as the history was built
            int prefill = Math.Min(newLength, chunk);
            Append(kept.AsSpan(0, prefill), new float[prefill * _model.Dim]);
            for (int i = prefill; i < newLength; i++) Append(kept.AsSpan(i, 1), new float[_model.Dim]);
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
        _chunkLength = 0;
    }

    /// <inheritdoc />
    public void Dispose()
    {
    }
}
