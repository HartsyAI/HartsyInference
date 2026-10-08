using System.Runtime.InteropServices;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>The engine's view of one V4.1 sequence: the host model's state plus the committed token ids, so the sequence can be rolled back.</summary>
/// <remarks>The compressor's partial group and the sliding-window ring cannot be truncated in place, so <see cref="Truncate"/> resets and replays the kept
/// prefix. The replay follows how the history was built: the first append runs as one prefill chunk and every later token runs one at a time. Those two
/// modes are not arithmetically equivalent (the sparse attention's selection can differ between them), so a single-chunk replay of a decoded tail would
/// not restore the state that decoding built. A rollback into the first chunk is the exception: it replays a shorter chunk, which is not the original chunk's
/// arithmetic. <see cref="SyncTo"/> never does that: a context that diverges inside the prompt is prefilled afresh instead.
/// The final hidden row of the last committed token is kept, so reading its logits needs no replay. With <c>recordMainRows</c> the state also keeps each
/// committed position's DSpark target rows, and a replay recomputes them with the rest of the history.</remarks>
public sealed class DeepSeekV41GenerationState : ISequenceState
{
    private readonly DeepSeekV41HostModel _model;
    private readonly DeepSeekV41SequenceState _state;
    private readonly List<int> _tokens = [];
    private readonly float[] _lastHidden;
    private readonly int _mainWidth;
    private readonly List<float> _mainRows = [];

    // tokens in the first append, which the host ran as one prefill chunk; each later token ran on its own
    private int _chunkLength;

    internal DeepSeekV41GenerationState(DeepSeekV41HostModel model, int capacity, bool recordMainRows = false)
    {
        _model = model;
        _state = model.CreateState(capacity);
        _lastHidden = new float[model.Dim];
        if (recordMainRows)
        {
            if (model.MainHiddenWidth == 0) throw new InvalidOperationException("The model has no DSpark target layers to record.");
            _mainWidth = model.MainHiddenWidth;
        }
    }

    /// <inheritdoc />
    public int Length => _state.Length;

    /// <summary>The committed token ids, oldest first.</summary>
    internal IReadOnlyList<int> Tokens => _tokens;

    /// <summary>Final normed hidden row of the last committed token, <c>[Dim]</c>; its logits are the next-token distribution. Valid after <see cref="SyncTo"/> or an append.</summary>
    internal ReadOnlySpan<float> LastHidden => _lastHidden;

    /// <summary>The DSpark target rows of committed position <paramref name="position"/>, <c>[MainWidth]</c>. Only for a state that records them.</summary>
    internal ReadOnlySpan<float> MainRow(int position)
    {
        if (_mainWidth == 0) throw new InvalidOperationException("This sequence does not record DSpark target rows.");
        if ((uint)position >= (uint)Length) throw new ArgumentOutOfRangeException(nameof(position), position, $"The sequence holds {Length} tokens.");
        return CollectionsMarshal.AsSpan(_mainRows).Slice(position * _mainWidth, _mainWidth);
    }

    /// <inheritdoc />
    public int Capacity => _state.Capacity;

    /// <inheritdoc />
    public int MaxRollback => Length;

    /// <summary>Runs <paramref name="ids"/> at the end of the sequence and writes every position's final hidden state to <paramref name="hidden"/>. A failure
    /// resets the sequence: the host can be left partly advanced, which the token list cannot describe, so the next call prefills again.</summary>
    internal void Append(ReadOnlySpan<int> ids, Span<float> hidden)
    {
        if (_tokens.Count == 0) _chunkLength = ids.Length;
        float[]? main = _mainWidth == 0 ? null : new float[ids.Length * _mainWidth];
        try
        {
            if (main is null) _model.Forward(ids, _state, hidden);
            else _model.Forward(ids, _state, hidden, main);
        }
        catch
        {
            Reset();
            throw;
        }
        _tokens.AddRange(ids.ToArray());
        if (main is not null) _mainRows.AddRange(main);
        hidden.Slice((ids.Length - 1) * _model.Dim, _model.Dim).CopyTo(_lastHidden);
    }

    /// <summary>Makes the committed sequence exactly <paramref name="context"/>. It rolls back only past the first token where the two differ, so when the
    /// sequence already holds the whole context nothing is replayed and <see cref="LastHidden"/> is the context's last row.</summary>
    internal void SyncTo(ReadOnlySpan<int> context)
    {
        if (context.IsEmpty) throw new ArgumentException("The context must hold at least one token.", nameof(context));
        IReadOnlyList<int> held = _tokens;
        int limit = Math.Min(held.Count, context.Length), common = 0;
        while (common < limit && held[common] == context[common]) common++;
        // a divergence inside the prompt chunk is prefilled afresh, as plain decoding of that context would prefill it
        if (common < Math.Min(_chunkLength, held.Count))
        {
            Truncate(0);
            common = 0;
        }
        else if (common < Length)
        {
            Truncate(common);
        }
        if (common < context.Length) Append(context[common..], new float[(context.Length - common) * _model.Dim]);
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
        _mainRows.Clear();
        _chunkLength = 0;
    }

    /// <inheritdoc />
    public void Dispose()
    {
    }
}
