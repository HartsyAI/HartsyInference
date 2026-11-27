using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Generation;

/// <summary>Runs tokens at the end of a model's caches and writes their hidden rows.</summary>
public delegate void HostReplayForward(ReadOnlySpan<int> tokenIds, Span<float> hidden);

/// <summary>A host reference sequence whose keys live in the model's own caches. It records the committed token ids and the size of each
/// append, so <see cref="Truncate"/> resets the caches and replays the kept prefix with the same chunking the history was built with.</summary>
/// <remarks>Replay costs one forward pass per recorded append, not one per token, so a rollback does not re-read the weights once per token.
/// A rollback into the middle of an append replays the kept part of it as one chunk. Not thread-safe: run one sequence at a time.</remarks>
public sealed class HostReplaySequenceState : ISequenceState
{
    private readonly HostReplayForward _forward;
    private readonly Action _reset;
    private readonly Action? _dispose;
    private readonly List<int> _tokens = [];
    private readonly List<int> _appendLengths = [];
    private readonly float[] _lastHidden;
    private readonly int _dim;
    private int _disposed;

    /// <param name="capacity">Most tokens the sequence may hold.</param>
    /// <param name="dim">Width of one hidden row.</param>
    /// <param name="forward">Runs the tokens at the end of the model's caches and writes every position's hidden row, <c>[tokens, dim]</c>. Throws
    /// on failure, after which the caches are reset.</param>
    /// <param name="reset">Empties the model's caches.</param>
    /// <param name="dispose">Releases the model's caches, or null when the owner releases them.</param>
    public HostReplaySequenceState(int capacity, int dim, HostReplayForward forward, Action reset, Action? dispose = null)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(dim, 1);
        ArgumentNullException.ThrowIfNull(forward);
        ArgumentNullException.ThrowIfNull(reset);
        Capacity = capacity;
        _dim = dim;
        _forward = forward;
        _reset = reset;
        _dispose = dispose;
        _lastHidden = new float[dim];
    }

    /// <inheritdoc />
    public int Length => _tokens.Count;

    /// <inheritdoc />
    public int Capacity { get; }

    /// <inheritdoc />
    public int MaxRollback => Length;

    /// <summary>The committed token ids, oldest first.</summary>
    public IReadOnlyList<int> Tokens => _tokens;

    /// <summary>Final normed hidden row of the last committed token, <c>[dim]</c>. Valid after an append.</summary>
    public ReadOnlySpan<float> LastHidden => _lastHidden;

    /// <summary>Runs <paramref name="ids"/> at the end of the sequence and writes every position's hidden row to <paramref name="hidden"/>.</summary>
    public void Append(ReadOnlySpan<int> ids, Span<float> hidden)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        if (ids.IsEmpty) throw new ArgumentException("Append needs at least one token.", nameof(ids));
        if (Length + ids.Length > Capacity)
            throw new InvalidOperationException($"The sequence holds {Capacity} tokens; {Length + ids.Length} were asked for.");
        if (hidden.Length != ids.Length * _dim) throw new ArgumentException($"hidden must hold {ids.Length} rows of {_dim} values.", nameof(hidden));
        try
        {
            _forward(ids, hidden);
        }
        catch
        {
            Reset();
            throw;
        }
        _tokens.AddRange(ids);
        _appendLengths.Add(ids.Length);
        hidden.Slice((ids.Length - 1) * _dim, _dim).CopyTo(_lastHidden);
    }

    /// <inheritdoc />
    public void Truncate(int newLength)
    {
        if ((uint)newLength > (uint)Length) throw new ArgumentOutOfRangeException(nameof(newLength), newLength, $"The sequence holds {Length} tokens.");
        if (newLength == Length) return;
        int[] kept = _tokens.GetRange(0, newLength).ToArray();
        List<int> chunks = [];
        int remaining = newLength;
        foreach (int length in _appendLengths)
        {
            if (remaining == 0) break;
            int take = Math.Min(length, remaining);
            chunks.Add(take);
            remaining -= take;
        }
        Reset();
        int start = 0;
        foreach (int chunk in chunks)
        {
            Append(kept.AsSpan(start, chunk), new float[chunk * _dim]);
            start += chunk;
        }
    }

    /// <inheritdoc />
    public void Reset()
    {
        _reset();
        _tokens.Clear();
        _appendLengths.Clear();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _dispose?.Invoke();
    }
}
