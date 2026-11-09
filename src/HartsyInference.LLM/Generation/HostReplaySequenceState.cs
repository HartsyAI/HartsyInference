using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Generation;

/// <summary>A host reference sequence whose keys live in the model's own caches. It records the committed token ids, so
/// <see cref="Truncate"/> resets the caches and replays the kept prefix.</summary>
/// <remarks>The replay follows how the history was built: the first append runs as one prefill chunk and every later token runs
/// on its own. Those two modes are not arithmetically equivalent, so a rollback into the first chunk replays the kept part of it as
/// one chunk and the rest token by token. Not thread-safe: run one sequence at a time.</remarks>
/// <summary>Runs tokens at the end of a model's caches and writes their hidden rows.</summary>
public delegate void HostReplayForward(ReadOnlySpan<int> tokenIds, Span<float> hidden);

public sealed class HostReplaySequenceState : ISequenceState
{
    private readonly HostReplayForward _forward;
    private readonly Action _reset;
    private readonly Action? _dispose;
    private readonly List<int> _tokens = [];
    private readonly float[] _lastHidden;
    private readonly int _dim;
    private int _chunkLength;
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
        if (_tokens.Count == 0) _chunkLength = ids.Length;
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
        hidden.Slice((ids.Length - 1) * _dim, _dim).CopyTo(_lastHidden);
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
        int prefill = Math.Min(newLength, chunk);
        Append(kept.AsSpan(0, prefill), new float[prefill * _dim]);
        for (int i = prefill; i < newLength; i++) Append(kept.AsSpan(i, 1), new float[_dim]);
    }

    /// <inheritdoc />
    public void Reset()
    {
        _reset();
        _tokens.Clear();
        _chunkLength = 0;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _dispose?.Invoke();
    }
}
