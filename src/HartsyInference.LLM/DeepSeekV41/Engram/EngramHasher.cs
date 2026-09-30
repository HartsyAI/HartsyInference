namespace HartsyInference.LLM.DeepSeekV41.Engram;

/// <summary>
/// Maps each position of one sequence to its Engram table rows: for every Engram layer, the hash ids of the 2-, 3- and 4-grams ending there
/// (8 heads each), shaped <c>[positions, layers, 24]</c>. A port of upstream <c>NgramHashState.forward</c>.
/// </summary>
/// <remarks>
/// Token ids go through the compressed token map, then a position is hashed with the tokens before it: the XOR of <c>compressed_id * multiplier</c>
/// over the lookbacks so far, modulo the prime of each (order, head), plus that column's cumulative offset. A lookback that runs past the start of
/// the sequence, or reaches (or crosses) a dead token (an image span), hashes as the pad id, and once a lookback is blocked so are all longer ones.
/// The compressed history is kept between calls, so a decode step sees the prefill's tokens; calling again with an earlier <c>startPos</c>
/// rewinds and overwrites, as speculative decoding needs. Not thread-safe.
/// </remarks>
public sealed class EngramHasher
{
    private readonly EngramConstants _constants;
    private readonly int _layers;
    private int[] _history = new int[256];
    private int _filled;

    /// <summary>Creates a hasher over <paramref name="constants"/> (the pinned ones unless a test supplies its own).</summary>
    public EngramHasher(EngramConstants? constants = null)
    {
        _constants = constants ?? EngramConstants.Default;
        _layers = _constants.LayerIds.Count;
    }

    /// <summary>Hash ids per position: layers times <see cref="EngramConstants.ColumnsPerLayer"/>.</summary>
    public int ValuesPerPosition => _layers * EngramConstants.ColumnsPerLayer;

    /// <summary>Positions of history recorded so far.</summary>
    public int Length => _filled;

    /// <summary>Forgets the sequence.</summary>
    public void Reset() => _filled = 0;

    /// <summary>Hashes <paramref name="inputIds"/>, which occupy sequence positions <paramref name="startPos"/> onward.</summary>
    /// <param name="inputIds">Tokenizer ids of the new tokens.</param>
    /// <param name="tokenMask">False for tokens that take no part in an n-gram (image spans); empty means all take part.</param>
    /// <param name="startPos">Sequence position of the first new token: 0 for a prefill, the running length for a decode step.</param>
    /// <param name="hashIds">Receives <c>inputIds.Length * ValuesPerPosition</c> row ids, position-major then layer then column.</param>
    public void Hash(ReadOnlySpan<int> inputIds, ReadOnlySpan<bool> tokenMask, int startPos, Span<long> hashIds)
    {
        int count = inputIds.Length;
        if (!tokenMask.IsEmpty && tokenMask.Length != count)
            throw new ArgumentException($"The mask has {tokenMask.Length} entries for {count} tokens.", nameof(tokenMask));
        if (startPos < 0 || startPos > _filled)
            throw new ArgumentOutOfRangeException(nameof(startPos), $"Position {startPos} leaves a gap: only {_filled} positions are recorded.");
        if (hashIds.Length < (long)count * ValuesPerPosition)
            throw new ArgumentException($"The output holds {hashIds.Length} ids but {count} positions need {(long)count * ValuesPerPosition}.", nameof(hashIds));

        int end = checked(startPos + count);
        if (_history.Length < end)
            Array.Resize(ref _history, Math.Max(end, _history.Length * 2));
        ReadOnlySpan<int> tokenMap = _constants.TokenMap;
        for (int i = 0; i < count; i++)
        {
            int id = inputIds[i];
            if ((uint)id >= (uint)tokenMap.Length)
                throw new ArgumentOutOfRangeException(nameof(inputIds), $"Token id {id} at index {i} is outside the {tokenMap.Length}-token vocabulary.");
            _history[startPos + i] = tokenMask.IsEmpty || tokenMask[i] ? tokenMap[id] : EngramConstants.Dead;
        }
        _filled = end;

        int maxOrder = _constants.MaxNgramSize;
        int heads = _constants.HeadCount;
        int pad = _constants.PadCompressedId;
        ReadOnlySpan<long> multipliers = _constants.Multipliers;
        ReadOnlySpan<long> primes = _constants.Primes;
        ReadOnlySpan<long> offsets = _constants.Offsets;
        Span<int> lookback = stackalloc int[maxOrder];
        int columns = EngramConstants.ColumnsPerLayer;
        for (int i = 0; i < count; i++)
        {
            int position = startPos + i;
            bool blocked = false;
            for (int shift = 0; shift < maxOrder; shift++)
            {
                int source = _history[Math.Max(position - shift, 0)];
                blocked |= position < shift || source == EngramConstants.Dead;
                lookback[shift] = blocked ? pad : source;
            }
            for (int layer = 0; layer < _layers; layer++)
            {
                long rolling = lookback[0] * multipliers[layer * maxOrder];
                for (int order = 1; order < maxOrder; order++)
                {
                    rolling ^= lookback[order] * multipliers[layer * maxOrder + order];
                    for (int head = 0; head < heads; head++)
                    {
                        int column = (order - 1) * heads + head;
                        int at = layer * columns + column;
                        hashIds[(i * _layers + layer) * columns + column] = rolling % primes[at] + offsets[at];
                    }
                }
            }
        }
    }
}
