namespace HartsyInference.Diffusion.Streaming;

/// <summary>Cuts a causal frame sequence into chunks for chunk-by-chunk generation. A bounded plan knows its total and
/// every chunk up front; an unbounded plan (a live stream) yields chunks for as long as it is asked. The bounded rules follow
/// the chunk-causal samplers they come from: a remainder shorter than the base size is never a chunk of its own.</summary>
public sealed class ChunkPlanner
{
    private readonly int[]? _boundaries;

    /// <summary>Plans chunks of <paramref name="chunkFrames"/> frames.</summary>
    /// <param name="totalFrames">Frames in the whole sequence, or null for an unbounded stream.</param>
    /// <exception cref="ArgumentException">The strategy needs a total and none was given, or the total is too short for it.</exception>
    public ChunkPlanner(int chunkFrames, ChunkSplitStrategy strategy, int? totalFrames = null)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(chunkFrames, 1);
        if (totalFrames is int total)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(total, 1, nameof(totalFrames));
            _boundaries = Boundaries(total, chunkFrames, strategy);
        }
        else if (strategy == ChunkSplitStrategy.FirstAbsorbsRemainder)
        {
            throw new ArgumentException("A first chunk that absorbs the remainder needs the total frame count.", nameof(totalFrames));
        }

        ChunkFrames = chunkFrames;
        Strategy = strategy;
        TotalFrames = totalFrames;
    }

    /// <summary>The base chunk size, in frames.</summary>
    public int ChunkFrames { get; }

    /// <summary>The split rule.</summary>
    public ChunkSplitStrategy Strategy { get; }

    /// <summary>Frames in the whole sequence, or null for an unbounded stream.</summary>
    public int? TotalFrames { get; }

    /// <summary>Chunks in a bounded plan, or null for an unbounded stream.</summary>
    public int? ChunkCount => _boundaries is null ? null : _boundaries.Length - 1;

    /// <summary>Frames in chunk 0.</summary>
    public int FirstChunkFrames => Span(0).Length;

    /// <summary>Chunk <paramref name="index"/>.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The index is negative or past the last chunk of a bounded plan.</exception>
    public ChunkSpan Span(int index)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        if (_boundaries is not null)
        {
            if (index >= _boundaries.Length - 1)
                throw new ArgumentOutOfRangeException(nameof(index), index, $"The plan has {_boundaries.Length - 1} chunks.");
            return new ChunkSpan(index, _boundaries[index], _boundaries[index + 1]);
        }

        int first = Strategy switch
        {
            ChunkSplitStrategy.FirstFrameAlone => 1,
            ChunkSplitStrategy.FirstChunkPlusOne => ChunkFrames + 1,
            _ => ChunkFrames,
        };
        if (index == 0)
            return new ChunkSpan(0, 0, first);
        int start = checked(first + (index - 1) * ChunkFrames);
        return new ChunkSpan(index, start, checked(start + ChunkFrames));
    }

    /// <summary>Whether <paramref name="index"/> is the last chunk of a bounded plan; always false for an unbounded stream.</summary>
    public bool IsLast(int index) => _boundaries is not null && index == _boundaries.Length - 2;

    /// <summary>Chunk boundaries <c>[0, c1, …, total]</c> of a bounded sequence.</summary>
    /// <exception cref="ArgumentException">The total is shorter than one base chunk under
    /// <see cref="ChunkSplitStrategy.FirstAbsorbsRemainder"/>.</exception>
    public static int[] Boundaries(int totalFrames, int chunkFrames, ChunkSplitStrategy strategy)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(totalFrames, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(chunkFrames, 1);
        List<int> starts = [0];
        switch (strategy)
        {
            case ChunkSplitStrategy.FirstAbsorbsRemainder:
                if (totalFrames < chunkFrames)
                    throw new ArgumentException($"{totalFrames} frames are fewer than one chunk of {chunkFrames}.", nameof(totalFrames));
                for (int start = chunkFrames + totalFrames % chunkFrames; start < totalFrames; start += chunkFrames)
                    starts.Add(start);
                break;
            case ChunkSplitStrategy.Uniform:
                AddUniform(starts, chunkFrames, totalFrames, chunkFrames, minimumStarts: 1);
                break;
            case ChunkSplitStrategy.FirstFrameAlone:
                AddUniform(starts, 1, totalFrames, chunkFrames, minimumStarts: 2);
                break;
            case ChunkSplitStrategy.FirstChunkPlusOne:
                AddUniform(starts, chunkFrames + 1, totalFrames, chunkFrames, minimumStarts: 1);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(strategy), strategy, "Unknown chunk split strategy.");
        }

        int[] boundaries = new int[starts.Count + 1];
        starts.CopyTo(boundaries);
        boundaries[^1] = totalFrames;
        return boundaries;
    }

    // Starts every chunkFrames from `first`; a last chunk shorter than the base merges into the one before it, unless that
    // would leave fewer than minimumStarts chunk starts.
    private static void AddUniform(List<int> starts, int first, int totalFrames, int chunkFrames, int minimumStarts)
    {
        for (int start = first; start < totalFrames; start += chunkFrames)
            starts.Add(start);
        if (starts.Count > minimumStarts && totalFrames - starts[^1] < chunkFrames)
            starts.RemoveAt(starts.Count - 1);
    }
}
