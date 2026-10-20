using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>A DiT sequence cut into contiguous <see cref="TokenSegment"/>s, each naming the modulation group its tokens
/// take, with the I32 <c>[TotalLength]</c> row index the row-indexed backend ops read. <see cref="Pin"/> keeps the index
/// resident on a backend, which a captured step graph needs; disposing unpins and frees it.</summary>
public sealed class TokenSegmentLayout : IDisposable
{
    private readonly TokenSegment[] _segments;
    private IBackend? _pinnedOn;
    private int _disposed;

    /// <summary>Lays out segments of <paramref name="lengths"/> tokens in order, segment <c>i</c> taking group
    /// <paramref name="groups"/>[i]. Groups must cover <c>0..max</c> without gaps.</summary>
    public unsafe TokenSegmentLayout(ReadOnlySpan<int> lengths, ReadOnlySpan<int> groups)
    {
        if (lengths.Length == 0 || lengths.Length != groups.Length)
            throw new ArgumentException($"{lengths.Length} segment lengths for {groups.Length} groups.", nameof(groups));
        _segments = new TokenSegment[lengths.Length];
        int start = 0, maxGroup = -1;
        for (int i = 0; i < lengths.Length; i++)
        {
            if (lengths[i] < 1 || groups[i] < 0)
                throw new ArgumentOutOfRangeException(nameof(lengths), $"Segment {i} has length {lengths[i]} and group {groups[i]}.");
            _segments[i] = new TokenSegment(start, lengths[i], groups[i]);
            start += lengths[i];
            maxGroup = Math.Max(maxGroup, groups[i]);
        }
        GroupCount = maxGroup + 1;
        Span<bool> seen = stackalloc bool[GroupCount];
        foreach (int group in groups) seen[group] = true;
        if (seen.Contains(false))
            throw new ArgumentException($"Segment groups must cover 0..{maxGroup} without gaps.", nameof(groups));
        TotalLength = start;
        RowIndex = new Tensor(new TensorShape(TotalLength), DType.I32);
        int* p = (int*)RowIndex.DataPointer;
        foreach (TokenSegment segment in _segments)
        {
            for (int t = segment.Start; t < segment.Stop; t++) p[t] = segment.Group;
        }
    }

    /// <summary>The segments in sequence order.</summary>
    public IReadOnlyList<TokenSegment> Segments => _segments;

    /// <summary>Tokens across every segment.</summary>
    public int TotalLength { get; }

    /// <summary>Distinct modulation groups (rows of a matching table).</summary>
    public int GroupCount { get; }

    /// <summary>I32 <c>[TotalLength]</c>: each token's group.</summary>
    public Tensor RowIndex { get; }

    /// <summary>Uploads the row index to <paramref name="backend"/> and keeps it there until disposed.</summary>
    public void Pin(IBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_pinnedOn is not null)
        {
            if (!ReferenceEquals(_pinnedOn, backend))
                throw new InvalidOperationException("A segment layout is pinned on one backend at a time.");
            return;
        }
        backend.PreloadWeights([RowIndex]);
        _pinnedOn = backend;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _pinnedOn?.FreeWeights([RowIndex]);
        RowIndex.Dispose();
    }
}
