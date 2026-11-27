using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>
/// The child logits each upsample level of a predicting decoder chose, in level order (upstream's <c>subs</c> of
/// <c>decode_shape_slat(..., return_subs=True)</c>). A guided decoder takes them as <c>guide_subs</c>: it keeps the children whose
/// logit is positive, so its voxels land exactly where the shape decoder's did. Each level also keeps a fingerprint of the voxel
/// coordinates entering it, so a guide recorded over other voxels is refused even when the counts agree. Owns its tensors.
/// </summary>
public sealed class ShapeVaeSubdivisions : IDisposable
{
    private readonly List<Tensor> _levels = [];
    private readonly List<(int Count, ulong Hash)> _prints = [];
    private bool _disposed;

    /// <summary>Levels recorded so far.</summary>
    public int Count => _levels.Count;

    /// <summary>The F32 logits of level <paramref name="level"/>, one row of <see cref="Children"/> per voxel entering that level.</summary>
    public Tensor Level(int level)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)level >= (uint)_levels.Count)
            throw new ArgumentOutOfRangeException(nameof(level), level, $"Only {_levels.Count} subdivision levels were recorded.");
        return _levels[level];
    }

    /// <summary>Children of one voxel per level (2³).</summary>
    public static int Children => ShapeVaeConfigRules.ChildGroup;

    /// <summary>Takes ownership of <paramref name="logits"/>, recorded for the voxels <paramref name="coords"/> entering the level.</summary>
    internal void Add(Tensor logits, int[] coords)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _levels.Add(logits);
        _prints.Add(Fingerprint(coords));
    }

    /// <summary>Whether level <paramref name="level"/> was recorded over exactly these voxels, in this order.</summary>
    internal bool Matches(int level, int[] coords)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return (uint)level < (uint)_prints.Count && _prints[level] == Fingerprint(coords);
    }

    private static (int Count, ulong Hash) Fingerprint(int[] coords)
    {
        ulong hash = 14695981039346656037UL;
        foreach (int value in coords)
            hash = (hash ^ (uint)value) * 1099511628211UL;
        return (coords.Length / 4, hash);
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        foreach (Tensor level in _levels)
            level.Dispose();
        _levels.Clear();
        _prints.Clear();
    }
}
