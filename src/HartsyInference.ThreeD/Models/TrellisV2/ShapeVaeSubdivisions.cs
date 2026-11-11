using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>
/// The child logits each upsample level of a predicting decoder chose, in level order (upstream's <c>subs</c> of
/// <c>decode_shape_slat(..., return_subs=True)</c>). A guided decoder takes them as <c>guide_subs</c>: it keeps the children whose
/// logit is positive, so its voxels land exactly where the shape decoder's did. Owns its tensors.
/// </summary>
public sealed class ShapeVaeSubdivisions : IDisposable
{
    private readonly List<Tensor> _levels = [];
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

    /// <summary>Takes ownership of <paramref name="logits"/>.</summary>
    internal void Add(Tensor logits)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _levels.Add(logits);
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        foreach (Tensor level in _levels)
            level.Dispose();
        _levels.Clear();
    }
}
