using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Requests;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>A MOSS-Image conditioning's reference images (the Ostris edit layout): their embedded tokens, and either the
/// prepass RoPE tables plus one <see cref="ReferenceKvCache"/> per skipped-layer set (<see cref="ReferenceAttentionMode.CachedKv"/>)
/// or the segment layout that gives them timestep-zero modulation inside the live sequence
/// (<see cref="ReferenceAttentionMode.Joint"/>). Built by <see cref="MossImageTransformer.PrepareConditioning"/> and
/// owned by its <see cref="MossImageConditioning"/>; host tables are pinned as resident weights so a captured step
/// graph reads them in place.</summary>
public sealed class MossImageReferenceState : IDisposable
{
    private readonly IBackend _backend;
    private readonly Dictionary<long, ReferenceKvCache> _caches = [];
    private int _disposed;

    internal MossImageReferenceState(IBackend backend, ReferenceAttentionMode mode, IReadOnlyList<(int Height, int Width)> grids,
        Tensor embedded, Tensor? cos, Tensor? sin, TokenSegmentLayout? layout)
    {
        _backend = backend;
        Mode = mode;
        Grids = grids;
        Embedded = embedded;
        Cos = cos;
        Sin = sin;
        Layout = layout;
        backend.PreloadWeights(Pinned());
        layout?.Pin(backend);
    }

    /// <summary>How the references are attended to.</summary>
    public ReferenceAttentionMode Mode { get; }

    /// <summary>Each reference's packed token grid, in picture order.</summary>
    public IReadOnlyList<(int Height, int Width)> Grids { get; }

    /// <summary>Reference tokens through <c>video_embedder</c>, <c>[1, S_ref, hidden]</c> F32.</summary>
    public Tensor Embedded { get; }

    /// <summary>Reference tokens across every image.</summary>
    public int TokenCount => (int)Embedded.Shape[1];

    /// <summary>Prepass RoPE cosines <c>[1, S_ref, headDim]</c>; cached mode only.</summary>
    public Tensor? Cos { get; }

    /// <summary>Prepass RoPE sines <c>[1, S_ref, headDim]</c>; cached mode only.</summary>
    public Tensor? Sin { get; }

    /// <summary><c>[image + text | references]</c> as modulation groups 0 and 1; joint mode only.</summary>
    public TokenSegmentLayout? Layout { get; }

    /// <summary>The cache built for skipped-layer set <paramref name="skipKey"/>, if any.</summary>
    public bool TryGetCache(long skipKey, out ReferenceKvCache? cache)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return _caches.TryGetValue(skipKey, out cache);
    }

    /// <summary>Stores <paramref name="cache"/> for <paramref name="skipKey"/>; this state disposes it.</summary>
    public void AddCache(long skipKey, ReferenceKvCache cache)
    {
        ArgumentNullException.ThrowIfNull(cache);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (Mode != ReferenceAttentionMode.CachedKv)
            throw new InvalidOperationException("Only cached-mode references hold key/value caches.");
        if (!_caches.TryAdd(skipKey, cache))
            throw new InvalidOperationException("A reference cache for this skipped-layer set already exists.");
    }

    private IEnumerable<Tensor> Pinned()
    {
        yield return Embedded;
        if (Cos is not null) yield return Cos;
        if (Sin is not null) yield return Sin;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (ReferenceKvCache cache in _caches.Values) cache.Dispose();
        _caches.Clear();
        _backend.FreeWeights(Pinned());
        Embedded.Dispose();
        Cos?.Dispose();
        Sin?.Dispose();
        Layout?.Dispose();
    }
}
