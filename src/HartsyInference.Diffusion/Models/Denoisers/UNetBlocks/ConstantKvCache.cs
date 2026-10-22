using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.UNetBlocks;

/// <summary>Cross-attention K/V cache: the projections of the text conditioning are step-invariant (the
/// context tensor is fixed across the denoise loop), so the k/v Linears + head-split permutes run
/// once per generation instead of once per step (~140 GEMMs + 140 permutes/step on SDXL). Keyed on
/// (backend, context) reference pairs for at most TWO backends — CFG-branch parallelism (SDXL CFG-parallel)
/// calls the sub-block concurrently from two threads against two different backends and two different
/// (fixed, step-invariant) context tensors; a single unkeyed slot would race (torn metadata, or one
/// thread's eviction Dispose freeing a tensor the other thread is mid-SDPA against — the sub-block is
/// SHARED across both threads, only the backend argument differs per call). The lock guards only
/// the lookup/install, never the GPU compute (Linear/Permute0213), so the two branches' actual work
/// still overlaps. The fused single-GPU loop only ever touches one backend (cache hits every
/// step after the first). Per-step slicing callers (legacy CFG loop) build a fresh context tensor every
/// call, so they never hit and behave exactly as before.
/// <para>Each backend keeps up to <c>contextsPerBackend</c> contexts and replaces its oldest when full; a backend
/// only ever evicts its own entries. Pinned entries survive the per-step activation sweep, so they live until
/// <see cref="Clear"/>: the owner clears at the end of the span over which the contexts are not rewritten.</para></summary>
internal sealed class ConstantKvCache : ICachedKv
{
    private readonly object _lock = new object();
    private readonly List<Entry> _entries = [];
    private readonly int _contextsPerBackend;
    private readonly bool _pinned;
    private long _stores;

    private sealed record Entry(IBackend Backend, Tensor Context, Tensor KeyMh, Tensor ValueMh, long Stamp);

    /// <summary>One context per backend, unpinned: the UNet's slot per backend.</summary>
    public ConstantKvCache()
        : this(contextsPerBackend: 1, pinned: false)
    {
    }

    public ConstantKvCache(int contextsPerBackend, bool pinned)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(contextsPerBackend, 1);
        _contextsPerBackend = contextsPerBackend;
        _pinned = pinned;
    }

    public bool TryGet(IBackend backend, Tensor kvInput, out Tensor keyMh, out Tensor valueMh)
    {
        lock (_lock)
        {
            foreach (Entry e in _entries)
            {
                if (ReferenceEquals(e.Backend, backend) && ReferenceEquals(e.Context, kvInput))
                {
                    keyMh = e.KeyMh;
                    valueMh = e.ValueMh;
                    return true;
                }
            }
        }
        keyMh = null!;
        valueMh = null!;
        return false;
    }

    public bool TryStore(IBackend backend, Tensor kvInput, Tensor keyMh, Tensor valueMh)
    {
        Entry? evicted = null;
        lock (_lock)
        {
            Entry? oldest = null;
            int own = 0;
            HashSet<IBackend> backends = new(ReferenceEqualityComparer.Instance);
            foreach (Entry e in _entries)
            {
                backends.Add(e.Backend);
                if (!ReferenceEquals(e.Backend, backend)) continue;
                own++;
                if (oldest is null || e.Stamp < oldest.Stamp) oldest = e;
            }
            // A third backend stores nothing: evicting another backend's entry could free K/V that backend's thread is attending over.
            if (own == 0 && backends.Count >= 2)
                return false;
            if (own >= _contextsPerBackend)
            {
                _entries.Remove(oldest!);
                evicted = oldest;
            }
            if (_pinned)
            {
                backend.PinActivation(keyMh);
                backend.PinActivation(valueMh);
            }
            _entries.Add(new Entry(backend, kvInput, keyMh, valueMh, ++_stores));
        }
        if (evicted is not null) Release(evicted);
        return true;
    }

    /// <summary>Disposes every entry; the next lookup misses.</summary>
    public void Clear()
    {
        Entry[] entries;
        lock (_lock)
        {
            entries = [.. _entries];
            _entries.Clear();
        }
        foreach (Entry e in entries) Release(e);
    }

    private void Release(Entry e)
    {
        if (_pinned)
        {
            e.Backend.UnpinActivation(e.KeyMh);
            e.Backend.UnpinActivation(e.ValueMh);
        }
        e.KeyMh.Dispose();
        e.ValueMh.Dispose();
    }
}
