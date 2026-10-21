using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.UNetBlocks;

/// <summary>Cross-attention K/V cache: the projections of the text conditioning are step-invariant (the
/// context tensor is fixed across the denoise loop), so the k/v Linears + head-split permutes run
/// once per generation instead of once per step (~140 GEMMs + 140 permutes/step on SDXL). Keyed on
/// (backend, context) reference pairs across TWO slots — CFG-branch parallelism (SDXL CFG-parallel)
/// calls the sub-block concurrently from two threads against two different backends and two different
/// (fixed, step-invariant) context tensors; a single unkeyed slot would race (torn metadata, or one
/// thread's eviction Dispose freeing a tensor the other thread is mid-SDPA against — the sub-block is
/// SHARED across both threads, only the backend argument differs per call). The lock guards only
/// the slot lookup/install, never the GPU compute (Linear/Permute0213), so the two branches' actual work
/// still overlaps. The fused single-GPU loop only ever touches slot 0 (one backend, cache hits every
/// step after the first). Per-step slicing callers (legacy CFG loop) build a fresh context tensor every
/// call, so they never hit either slot and behave exactly as before.</summary>
internal sealed class ConstantKvCache : ICachedKv
{
    private readonly object _lock = new object();
    private Slot _slot0;
    private Slot _slot1;

    private struct Slot
    {
        public IBackend? Backend;
        public Tensor? Context;
        public Tensor? KeyMh;
        public Tensor? ValueMh;
    }

    public bool TryGet(IBackend backend, Tensor kvInput, out Tensor keyMh, out Tensor valueMh)
    {
        lock (_lock)
        {
            if (ReferenceEquals(_slot0.Backend, backend) && ReferenceEquals(_slot0.Context, kvInput) && _slot0.KeyMh is not null)
            {
                keyMh = _slot0.KeyMh; valueMh = _slot0.ValueMh!;
                return true;
            }
            if (ReferenceEquals(_slot1.Backend, backend) && ReferenceEquals(_slot1.Context, kvInput) && _slot1.KeyMh is not null)
            {
                keyMh = _slot1.KeyMh; valueMh = _slot1.ValueMh!;
                return true;
            }
        }
        keyMh = null!;
        valueMh = null!;
        return false;
    }

    public bool TryStore(IBackend backend, Tensor kvInput, Tensor keyMh, Tensor valueMh)
    {
        lock (_lock)
        {
            // This backend's own slot, else an empty one. A third backend stores nothing: evicting another backend's
            // slot could free K/V that backend's thread is attending over.
            bool useSlot0 = ReferenceEquals(_slot0.Backend, backend) || _slot0.Backend is null;
            if (!useSlot0 && !ReferenceEquals(_slot1.Backend, backend) && _slot1.Backend is not null)
                return false;
            if (useSlot0)
            {
                _slot0.KeyMh?.Dispose();
                _slot0.ValueMh?.Dispose();
                _slot0.Backend = backend;
                _slot0.Context = kvInput;
                _slot0.KeyMh = keyMh;
                _slot0.ValueMh = valueMh;
            }
            else
            {
                _slot1.KeyMh?.Dispose();
                _slot1.ValueMh?.Dispose();
                _slot1.Backend = backend;
                _slot1.Context = kvInput;
                _slot1.KeyMh = keyMh;
                _slot1.ValueMh = valueMh;
            }
        }
        return true;
    }
}
