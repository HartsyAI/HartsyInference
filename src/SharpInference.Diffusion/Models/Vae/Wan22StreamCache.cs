using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Vae;

/// <summary>Per-conv temporal cache for streaming Wan2.2 VAE decode (the <c>feat_cache</c> / <c>feat_idx</c> machine from <c>vae2_2.py</c>). The decode driver processes one latent frame per call; each <c>CausalConv3d</c> stashes the last <c>CACHE_T=2</c> frames of its input here so the next frame's causal convolution sees real left-context instead of zero padding — giving temporal continuity across the per-frame chunks without ever materializing the whole clip.
///
/// <para>Slots are addressed by forward-traversal order (auto-sized on the first frame): call <see cref="NewFrame"/> before each latent frame, then <see cref="StepConv"/> / <see cref="StepTimeConv"/> in the exact same order every frame. A slot holds <c>null</c> (unused), the <see cref="RepMarker"/> sentinel (Resample temporal first-chunk), or a Tensor (the cached frames).</para></summary>
public sealed unsafe class Wan22StreamCache : IDisposable
{
    private const int CacheT = 2;
    /// <summary>Sentinel marking a Resample temporal slot that has been initialized but not yet cached (the upstream "Rep" string).</summary>
    private static readonly object RepMarker = new();

    private readonly List<object?> _slots = new();
    private int _idx;
    private int _disposed;

    /// <summary>Resets the slot index — call once before decoding each latent frame.</summary>
    public void NewFrame() => _idx = 0;

    private int Next()
    {
        int idx = _idx++;
        while (_slots.Count <= idx) _slots.Add(null);
        return idx;
    }

    /// <summary>Standard residual/decoder/head <c>CausalConv3d</c>: returns the previous frame's cache (to feed <c>conv.Forward(x, cache)</c>, then dispose it) and stashes this frame's last frames. Returns null on the first frame (zero-pad).</summary>
    public Tensor? StepConv(Tensor convInput)
    {
        int idx = Next();
        Tensor? prev = _slots[idx] as Tensor;
        Tensor cacheX = Vae3dLayout.SliceFrames(convInput, FramesOf(convInput) - Math.Min(CacheT, FramesOf(convInput)), Math.Min(CacheT, FramesOf(convInput)));
        if (FramesOf(cacheX) < 2 && prev is not null)
        {
            Tensor merged = Vae3dLayout.PrependLastFrameOf(prev, cacheX);
            cacheX.Dispose();
            cacheX = merged;
        }
        _slots[idx] = cacheX;   // prev (old slot) is returned to the caller, who disposes it after the conv
        return prev;
    }

    /// <summary>Resample temporal <c>time_conv</c>: on the first frame, marks the slot "Rep" and returns <c>(skip:true)</c> so the caller skips the temporal conv entirely (image / first chunk). Afterwards returns the cache (or null when transitioning from "Rep") to feed <c>time_conv.Forward(x, cache)</c>.</summary>
    public (bool Skip, Tensor? Cache) StepTimeConv(Tensor convInput)
    {
        int idx = Next();
        object? slot = _slots[idx];
        if (slot is null)
        {
            _slots[idx] = RepMarker;
            return (true, null);
        }

        bool isRep = ReferenceEquals(slot, RepMarker);
        int avail = Math.Min(CacheT, FramesOf(convInput));
        Tensor cacheX = Vae3dLayout.SliceFrames(convInput, FramesOf(convInput) - avail, avail);
        if (FramesOf(cacheX) < 2 && !isRep)
        {
            Tensor merged = Vae3dLayout.PrependLastFrameOf((Tensor)slot, cacheX);
            cacheX.Dispose();
            cacheX = merged;
        }
        else if (FramesOf(cacheX) < 2 && isRep)
        {
            Tensor merged = Vae3dLayout.PrependZeroFrame(cacheX);
            cacheX.Dispose();
            cacheX = merged;
        }

        Tensor? convCache = isRep ? null : (Tensor)slot;
        _slots[idx] = cacheX;
        return (false, convCache);
    }

    private static int FramesOf(Tensor t) => (int)t.Shape[2];

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (object? s in _slots) (s as Tensor)?.Dispose();
        _slots.Clear();
        GC.SuppressFinalize(this);
    }
}
