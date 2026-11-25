using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>The per-layer self-attention keys and values of an autoregressive video model that denoises one latent frame at a
/// time, kept frame by frame: a slot holds the frame's already normalized and rotated K and V as the attention layer produced
/// them. A frame's pass stores its own slot (<see cref="RewriteFor"/>) and attends over slots 0 through the frame, so the same
/// code serves the first frame, a later frame and a re-pass at timestep 0. A model that replaces a finished frame's entries (a
/// refinement network) writes the slot again with <see cref="Set"/>. Slots keep the dtype and shape the attention path
/// produced, which is fixed by the backend and the branch dtype, so one cache is used with one path. Slots are pinned as
/// activations so they survive the per-step activation sweep; disposing unpins and frees them.</summary>
public sealed class FrameKvCache : IDisposable
{
    private readonly KvSegmentStore _store;

    /// <summary>A cache of <paramref name="layers"/> layers with room for <paramref name="maxFrames"/> frame slots each.</summary>
    public FrameKvCache(IBackend backend, int layers, int maxFrames)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxFrames, 1);
        _store = new KvSegmentStore(backend, layers, tokenAxis: 0);
        MaxFrames = maxFrames;
    }

    /// <summary>Layers the cache spans.</summary>
    public int Layers => _store.Layers;

    /// <summary>Frame slots per layer.</summary>
    public int MaxFrames { get; }

    /// <summary>Bytes a full cache takes: keys and values of <paramref name="tokensPerFrame"/> × <paramref name="hidden"/> for every
    /// frame of every layer.</summary>
    public static long EstimateBytes(int layers, int frames, int tokensPerFrame, int hidden, DType dtype) =>
        KvSegmentStore.EstimateBytes(layers, (long)frames * tokensPerFrame, hidden, dtype);

    /// <summary>True when slot <paramref name="frame"/> of <paramref name="layer"/> holds a key and a value.</summary>
    public bool Has(int layer, int frame)
    {
        RequireFrame(frame);
        return _store.Has(layer, frame);
    }

    /// <summary>The key stored for <paramref name="layer"/> and <paramref name="frame"/>; borrowed, never disposed by the caller.</summary>
    public Tensor Keys(int layer, int frame) => _store.Keys(layer, frame);

    /// <summary>The value stored for <paramref name="layer"/> and <paramref name="frame"/>; borrowed.</summary>
    public Tensor Values(int layer, int frame) => _store.Values(layer, frame);

    /// <summary>Writes slot <paramref name="frame"/> of <paramref name="layer"/>, taking ownership of both tensors (also when it
    /// throws) and freeing the entries it replaces.</summary>
    /// <exception cref="ArgumentException">The two tensors differ in shape or dtype, or are not token-major <c>[S, C]</c>.</exception>
    public void Set(int layer, int frame, Tensor keys, Tensor values)
    {
        ArgumentNullException.ThrowIfNull(keys);
        ArgumentNullException.ThrowIfNull(values);
        if (frame < 0 || frame >= MaxFrames || keys.Shape.Rank != 2)
        {
            TensorShape shape = keys.Shape;
            keys.Dispose();
            values.Dispose();
            RequireFrame(frame);
            throw new ArgumentException($"A slot is token-major [S, C]; got {shape}.");
        }
        _store.Set(layer, frame, keys, values);
    }

    /// <summary>Joins slots 0 through <paramref name="lastFrame"/> of <paramref name="layer"/> along the token axis into fresh
    /// tensors the caller owns. The slots are not disposed.</summary>
    /// <exception cref="InvalidOperationException">A slot in the range is empty.</exception>
    public (Tensor Keys, Tensor Values) JoinPrefix(int layer, int lastFrame) => _store.Join(layer, 0, lastFrame + 1);

    /// <summary>The attention rewrite of one block's pass over frame <paramref name="frame"/>: it stores the live keys and values
    /// as the frame's slot and returns the joined slots 0 through the frame for the attention to read.</summary>
    public IKvRewrite RewriteFor(int layer, int frame)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(layer);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(layer, Layers);
        ArgumentOutOfRangeException.ThrowIfNegative(frame);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(frame, MaxFrames);
        return new FrameRewrite(this, layer, frame);
    }

    private void RequireFrame(int frame)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(frame);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(frame, MaxFrames);
    }

    /// <inheritdoc/>
    public void Dispose() => _store.Dispose();

    private sealed class FrameRewrite(FrameKvCache cache, int layer, int frame) : IKvRewrite
    {
        public (Tensor K, Tensor V) Rewrite(IBackend backend, Tensor k, Tensor v)
        {
            cache.Set(layer, frame, k, v);
            return cache.JoinPrefix(layer, frame);
        }
    }
}
