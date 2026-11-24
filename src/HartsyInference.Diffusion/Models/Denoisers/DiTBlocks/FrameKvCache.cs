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
    private readonly IBackend _backend;
    private readonly Tensor?[][] _keys;
    private readonly Tensor?[][] _values;
    private int _disposed;
    private int _channels = -1;

    /// <summary>A cache of <paramref name="layers"/> layers with room for <paramref name="maxFrames"/> frame slots each.</summary>
    public FrameKvCache(IBackend backend, int layers, int maxFrames)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentOutOfRangeException.ThrowIfLessThan(layers, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxFrames, 1);
        _backend = backend;
        _keys = new Tensor?[layers][];
        _values = new Tensor?[layers][];
        for (int i = 0; i < layers; i++)
        {
            _keys[i] = new Tensor?[maxFrames];
            _values[i] = new Tensor?[maxFrames];
        }
    }

    /// <summary>Layers the cache spans.</summary>
    public int Layers => _keys.Length;

    /// <summary>Frame slots per layer.</summary>
    public int MaxFrames => _keys[0].Length;

    /// <summary>Bytes a full cache takes: keys and values of <paramref name="tokensPerFrame"/> × <paramref name="hidden"/> for every
    /// frame of every layer.</summary>
    public static long EstimateBytes(int layers, int frames, int tokensPerFrame, int hidden, DType dtype) =>
        2L * layers * frames * tokensPerFrame * hidden * dtype.SizeInBytes;

    /// <summary>True when slot <paramref name="frame"/> of <paramref name="layer"/> holds a key and a value.</summary>
    public bool Has(int layer, int frame) => _keys[layer][frame] is not null;

    /// <summary>The key stored for <paramref name="layer"/> and <paramref name="frame"/>; borrowed, never disposed by the caller.</summary>
    public Tensor Keys(int layer, int frame) => _keys[layer][frame] ?? throw Empty(layer, frame);

    /// <summary>The value stored for <paramref name="layer"/> and <paramref name="frame"/>; borrowed.</summary>
    public Tensor Values(int layer, int frame) => _values[layer][frame] ?? throw Empty(layer, frame);

    /// <summary>Writes slot <paramref name="frame"/> of <paramref name="layer"/>, taking ownership of both tensors (also when it
    /// throws) and freeing the entries it replaces.</summary>
    /// <exception cref="ArgumentException">The two tensors differ in shape or dtype.</exception>
    public void Set(int layer, int frame, Tensor keys, Tensor values)
    {
        ArgumentNullException.ThrowIfNull(keys);
        ArgumentNullException.ThrowIfNull(values);
        if (keys.Shape != values.Shape || keys.DType != values.DType)
        {
            keys.Dispose();
            values.Dispose();
            throw new ArgumentException($"Keys {keys.Shape} {keys.DType} and values {values.Shape} {values.DType} must match.");
        }
        if (keys.Shape.Rank != 2)
        {
            keys.Dispose();
            values.Dispose();
            throw new ArgumentException($"A slot is token-major [S, C]; got {keys.Shape}.");
        }
        if (_channels >= 0 && keys.Shape[1] != _channels)
        {
            keys.Dispose();
            values.Dispose();
            throw new ArgumentException($"The slots hold {_channels} channels; this one holds {keys.Shape[1]}.");
        }
        if (Volatile.Read(ref _disposed) != 0)
        {
            keys.Dispose();
            values.Dispose();
            throw new ObjectDisposedException(nameof(FrameKvCache));
        }
        _channels = (int)keys.Shape[1];
        Release(layer, frame);
        _backend.PinActivation(keys);
        _backend.PinActivation(values);
        _keys[layer][frame] = keys;
        _values[layer][frame] = values;
    }

    /// <summary>Joins slots 0 through <paramref name="lastFrame"/> of <paramref name="layer"/> along the token axis into fresh
    /// tensors the caller owns. The slots are not disposed.</summary>
    /// <exception cref="InvalidOperationException">A slot in the range is empty.</exception>
    public (Tensor Keys, Tensor Values) JoinPrefix(int layer, int lastFrame)
    {
        Tensor[] keys = new Tensor[lastFrame + 1];
        Tensor[] values = new Tensor[lastFrame + 1];
        for (int f = 0; f <= lastFrame; f++)
        {
            keys[f] = Keys(layer, f);
            values[f] = Values(layer, f);
        }
        Tensor joinedKeys = Join(keys);
        try
        {
            return (joinedKeys, Join(values));
        }
        catch
        {
            joinedKeys.Dispose();
            throw;
        }
    }

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

    private Tensor Join(Tensor[] parts)
    {
        // Set enforces token-major [S, C] slots with one channel count, so the prefix joins on the token axis, axis 0.
        long tokens = 0;
        foreach (Tensor part in parts)
            tokens += part.Shape[0];
        Tensor joined = new(new TensorShape([tokens, parts[0].Shape[1]]), parts[0].DType);
        try
        {
            _backend.Concat(joined, parts, 0);
        }
        catch
        {
            joined.Dispose();
            throw;
        }
        return joined;
    }

    private void Release(int layer, int frame)
    {
        foreach (Tensor?[] slots in new[] { _keys[layer], _values[layer] })
        {
            if (slots[frame] is not Tensor old)
                continue;
            _backend.UnpinActivation(old);
            old.Dispose();
            slots[frame] = null;
        }
    }

    private static InvalidOperationException Empty(int layer, int frame) =>
        new($"Layer {layer} has no keys and values stored for frame {frame}.");

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        for (int layer = 0; layer < Layers; layer++)
            for (int frame = 0; frame < MaxFrames; frame++)
                Release(layer, frame);
    }

    private sealed class FrameRewrite(FrameKvCache cache, int layer, int frame) : IKvRewrite
    {
        public (Tensor K, Tensor V) Rewrite(IBackend backend, Tensor k, Tensor v)
        {
            cache.Set(layer, frame, k, v);
            return cache.JoinPrefix(layer, frame);
        }
    }
}
