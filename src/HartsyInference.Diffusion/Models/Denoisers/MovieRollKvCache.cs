using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>MovieRoll's streaming KV cache (<c>KVCacheManager</c> in the reference): one merged K and V per double-stream block, plus the latent
/// frame range of every chunk appended to it. Each layer's K and V are <c>[1, tokens, heads, headDim]</c>, with <c>tokens = frames·H·W</c>.
/// Chunks are appended and never popped, except by <see cref="TrimToSink"/> (HieraKV's sink-recent blocks) and by <see cref="ReplaceLayer"/> and
/// <see cref="SoftClear"/> (KV recompute).
///
/// <para>The cache owns the tensors it stores. Each stored tensor is pinned with <see cref="IBackend.PinActivation"/>, so the per-step activation
/// sweep leaves it on the device, and it is unpinned before it is disposed. Inputs to <see cref="Update"/> and <see cref="ReplaceLayer"/> are
/// copied, so the caller keeps its own. <see cref="Get"/> returns the stored tensors without copying: they stay valid until that layer changes.</para>
///
/// <para>This is not built on <see cref="FrameKvCache"/>. That cache keeps one fixed slot per frame and attends over slots 0 through the current frame.
/// This cache keeps variable-size chunks with their frame ranges as one contiguous token history, and its sink trim drops the middle chunks,
/// which a slot-per-frame layout does not express.</para>
///
/// <para>Each <see cref="Update"/> re-concatenates the whole stored history, as the reference's <c>torch.cat</c> does. That copy grows with the
/// chunk count, which <see cref="ShouldRecompute"/> bounds.</para>
/// </summary>
public sealed class MovieRollKvCache : IDisposable
{
    private readonly IBackend _backend;
    private readonly Dictionary<string, Entry> _layers = new(StringComparer.Ordinal);
    private int _tokensPerFrame;
    private int _height, _width;
    private string? _firstLayer;
    private bool _disposed;

    private sealed class Entry
    {
        public Tensor? Keys;
        public Tensor? Values;
        public readonly List<(int Start, int End)> Frames = [];
    }

    /// <summary>A cache whose stored tensors live on <paramref name="backend"/>; it is held for the pins and unpins.</summary>
    public MovieRollKvCache(IBackend backend)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
    }

    /// <summary>Whether <see cref="Init"/> has run (and <see cref="Clear"/> has not).</summary>
    public bool IsInit { get; private set; }

    /// <summary>The latent grid the cache was initialised for: every chunk carries <c>Height × Width</c> tokens per frame.</summary>
    public int Height { get { RequireInit(); return _height; } }

    /// <summary>The latent grid width the cache was initialised for.</summary>
    public int Width { get { RequireInit(); return _width; } }

    /// <summary>The first layer's name: the layer whose chunk count drives <see cref="ShouldRecompute"/>.</summary>
    public string? FirstLayer => _firstLayer;

    /// <summary>Starts the cache for <paramref name="layers"/> with a token grid of <paramref name="height"/>·<paramref name="width"/> per frame.
    /// Everything is validated before the cache changes, so a failed call leaves the previous state in place.</summary>
    public void Init(IEnumerable<string> layers, int height, int width)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(layers);
        if (height <= 0 || width <= 0)
            throw new ArgumentOutOfRangeException(nameof(height), $"The token grid must be positive; got {height}x{width}.");
        List<string> names = layers.ToList();
        if (names.Count == 0) throw new ArgumentException("A KV cache needs at least one layer.", nameof(layers));
        if (names.Any(string.IsNullOrEmpty)) throw new ArgumentException("Layer names must be non-empty.", nameof(layers));
        if (names.Distinct(StringComparer.Ordinal).Count() != names.Count)
            throw new ArgumentException("Layer names must be unique.", nameof(layers));

        ClearLayers();
        foreach (string name in names) _layers.Add(name, new Entry());
        _firstLayer = names[0];
        _tokensPerFrame = height * width;
        _height = height;
        _width = width;
        IsInit = true;
    }

    /// <summary>Appends one chunk to a layer: <paramref name="keys"/> and <paramref name="values"/> cover latent frames <c>[start, end)</c>, so their
    /// token count must be <c>(end − start)·H·W</c>, and their dtype, batch, head and head-dim extents must match the chunks already stored.
    /// If either copy fails, the layer keeps its previous state.</summary>
    public void Update(string layer, Tensor keys, Tensor values, int start, int end)
    {
        RequireInit();
        Entry entry = Layer(layer);
        RequireKeyValuePair(keys, values);
        if (start < 0 || end <= start)
            throw new ArgumentOutOfRangeException(nameof(end), $"The frame range [{start}, {end}) must be non-empty and start at 0 or later.");
        if (entry.Frames.Count > 0 && start < entry.Frames[entry.Frames.Count - 1].End)
            throw new ArgumentOutOfRangeException(nameof(start), start, $"Chunk [{start}, {end}) starts before the last cached chunk ends at {entry.Frames[entry.Frames.Count - 1].End}; chunks append in frame order.");
        long expectedTokens = (long)(end - start) * _tokensPerFrame;
        if (keys.Shape[1] != expectedTokens)
            throw new ArgumentException($"Chunk [{start}, {end}) carries {keys.Shape[1]} tokens; {expectedTokens} are expected ({end - start} frames of {_tokensPerFrame}).", nameof(keys));
        RequireCompatible(entry, keys, layer);

        Tensor newKeys = Concat(entry.Keys, keys);
        Tensor newValues;
        try
        {
            newValues = Concat(entry.Values, values);
        }
        catch
        {
            newKeys.Dispose();
            throw;
        }
        Release(entry.Keys);
        Release(entry.Values);
        entry.Keys = newKeys;
        entry.Values = newValues;
        entry.Frames.Add((start, end));
    }

    /// <summary>HieraKV's sink-recent layers: keeps the first <paramref name="sinkChunks"/> chunks of <paramref name="layer"/> and drops the rest, so the
    /// next chunk attends to the sink only. A layer with no more chunks than that is left alone. Keeping zero chunks leaves the layer empty, as
    /// <see cref="SoftClear"/> does.</summary>
    public void TrimToSink(string layer, int sinkChunks)
    {
        RequireInit();
        if (sinkChunks < 0) throw new ArgumentOutOfRangeException(nameof(sinkChunks));
        Entry entry = Layer(layer);
        if (entry.Frames.Count <= sinkChunks) return;
        int keepTokens = 0;
        for (int i = 0; i < sinkChunks; i++) keepTokens += (entry.Frames[i].End - entry.Frames[i].Start) * _tokensPerFrame;
        entry.Frames.RemoveRange(sinkChunks, entry.Frames.Count - sinkChunks);
        if (keepTokens == 0)
        {
            Release(entry.Keys);
            Release(entry.Values);
            entry.Keys = null;
            entry.Values = null;
            return;
        }
        Tensor newKeys = Prefix(entry.Keys!, keepTokens);
        Tensor newValues;
        try
        {
            newValues = Prefix(entry.Values!, keepTokens);
        }
        catch
        {
            newKeys.Dispose();
            throw;
        }
        Release(entry.Keys);
        Release(entry.Values);
        entry.Keys = newKeys;
        entry.Values = newValues;
    }

    /// <summary>Replaces a layer's stored K, V and frame ranges (KV recompute rebuilds them). The arrays are copied. Every range must be non-empty
    /// and start at 0 or later, and the ranges must hold exactly the tokens of the arrays. Nothing is replaced when a check fails.</summary>
    public void ReplaceLayer(string layer, Tensor keys, Tensor values, IReadOnlyList<(int Start, int End)> frames)
    {
        RequireInit();
        ArgumentNullException.ThrowIfNull(frames);
        Entry entry = Layer(layer);
        RequireKeyValuePair(keys, values);
        if (frames.Count == 0) throw new ArgumentException("A replaced layer needs at least one frame range.", nameof(frames));
        long expectedTokens = 0;
        foreach ((int start, int end) in frames)
        {
            if (start < 0 || end <= start)
                throw new ArgumentOutOfRangeException(nameof(frames), $"The frame range [{start}, {end}) must be non-empty and start at 0 or later.");
            expectedTokens += (long)(end - start) * _tokensPerFrame;
        }
        if (keys.Shape[1] != expectedTokens)
            throw new ArgumentException($"The frames carry {expectedTokens} tokens; the arrays hold {keys.Shape[1]}.", nameof(keys));

        Tensor newKeys = Clone(keys);
        Tensor newValues;
        try
        {
            newValues = Clone(values);
        }
        catch
        {
            newKeys.Dispose();
            throw;
        }
        Release(entry.Keys);
        Release(entry.Values);
        entry.Keys = newKeys;
        entry.Values = newValues;
        entry.Frames.Clear();
        entry.Frames.AddRange(frames);
    }

    /// <summary>The stored K and V of a layer, or null when the layer holds no chunk. Not copied: valid until that layer changes.</summary>
    public (Tensor Keys, Tensor Values)? Get(string layer)
    {
        RequireInit();
        Entry entry = Layer(layer);
        return entry.Keys is null || entry.Values is null ? null : (entry.Keys, entry.Values);
    }

    /// <summary>The frame ranges of a layer's chunks, in order.</summary>
    public IReadOnlyList<(int Start, int End)> FrameRanges(string layer)
    {
        RequireInit();
        return Layer(layer).Frames.AsReadOnly();
    }

    /// <summary>The latent frame count of each cached chunk of a layer, in order (the cond chunk sizes of the next forward).</summary>
    public IReadOnlyList<int> CachedChunkSizes(string layer)
    {
        RequireInit();
        return Layer(layer).Frames.Select(f => f.End - f.Start).ToArray();
    }

    /// <summary>Whether the first layer holds at least <paramref name="threshold"/> chunks: the KV recompute trigger.</summary>
    public bool ShouldRecompute(int threshold)
    {
        ThrowIfDisposed();
        if (threshold <= 0) throw new ArgumentOutOfRangeException(nameof(threshold), "The recompute threshold must be positive.");
        if (!IsInit || _firstLayer is null) return false;
        return _layers[_firstLayer].Frames.Count >= threshold;
    }

    /// <summary>Drops every layer's stored tensors and frame ranges, keeping the layers and the token grid (KV recompute starts from here).</summary>
    public void SoftClear()
    {
        RequireInit();
        foreach (Entry entry in _layers.Values)
        {
            Release(entry.Keys);
            Release(entry.Values);
            entry.Keys = null;
            entry.Values = null;
            entry.Frames.Clear();
        }
    }

    /// <summary>Ends the cache: unpins and disposes every stored tensor and forgets the layers, so <see cref="Init"/> can run again.</summary>
    public void Clear()
    {
        ThrowIfDisposed();
        ClearLayers();
        IsInit = false;
        _firstLayer = null;
        _tokensPerFrame = 0;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        ClearLayers();
        IsInit = false;
        _firstLayer = null;
        _disposed = true;
    }

    private void ClearLayers()
    {
        foreach (Entry entry in _layers.Values)
        {
            Release(entry.Keys);
            Release(entry.Values);
        }
        _layers.Clear();
    }

    private Entry Layer(string layer)
    {
        if (!_layers.TryGetValue(layer, out Entry? entry))
            throw new KeyNotFoundException($"The KV cache has no layer '{layer}'.");
        return entry;
    }

    private void RequireInit()
    {
        ThrowIfDisposed();
        if (!IsInit) throw new InvalidOperationException("The KV cache is not initialized; call Init first.");
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(MovieRollKvCache));
    }

    /// <summary>Checks rank, batch, dtype and shape of one K/V pair (the values must match the keys exactly).</summary>
    private static void RequireKeyValuePair(Tensor keys, Tensor values)
    {
        if (keys.Shape.Rank != 4 || keys.Shape[0] != 1)
            throw new ArgumentException($"KV arrays are [1, tokens, heads, headDim]; got {keys.Shape}.", nameof(keys));
        if (keys.Shape != values.Shape)
            throw new ArgumentException($"Keys {keys.Shape} and values {values.Shape} must match.", nameof(values));
        if (keys.DType != values.DType)
            throw new ArgumentException($"Keys are {keys.DType} and values are {values.DType}; they must match.", nameof(values));
    }

    /// <summary>A new chunk must match the stored chunks' dtype, batch, heads and head dimension.</summary>
    private static void RequireCompatible(Entry entry, Tensor keys, string layer)
    {
        if (entry.Keys is null) return;
        if (entry.Keys.DType != keys.DType || entry.Keys.Shape[0] != keys.Shape[0]
            || entry.Keys.Shape[2] != keys.Shape[2] || entry.Keys.Shape[3] != keys.Shape[3])
            throw new ArgumentException($"KV mismatch for {layer}: stored {entry.Keys.Shape} {entry.Keys.DType}, new {keys.Shape} {keys.DType}.", nameof(keys));
    }

    /// <summary>Unpins and disposes a stored tensor; null is ignored.</summary>
    private void Release(Tensor? stored)
    {
        if (stored is null) return;
        _backend.UnpinActivation(stored);
        stored.Dispose();
    }

    /// <summary>A pinned copy of <paramref name="source"/>, owned by the cache.</summary>
    private Tensor Clone(Tensor source)
    {
        Tensor copy = new(source.Shape, source.DType);
        StridedCopies.SliceAxis(_backend, copy, source, 1, 0, source.Shape[1]);
        _backend.PinActivation(copy);
        return copy;
    }

    /// <summary>The first <paramref name="tokens"/> tokens of <paramref name="source"/> as a new pinned tensor. <paramref name="source"/> is not released:
    /// the caller releases it once the replacement is in place.</summary>
    private Tensor Prefix(Tensor source, int tokens)
    {
        Tensor output = new(new TensorShape([1L, tokens, source.Shape[2], source.Shape[3]]), source.DType);
        StridedCopies.SliceAxis(_backend, output, source, 1, 0, tokens);
        _backend.PinActivation(output);
        return output;
    }

    /// <summary>A new pinned tensor holding <paramref name="stored"/> followed by <paramref name="chunk"/>, or a copy of <paramref name="chunk"/>
    /// when nothing is stored. Neither input is released: the caller swaps the layer once both halves succeed.</summary>
    private Tensor Concat(Tensor? stored, Tensor chunk)
    {
        if (stored is null) return Clone(chunk);
        int tokens = (int)(stored.Shape[1] + chunk.Shape[1]);
        Tensor output = new(new TensorShape([1L, tokens, chunk.Shape[2], chunk.Shape[3]]), chunk.DType);
        _backend.Concat(output, new[] { stored, chunk }, dim: 1);
        _backend.PinActivation(output);
        return output;
    }
}
