using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>MovieRoll's streaming KV cache (<c>KVCacheManager</c> in the reference): one merged K and V per double-stream block, plus the latent
/// frame range of every chunk appended to it. Each layer's K and V are <c>[B, tokens, heads, headDim]</c>, with <c>tokens = frames·H·W</c>.
/// Chunks are appended and never popped, except by <see cref="TrimToSink"/> (HieraKV's sink-recent blocks) and by <see cref="ReplaceLayer"/> and
/// <see cref="SoftClear"/> (KV recompute).
///
/// <para>The cache owns the tensors it stores. Inputs to <see cref="Update"/> and <see cref="ReplaceLayer"/> are copied, so the caller keeps its own.
/// <see cref="Get"/> returns the stored tensors without copying: they stay valid until the next mutation of that layer.</para>
/// </summary>
public sealed class MovieRollKvCache : IDisposable
{
    private readonly Dictionary<string, Entry> _layers = new(StringComparer.Ordinal);
    private int _tokensPerFrame;
    private string? _firstLayer;
    private bool _disposed;

    private sealed class Entry
    {
        public Tensor? Keys;
        public Tensor? Values;
        public readonly List<(int Start, int End)> Frames = [];
    }

    /// <summary>Whether <see cref="Init"/> has run (and <see cref="Clear"/> has not).</summary>
    public bool IsInit { get; private set; }

    /// <summary>The first layer's name: the layer whose chunk count drives <see cref="ShouldRecompute"/>.</summary>
    public string? FirstLayer => _firstLayer;

    /// <summary>Starts the cache for <paramref name="layers"/> with a token grid of <paramref name="height"/>·<paramref name="width"/> per frame.</summary>
    public void Init(IEnumerable<string> layers, int height, int width)
    {
        ArgumentNullException.ThrowIfNull(layers);
        if (height <= 0 || width <= 0)
            throw new ArgumentOutOfRangeException(nameof(height), $"The token grid must be positive; got {height}x{width}.");
        ClearLayers();
        foreach (string name in layers)
        {
            if (_layers.Count == 0) _firstLayer = name;
            _layers.Add(name, new Entry());
        }
        if (_layers.Count == 0) throw new ArgumentException("A KV cache needs at least one layer.", nameof(layers));
        _tokensPerFrame = height * width;
        IsInit = true;
    }

    /// <summary>Appends one chunk to a layer: <paramref name="keys"/> and <paramref name="values"/> cover latent frames <c>[start, end)</c>, so their
    /// token count must be <c>(end − start)·H·W</c>, and their batch, head and head-dim extents must match the layer's chunks already stored.</summary>
    public void Update(IBackend backend, string layer, Tensor keys, Tensor values, int start, int end)
    {
        RequireInit();
        ArgumentNullException.ThrowIfNull(backend);
        Entry entry = Layer(layer);
        RequireSameShape(keys, values);
        if (end <= start || start < 0)
            throw new ArgumentOutOfRangeException(nameof(end), $"The frame range [{start}, {end}) is empty or negative.");
        long expectedTokens = (long)(end - start) * _tokensPerFrame;
        if (keys.Shape[1] != expectedTokens)
            throw new ArgumentException($"Chunk [{start}, {end}) carries {keys.Shape[1]} tokens; {expectedTokens} are expected ({end - start} frames of {_tokensPerFrame}).", nameof(keys));
        if (entry.Keys is not null && (entry.Keys.Shape[0] != keys.Shape[0] || entry.Keys.Shape[2] != keys.Shape[2] || entry.Keys.Shape[3] != keys.Shape[3]))
            throw new ArgumentException($"KV dimension mismatch for {layer}: stored {entry.Keys.Shape}, new {keys.Shape}.", nameof(keys));

        entry.Keys = Append(backend, entry.Keys, keys);
        entry.Values = Append(backend, entry.Values, values);
        entry.Frames.Add((start, end));
    }

    /// <summary>HieraKV's sink-recent layers: keeps the first <paramref name="sinkChunks"/> chunks of <paramref name="layer"/> and drops the rest, so the
    /// next chunk attends to the sink only. A layer with no more chunks than that is left alone.</summary>
    public void TrimToSink(IBackend backend, string layer, int sinkChunks)
    {
        RequireInit();
        ArgumentNullException.ThrowIfNull(backend);
        if (sinkChunks < 0) throw new ArgumentOutOfRangeException(nameof(sinkChunks));
        Entry entry = Layer(layer);
        if (entry.Frames.Count <= sinkChunks) return;
        int keepTokens = 0;
        for (int i = 0; i < sinkChunks; i++) keepTokens += (entry.Frames[i].End - entry.Frames[i].Start) * _tokensPerFrame;
        entry.Frames.RemoveRange(sinkChunks, entry.Frames.Count - sinkChunks);
        if (entry.Keys is not null) entry.Keys = Prefix(backend, entry.Keys, keepTokens);
        if (entry.Values is not null) entry.Values = Prefix(backend, entry.Values, keepTokens);
    }

    /// <summary>Replaces a layer's stored K, V and frame ranges (KV recompute rebuilds them). The arrays are copied.</summary>
    public void ReplaceLayer(IBackend backend, string layer, Tensor keys, Tensor values, IReadOnlyList<(int Start, int End)> frames)
    {
        RequireInit();
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(frames);
        Entry entry = Layer(layer);
        RequireSameShape(keys, values);
        long expectedTokens = frames.Sum(f => (long)(f.End - f.Start) * _tokensPerFrame);
        if (keys.Shape[1] != expectedTokens)
            throw new ArgumentException($"The frames carry {expectedTokens} tokens; the arrays hold {keys.Shape[1]}.", nameof(keys));
        Tensor newKeys = Clone(backend, keys), newValues = Clone(backend, values);
        entry.Keys?.Dispose();
        entry.Values?.Dispose();
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
    public IReadOnlyList<(int Start, int End)> FrameRanges(string layer) => Layer(layer).Frames;

    /// <summary>The latent frame count of each cached chunk of a layer, in order (the cond chunk sizes of the next forward).</summary>
    public IReadOnlyList<int> CachedChunkSizes(string layer) => Layer(layer).Frames.Select(f => f.End - f.Start).ToArray();

    /// <summary>Whether the first layer holds at least <paramref name="threshold"/> chunks: the KV recompute trigger.</summary>
    public bool ShouldRecompute(int threshold)
    {
        if (!IsInit || _firstLayer is null) return false;
        return _layers[_firstLayer].Frames.Count >= threshold;
    }

    /// <summary>Drops every layer's stored tensors and frame ranges, keeping the layers and the token grid (KV recompute starts from here).</summary>
    public void SoftClear()
    {
        RequireInit();
        foreach (Entry entry in _layers.Values)
        {
            entry.Keys?.Dispose();
            entry.Values?.Dispose();
            entry.Keys = null;
            entry.Values = null;
            entry.Frames.Clear();
        }
    }

    /// <summary>Ends the cache: disposes every stored tensor and forgets the layers, so <see cref="Init"/> can run again.</summary>
    public void Clear()
    {
        ClearLayers();
        IsInit = false;
        _firstLayer = null;
        _tokensPerFrame = 0;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        ClearLayers();
    }

    private void ClearLayers()
    {
        foreach (Entry entry in _layers.Values)
        {
            entry.Keys?.Dispose();
            entry.Values?.Dispose();
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
        if (!IsInit) throw new InvalidOperationException("The KV cache is not initialized; call Init first.");
    }

    private static void RequireSameShape(Tensor keys, Tensor values)
    {
        if (keys.Shape.Rank != 4 || keys.Shape[0] != 1)
            throw new ArgumentException($"KV arrays are [1, tokens, heads, headDim]; got {keys.Shape}.", nameof(keys));
        if (keys.Shape != values.Shape)
            throw new ArgumentException($"Keys {keys.Shape} and values {values.Shape} must match.", nameof(values));
    }

    /// <summary>A new tensor holding <paramref name="source"/> (copied on the backend).</summary>
    private static Tensor Clone(IBackend backend, Tensor source)
    {
        Tensor copy = new(source.Shape, source.DType);
        StridedCopies.SliceAxis(backend, copy, source, 1, 0, source.Shape[1]);
        return copy;
    }

    /// <summary>The first <paramref name="tokens"/> tokens of <paramref name="source"/> as a new tensor.</summary>
    private static Tensor Prefix(IBackend backend, Tensor source, int tokens)
    {
        Tensor output = new(new TensorShape([1L, tokens, source.Shape[2], source.Shape[3]]), source.DType);
        StridedCopies.SliceAxis(backend, output, source, 1, 0, tokens);
        source.Dispose();
        return output;
    }

    /// <summary>A new tensor with <paramref name="chunk"/> appended after <paramref name="stored"/> (or a copy when nothing is stored yet).
    /// Disposes <paramref name="stored"/>.</summary>
    private static Tensor Append(IBackend backend, Tensor? stored, Tensor chunk)
    {
        if (stored is null) return Clone(backend, chunk);
        int tokens = (int)(stored.Shape[1] + chunk.Shape[1]);
        Tensor output = new(new TensorShape([1L, tokens, chunk.Shape[2], chunk.Shape[3]]), chunk.DType);
        backend.Concat(output, new[] { stored, chunk }, dim: 1);
        stored.Dispose();
        return output;
    }
}
