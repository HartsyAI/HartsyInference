using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Per-layer keys and values of a DiT's in-context reference tokens, computed once in a prepass and appended to
/// the live sequence's keys and values on every step (<see cref="ConcatTokens"/>). Each entry is token-major
/// <c>[1, S_ref, heads, headDim]</c>, keys already normalized and rotated; a layer the prepass skipped has none. Entries
/// are pinned as activations so they survive the per-step activation sweep; disposing unpins and frees them.</summary>
public sealed class ReferenceKvCache : IDisposable
{
    private readonly IBackend _backend;
    private readonly Tensor?[] _keys;
    private readonly Tensor?[] _values;
    private int _disposed;

    private ReferenceKvCache(IBackend backend, Tensor?[] keys, Tensor?[] values, int tokens, long fingerprint)
    {
        _backend = backend;
        _keys = keys;
        _values = values;
        Tokens = tokens;
        Fingerprint = fingerprint;
    }

    /// <summary>Reference tokens per entry.</summary>
    public int Tokens { get; }

    /// <summary>Layers the cache spans (skipped ones included, with no entry).</summary>
    public int Layers => _keys.Length;

    /// <summary>Identity of what the cache was built from, for step-graph signatures.</summary>
    public long Fingerprint { get; }

    /// <summary>Layer <paramref name="layer"/>'s keys, or null when the prepass skipped it.</summary>
    public Tensor? Keys(int layer) => _keys[layer];

    /// <summary>Layer <paramref name="layer"/>'s values, or null when the prepass skipped it.</summary>
    public Tensor? Values(int layer) => _values[layer];

    /// <summary>Bytes a cache of <paramref name="activeLayers"/> layers takes: keys and values of
    /// <paramref name="tokens"/> × <paramref name="hidden"/> each.</summary>
    public static long EstimateBytes(int activeLayers, int tokens, int hidden, DType dtype) =>
        2L * activeLayers * tokens * hidden * dtype.SizeInBytes;

    /// <summary>Returns <c>[1, S_live + S_ref, heads, headDim]</c>: <paramref name="live"/>'s tokens, then
    /// <paramref name="cached"/>'s. Neither input is disposed.</summary>
    public static Tensor ConcatTokens(IBackend backend, Tensor live, Tensor cached)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(live);
        ArgumentNullException.ThrowIfNull(cached);
        if (live.Shape.Rank != 4 || cached.Shape.Rank != 4 || live.Shape[0] != 1 || cached.Shape[0] != 1
            || live.Shape[2] != cached.Shape[2] || live.Shape[3] != cached.Shape[3] || live.DType != cached.DType)
        {
            throw new ArgumentException(
                $"Cached keys/values {cached.Shape} {cached.DType} cannot extend live {live.Shape} {live.DType}; both must be [1, S, heads, headDim] of one dtype.");
        }
        Tensor joined = new(new TensorShape(1, live.Shape[1] + cached.Shape[1], live.Shape[2], live.Shape[3]), live.DType);
        backend.Concat(joined, [live, cached], 1);
        return joined;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        Release(_backend, _keys);
        Release(_backend, _values);
    }

    private static void Release(IBackend backend, Tensor?[] entries)
    {
        for (int i = 0; i < entries.Length; i++)
        {
            if (entries[i] is not Tensor t)
                continue;
            backend.UnpinActivation(t);
            t.Dispose();
            entries[i] = null;
        }
    }

    /// <summary>Collects a prepass's per-layer keys and values; owns them until <see cref="Build"/> hands them over.</summary>
    public sealed class Builder(int layers) : IDisposable
    {
        private readonly Tensor?[] _keys = new Tensor?[layers];
        private readonly Tensor?[] _values = new Tensor?[layers];
        private int _tokens = -1;
        private bool _built;

        /// <summary>Takes ownership of layer <paramref name="layer"/>'s keys and values.</summary>
        public void Set(int layer, Tensor keys, Tensor values)
        {
            ArgumentNullException.ThrowIfNull(keys);
            ArgumentNullException.ThrowIfNull(values);
            if (_built)
                throw new InvalidOperationException("The cache was already built.");
            if (keys.Shape.Rank != 4 || keys.Shape != values.Shape || keys.DType != values.DType)
                throw new ArgumentException($"Reference keys {keys.Shape} and values {values.Shape} must match as [1, S, heads, headDim].");
            if (_tokens >= 0 && keys.Shape[1] != _tokens)
                throw new ArgumentException($"Layer {layer} caches {keys.Shape[1]} tokens; earlier layers cached {_tokens}.");
            if (_keys[layer] is not null)
                throw new InvalidOperationException($"Layer {layer} is already cached.");
            _tokens = (int)keys.Shape[1];
            _keys[layer] = keys;
            _values[layer] = values;
        }

        /// <summary>Pins every entry on <paramref name="backend"/> and returns the cache, which now owns them.</summary>
        public ReferenceKvCache Build(IBackend backend, long fingerprint, string family)
        {
            ArgumentNullException.ThrowIfNull(backend);
            if (_built)
                throw new InvalidOperationException("The cache was already built.");
            if (_tokens < 0)
                throw new InvalidOperationException("A reference cache needs at least one layer.");
            int active = 0;
            long bytes = 0;
            for (int i = 0; i < _keys.Length; i++)
            {
                if (_keys[i] is not Tensor k)
                    continue;
                backend.PinActivation(k);
                backend.PinActivation(_values[i]!);
                bytes += 2L * k.ElementCount * k.DType.SizeInBytes;
                active++;
            }
            _built = true;
            Logs.Info($"{family}: cached reference keys/values for {active} of {_keys.Length} layers, {_tokens} tokens, "
                + $"{bytes / (1024 * 1024)} MiB.");
            return new ReferenceKvCache(backend, _keys, _values, _tokens, fingerprint);
        }

        /// <summary>Frees the entries if the cache was never built.</summary>
        public void Dispose()
        {
            if (_built)
                return;
            _built = true;
            for (int i = 0; i < _keys.Length; i++)
            {
                _keys[i]?.Dispose();
                _values[i]?.Dispose();
            }
        }
    }
}
