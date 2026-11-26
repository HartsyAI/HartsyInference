using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Layers.Attention;

/// <summary>Keys and values a model keeps between attention calls, per layer and in segments along the token axis: a latent frame, a chunk, a group
/// of rows. The store owns what it is given, pins it so the activation sweep between steps leaves it on the device, and unpins and frees it when
/// it is replaced, removed or the store is disposed. A layer can also be moved to host memory (<see cref="MoveLayerToHost"/>): its segments are then
/// host tensors, never pinned, and the caller stages them to the device for each read. Which segments an attention reads, and when one is dropped,
/// stay with the model.</summary>
/// <remarks>Segments may differ in length; every other extent, and the dtype, must agree. Nothing is copied on the way in. A read that needs one
/// tensor is a <see cref="Join"/>, or a <see cref="Compact"/> once for a history many blocks read. Used from one thread at a time.</remarks>
public sealed class KvSegmentStore : IDisposable
{
    private readonly IBackend _backend;
    private readonly List<Segment?>[] _layers;
    private readonly int _tokenAxis;
    private TensorShape _shape;
    private DType _dtype;
    private bool _shaped;
    private int _disposed;

    private sealed record Segment(Tensor Keys, Tensor Values, bool Host = false);

    /// <summary>A store of <paramref name="layers"/> layers whose tensors carry tokens on axis <paramref name="tokenAxis"/>.</summary>
    public KvSegmentStore(IBackend backend, int layers, int tokenAxis)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentOutOfRangeException.ThrowIfLessThan(layers, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(tokenAxis);
        _backend = backend;
        _tokenAxis = tokenAxis;
        _layers = new List<Segment?>[layers];
        for (int i = 0; i < layers; i++)
            _layers[i] = [];
    }

    /// <summary>Layers the store spans.</summary>
    public int Layers => _layers.Length;

    /// <summary>Bytes a store takes: keys and values of <paramref name="tokens"/> × <paramref name="width"/> in every layer.</summary>
    public static long EstimateBytes(int layers, long tokens, long width, DType dtype) => checked(2L * layers * dtype.ComputeByteCount(tokens * width));

    /// <summary>Bytes the stored keys and values take now.</summary>
    public long Bytes
    {
        get
        {
            long total = 0;
            foreach (List<Segment?> layer in _layers)
                foreach (Segment? segment in layer)
                    if (segment is not null)
                        total += 2 * segment.Keys.DType.ComputeByteCount(segment.Keys.ElementCount);
            return total;
        }
    }

    /// <summary>Segment slots of <paramref name="layer"/>, empty ones included.</summary>
    public int Count(int layer) => Layer(layer).Count;

    /// <summary>Tokens stored in <paramref name="layer"/>.</summary>
    public long Tokens(int layer)
    {
        long tokens = 0;
        foreach (Segment? segment in Layer(layer))
            tokens += segment?.Keys.Shape[_tokenAxis] ?? 0;
        return tokens;
    }

    /// <summary>True when <paramref name="slot"/> of <paramref name="layer"/> holds keys and values.</summary>
    public bool Has(int layer, int slot) =>
        (uint)layer < (uint)_layers.Length && (uint)slot < (uint)_layers[layer].Count && _layers[layer][slot] is not null;

    /// <summary>The keys of a slot, borrowed until the slot changes.</summary>
    /// <exception cref="InvalidOperationException">The slot is empty.</exception>
    public Tensor Keys(int layer, int slot) => At(layer, slot).Keys;

    /// <summary>The values of a slot, borrowed until the slot changes.</summary>
    /// <exception cref="InvalidOperationException">The slot is empty.</exception>
    public Tensor Values(int layer, int slot) => At(layer, slot).Values;

    /// <summary>Writes a slot, taking ownership of both tensors (also when it throws) and freeing what it replaces. Slots before it that were never
    /// written stay empty. A tensor belongs to one slot; writing a slot again with a tensor it already holds keeps that tensor, also when the write
    /// is refused.</summary>
    /// <exception cref="ArgumentException">The two tensors differ from each other, or from the stored segments outside the token axis.</exception>
    public void Set(int layer, int slot, Tensor keys, Tensor values)
    {
        ArgumentNullException.ThrowIfNull(keys);
        ArgumentNullException.ThrowIfNull(values);
        Segment? held = Has(layer, slot) ? _layers[layer][slot] : null;
        bool stored = false;
        try
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            List<Segment?> segments = Layer(layer);
            ArgumentOutOfRangeException.ThrowIfNegative(slot);
            Validate(keys, values);
            Pin(keys, values, held);
            stored = true;
            while (segments.Count <= slot)
                segments.Add(null);
            segments[slot] = new Segment(keys, values);
            if (held is not null)
            {
                if (!ReferenceEquals(held.Keys, keys) && !ReferenceEquals(held.Keys, values)) Free(held.Keys, held.Host);
                if (!ReferenceEquals(held.Values, keys) && !ReferenceEquals(held.Values, values)) Free(held.Values, held.Host);
            }
            _shape = keys.Shape;
            _dtype = keys.DType;
            _shaped = true;
        }
        finally
        {
            if (!stored)
            {
                if (!Holds(held, keys)) keys.Dispose();
                if (!Holds(held, values)) values.Dispose();
            }
        }
    }

    /// <summary>Adds a segment after the last one; ownership as for <see cref="Set"/>.</summary>
    public void Append(int layer, Tensor keys, Tensor values) =>
        Set(layer, (uint)layer < (uint)_layers.Length ? _layers[layer].Count : 0, keys, values);

    /// <summary>Drops <paramref name="count"/> segments from <paramref name="first"/> on; the later ones move down.</summary>
    public void RemoveRange(int layer, int first, int count)
    {
        List<Segment?> segments = Layer(layer);
        ArgumentOutOfRangeException.ThrowIfNegative(first);
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        if (first + count > segments.Count)
            throw new ArgumentOutOfRangeException(nameof(count), count, $"Layer {layer} holds {segments.Count} segments; [{first}, +{count}) leaves them.");
        for (int i = first; i < first + count; i++)
            Release(segments[i]);
        segments.RemoveRange(first, count);
        // An empty store takes any shape again, so one store can serve runs of different sizes.
        if (_layers.All(static layerSegments => layerSegments.TrueForAll(static s => s is null)))
            _shaped = false;
    }

    /// <summary>Moves a layer's segments to host memory: each device segment's keys and values are copied out, the device tensors are unpinned and freed,
    /// and the segment holds host tensors from then on. A host segment is read again through <see cref="Keys"/> and <see cref="Values"/> like any
    /// other, and the caller stages it to the device for the reads (the weight staging of the streaming code does this for the KV cache).</summary>
    public void MoveLayerToHost(int layer)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        List<Segment?> segments = Layer(layer);
        for (int slot = 0; slot < segments.Count; slot++)
        {
            Segment? segment = segments[slot];
            if (segment is null || segment.Host)
                continue;
            Tensor hostKeys = CopyToHost(segment.Keys), hostValues = CopyToHost(segment.Values);
            segments[slot] = new Segment(hostKeys, hostValues, Host: true);
            Free(segment.Keys, host: false);
            Free(segment.Values, host: false);
        }
    }

    private static Tensor CopyToHost(Tensor device)
    {
        Tensor host = new(device.Shape, device.DType);
        device.AsReadOnlySpan<float>().CopyTo(host.AsSpan<float>());
        return host;
    }

    /// <summary>Drops every segment of <paramref name="layer"/>.</summary>
    public void Clear(int layer) => RemoveRange(layer, 0, Count(layer));

    /// <summary>Segments <c>[first, first + count)</c> joined along the token axis into fresh tensors the caller owns.</summary>
    /// <exception cref="InvalidOperationException">A slot in the range is empty.</exception>
    public (Tensor Keys, Tensor Values) Join(int layer, int first, int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(first);
        ArgumentOutOfRangeException.ThrowIfLessThan(count, 1);
        Tensor[] keys = new Tensor[count], values = new Tensor[count];
        for (int i = 0; i < count; i++)
        {
            Segment at = At(layer, first + i);
            keys[i] = at.Keys;
            values[i] = at.Values;
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

    /// <summary>Replaces the segments of <paramref name="layer"/> by one that holds all their tokens in order, so every later read is a borrowed one.
    /// The old and the new copy exist together while it runs.</summary>
    public void Compact(int layer)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        int count = Count(layer);
        if (count <= 1)
            return;
        (Tensor keys, Tensor values) = Join(layer, 0, count);
        Set(layer, 0, keys, values);
        RemoveRange(layer, 1, count - 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        for (int layer = 0; layer < _layers.Length; layer++)
            Clear(layer);
    }

    private Segment At(int layer, int slot)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        List<Segment?> segments = Layer(layer);
        return (slot >= 0 && slot < segments.Count ? segments[slot] : null)
            ?? throw new InvalidOperationException($"Layer {layer} has no keys and values stored in slot {slot}.");
    }

    private List<Segment?> Layer(int layer)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(layer);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(layer, _layers.Length);
        return _layers[layer];
    }

    private void Validate(Tensor keys, Tensor values)
    {
        if (ReferenceEquals(keys, values))
            throw new ArgumentException("Keys and values are the same tensor; each is owned and freed on its own.", nameof(values));
        if (keys.Shape != values.Shape || keys.DType != values.DType)
            throw new ArgumentException($"Keys {keys.Shape} {keys.DType} and values {values.Shape} {values.DType} must match.", nameof(values));
        if (_tokenAxis >= keys.Shape.Rank)
            throw new ArgumentException($"The token axis is {_tokenAxis}; got {keys.Shape}.", nameof(keys));
        if (!_shaped)
            return;
        bool same = keys.DType == _dtype && keys.Shape.Rank == _shape.Rank;
        for (int axis = 0; same && axis < _shape.Rank; axis++)
            same = axis == _tokenAxis || keys.Shape[axis] == _shape[axis];
        if (!same)
            throw new ArgumentException($"The segments are {_dtype} {_shape} apart from axis {_tokenAxis}; this one is {keys.DType} {keys.Shape}.", nameof(keys));
    }

    private static bool Holds(Segment? segment, Tensor tensor) =>
        segment is not null && (ReferenceEquals(segment.Keys, tensor) || ReferenceEquals(segment.Values, tensor));

    // Pins what the slot does not hold already; a failure leaves the pins as they were.
    private void Pin(Tensor keys, Tensor values, Segment? held)
    {
        bool pinKeys = !Holds(held, keys);
        if (pinKeys)
            _backend.PinActivation(keys);
        try
        {
            if (!Holds(held, values))
                _backend.PinActivation(values);
        }
        catch
        {
            if (pinKeys)
                _backend.UnpinActivation(keys);
            throw;
        }
    }

    private void Release(Segment? segment)
    {
        if (segment is null)
            return;
        Free(segment.Keys, segment.Host);
        Free(segment.Values, segment.Host);
    }

    // A host segment's tensors were never pinned: they are not on the device, so there is nothing to unpin.
    private void Free(Tensor stored, bool host)
    {
        if (!host)
            _backend.UnpinActivation(stored);
        stored.Dispose();
    }

    private Tensor Join(Tensor[] parts)
    {
        Span<long> dims = stackalloc long[parts[0].Shape.Rank];
        parts[0].Shape.CopyDimsTo(dims);
        for (int i = 1; i < parts.Length; i++)
            dims[_tokenAxis] += parts[i].Shape[_tokenAxis];
        Tensor joined = new(new TensorShape(dims), parts[0].DType);
        try
        {
            _backend.Concat(joined, parts, _tokenAxis);
        }
        catch
        {
            joined.Dispose();
            throw;
        }
        return joined;
    }
}
