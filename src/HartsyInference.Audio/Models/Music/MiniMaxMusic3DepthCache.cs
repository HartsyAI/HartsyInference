using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Music;

/// <summary>One frame's key/value cache for <see cref="MiniMaxMusic3DepthDecoder"/>: without it every residual step re-runs the whole depth sequence, 44 token-passes per frame where 8 suffice.</summary>
/// <remarks>
/// <para>The classifier-free pair is folded into the head axis — buffers are <c>[1, batch·heads, maxSteps, headDim]</c>
/// — because <see cref="IBackend.KvCacheAppend"/> takes its head count from dimension 1 and would otherwise append
/// only batch row 0. <c>[B,H,S,D]</c> is contiguous, so the bytes are identical and attention treats a batch element
/// and a head the same way; it also makes a single step's projection output already head-major, so the one-step path
/// needs no permute at all.</para>
///
/// <para>Attention reads the WHOLE buffer — SDPA takes its key count from the tensor shape, so a partially filled
/// prefix cannot be handed over without copying it out — and <see cref="Mask"/> hides the unwritten tail. That makes
/// the buffers' zero-fill load-bearing: they must reach the device as zeros, which is why they are uploaded lazily by
/// the first append rather than through <see cref="IBackend.ResidentAllocateKv"/>, which skips the upload and would
/// leave the tail as whatever the allocator returned — and a NaN there survives an additive mask.</para>
/// </remarks>
public sealed unsafe class MiniMaxMusic3DepthCache : IDisposable
{
    private readonly Tensor[] _keys;
    private readonly Tensor[] _values;
    private readonly Tensor?[,] _masks;
    private readonly int _maxSteps;
    private int _length;
    private int _disposed;

    /// <summary>Allocates one <c>[1, heads, maxSteps, headDim]</c> key and value buffer per layer.</summary>
    /// <param name="heads">Attention heads times the classifier-free batch, which this cache folds into one axis.</param>
    public MiniMaxMusic3DepthCache(int layers, int heads, int headDim, int maxSteps)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layers);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(heads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSteps);
        _maxSteps = maxSteps;
        _keys = new Tensor[layers];
        _values = new Tensor[layers];
        _masks = new Tensor?[maxSteps, maxSteps];
        for (int layer = 0; layer < layers; layer++)
        {
            TensorShape shape = new TensorShape(1, heads, maxSteps, headDim);
            _keys[layer] = new Tensor(shape, DType.F32);
            _values[layer] = new Tensor(shape, DType.F32);
        }
    }

    /// <summary>Steps already written; also the absolute position of the next step.</summary>
    public int Length
    {
        get
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            return _length;
        }
    }

    /// <summary>Layer <paramref name="layer"/>'s full key buffer; entries past <see cref="Length"/> are masked out.</summary>
    public Tensor Keys(int layer) => _keys[layer];

    /// <summary>Layer <paramref name="layer"/>'s full value buffer, the twin of <see cref="Keys"/>.</summary>
    public Tensor Values(int layer) => _values[layer];

    /// <summary>Writes one layer's new keys and values at the current position, in place.</summary>
    public void Append(IBackend backend, int layer, Tensor key, Tensor value)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        int steps = (int)key.Shape[2];
        if (_length + steps > _maxSteps)
        {
            throw new InvalidOperationException(
                $"{nameof(MiniMaxMusic3DepthCache)} overflow: length {_length} plus {steps} step(s) exceeds {_maxSteps}.");
        }
        backend.KvCacheAppend(_keys[layer], key, _length);
        backend.KvCacheAppend(_values[layer], value, _length);
    }

    /// <summary>Moves past a step block, once every layer has appended it.</summary>
    public void Advance(int steps)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        _length += steps;
    }

    /// <summary>Additive mask for a <paramref name="steps"/>-token block starting at <see cref="Length"/>: causal within the block and <c>-inf</c> over the buffer's stale tail.</summary>
    /// <remarks>Kept per (position, block size) rather than rebuilt — the same handful of masks is uploaded for
    /// every frame of the generation, so a cached one stays resident on the device.</remarks>
    public Tensor Mask(int steps)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (steps <= 0 || _length + steps > _maxSteps)
        {
            throw new ArgumentOutOfRangeException(nameof(steps),
                $"A {steps}-step block at position {_length} does not fit {_maxSteps} steps.");
        }
        Tensor? cached = _masks[_length, steps - 1];
        if (cached is not null)
        {
            return cached;
        }
        Tensor mask = new Tensor(new TensorShape(steps, _maxSteps), DType.F32);
        float* data = (float*)mask.DataPointer;
        for (int query = 0; query < steps; query++)
        {
            for (int key = 0; key < _maxSteps; key++)
            {
                data[(query * _maxSteps) + key] = key <= _length + query ? 0f : float.NegativeInfinity;
            }
        }
        _masks[_length, steps - 1] = mask;
        return mask;
    }

    /// <summary>Drops the sequence for the next frame; the stale contents are finite and stay masked out.</summary>
    public void Reset()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        _length = 0;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        foreach (Tensor tensor in _keys)
        {
            tensor.Dispose();
        }
        foreach (Tensor tensor in _values)
        {
            tensor.Dispose();
        }
        foreach (Tensor? mask in _masks)
        {
            mask?.Dispose();
        }
    }
}
