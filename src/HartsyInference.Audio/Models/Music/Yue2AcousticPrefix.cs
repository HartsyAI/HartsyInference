using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Music;

/// <summary>One acoustic chunk's attention keys and values, held for the whole ODE solve.</summary>
/// <remarks>The AR prefix's K/V never change across a chunk's velocity evaluations — 64 of them for a 32-step
/// midpoint solve — but they were being re-concatenated and re-widened to full heads on every one, ~8,700 rows a
/// layer that are identical every time. Here the buffer is allocated once per chunk, the prefix rows are written
/// once, and each evaluation only appends its own <c>tokens</c> rows at the tail. Storing it at F16 where the
/// backend has F16 kernels also lets <see cref="IBackend.ScaledDotProductAttention"/> take cuDNN's native-F16
/// entry, which casts nothing — the F32 route re-narrows the whole key and value tensors on every call.</remarks>
internal sealed class Yue2AcousticPrefix : IDisposable
{
    private readonly Tensor[] _keys;
    private readonly Tensor[] _values;
    private int _disposed;

    /// <summary>Widens each layer's grouped prefix K/V to full heads and stores them at <paramref name="dtype"/>.
    /// The fused attention engine is MHA-only, so the widening has to happen somewhere; doing it here costs one
    /// pass per chunk instead of one per evaluation.</summary>
    public Yue2AcousticPrefix(IBackend backend, (Tensor Key, Tensor Value)[] arPrefix, int arLength, int tokens,
        int heads, int kvHeads, int headDim, DType dtype, Tensor cos, Tensor sin)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(arPrefix);
        Cos = cos;
        Sin = sin;
        ArLength = arLength;
        KvLength = arLength + tokens;
        Dtype = dtype;
        _keys = new Tensor[arPrefix.Length];
        _values = new Tensor[arPrefix.Length];
        int groupSize = heads / kvHeads;
        try
        {
            for (int i = 0; i < arPrefix.Length; i++)
            {
                _keys[i] = new Tensor(new TensorShape(1, heads, KvLength, headDim), dtype);
                _values[i] = new Tensor(new TensorShape(1, heads, KvLength, headDim), dtype);
                // KvCacheAppend narrows F32 to an F16 destination on write, so the widened staging stays F32.
                using Tensor wideKey = new(new TensorShape(1, heads, arLength, headDim), DType.F32);
                using Tensor wideValue = new(new TensorShape(1, heads, arLength, headDim), DType.F32);
                backend.RepeatKvHeads(wideKey, arPrefix[i].Key, kvHeads, groupSize);
                backend.RepeatKvHeads(wideValue, arPrefix[i].Value, kvHeads, groupSize);
                backend.KvCacheAppend(_keys[i], wideKey, 0);
                backend.KvCacheAppend(_values[i], wideValue, 0);
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>Prefix rows, which is where each evaluation's own K/V start.</summary>
    public int ArLength { get; }

    /// <summary>Total attended length, prefix plus this chunk's tokens.</summary>
    public int KvLength { get; }

    /// <summary>Storage dtype of the key/value buffers, which the caller must match for Q and the output.</summary>
    public DType Dtype { get; }

    /// <summary>RoPE tables for this chunk's token span. They depend only on the span, so they are the same for
    /// every evaluation — and building them is a host loop over <c>tokens * headDim/2</c> Pow/Cos/Sin triples,
    /// which is not something to repeat 64 times per chunk.</summary>
    public Tensor Cos { get; }

    public Tensor Sin { get; }

    public Tensor Keys(int layer) => _keys[layer];

    public Tensor Values(int layer) => _values[layer];

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        for (int i = 0; i < _keys.Length; i++) { _keys[i]?.Dispose(); _values[i]?.Dispose(); }
        Cos?.Dispose();
        Sin?.Dispose();
    }
}
