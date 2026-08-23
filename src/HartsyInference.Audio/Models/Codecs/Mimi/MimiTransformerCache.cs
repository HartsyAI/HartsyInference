using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Mimi;

/// <summary>Growing K/V history for <see cref="MimiTransformer.ForwardStreaming"/>, one per decoder-transformer instance per streamed utterance.</summary>
/// <remarks>Unlike <c>FixedKvCache</c> (the backbone's O(1)-append, fixed-capacity
/// design, sized for LLM-scale sequences), this is a plain reallocate-and-copy growth — appropriate here because
/// a Mimi decode utterance is capped at a few hundred frames total, so the O(n) copy per call costs nothing
/// measurable, and it avoids the fixed-capacity/no-batch-support constraints an LLM-oriented cache carries for no
/// benefit at this scale.</remarks>
internal sealed unsafe class MimiTransformerCache : IDisposable
{
    private Tensor?[] _k = [];
    private Tensor?[] _v = [];
    private int _disposed;

    /// <summary>Frames committed so far (across all layers uniformly — every layer advances together).</summary>
    public int Length { get; set; }

    /// <summary>Allocates the per-layer slot arrays on first use; a no-op once sized.</summary>
    public void EnsureLayers(int layers)
    {
        if (_k.Length == layers) return;
        if (_k.Length != 0) throw new InvalidOperationException($"MimiTransformerCache already sized for {_k.Length} layers, got {layers}.");
        _k = new Tensor?[layers];
        _v = new Tensor?[layers];
    }

    /// <summary>Appends this layer's new-frame K/V (<c>[1,heads,t,headDim]</c>) to whatever history is stored, returning the concatenated <c>[1,heads,priorLen+t,headDim]</c> tensors — owned by the cache; the caller must NOT dispose them.</summary>
    /// <remarks>Every layer's history advances together, driven by <see cref="MimiTransformer.ForwardStreaming"/>'s single <see cref="Length"/> update after all layers run.</remarks>
    public (Tensor K, Tensor V) AppendAndGet(int layer, Tensor kNew, Tensor vNew, int t)
    {
        ThrowIfDisposed();
        Tensor kAll = Concat(_k[layer], kNew, t);
        Tensor vAll = Concat(_v[layer], vNew, t);
        _k[layer]?.Dispose();
        _v[layer]?.Dispose();
        _k[layer] = kAll;
        _v[layer] = vAll;
        return (kAll, vAll);
    }

    private static Tensor Concat(Tensor? prior, Tensor fresh, int tNew)
    {
        int heads = (int)fresh.Shape[1];
        int headDim = (int)fresh.Shape[3];
        int priorLen = prior is null ? 0 : (int)prior.Shape[2];
        int total = priorLen + tNew;
        Tensor outT = new(new TensorShape(1, heads, total, headDim), DType.F32);
        float* dst = (float*)outT.DataPointer;
        for (int h = 0; h < heads; h++)
        {
            long dstHeadBase = (long)h * total * headDim;
            if (prior is not null)
            {
                float* src = (float*)prior.DataPointer;
                long srcHeadBase = (long)h * priorLen * headDim;
                Buffer.MemoryCopy(src + srcHeadBase, dst + dstHeadBase, (long)priorLen * headDim * 4, (long)priorLen * headDim * 4);
            }
            float* freshSrc = (float*)fresh.DataPointer;
            long freshHeadBase = (long)h * tNew * headDim;
            Buffer.MemoryCopy(freshSrc + freshHeadBase, dst + dstHeadBase + (long)priorLen * headDim, (long)tNew * headDim * 4, (long)tNew * headDim * 4);
        }
        return outT;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(MimiTransformerCache));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor? t in _k) t?.Dispose();
        foreach (Tensor? t in _v) t?.Dispose();
    }
}
