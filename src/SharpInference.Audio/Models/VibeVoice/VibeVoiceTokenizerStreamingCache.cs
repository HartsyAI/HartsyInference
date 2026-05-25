using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.VibeVoice;

/// <summary>Per-layer, per-sample history buffer for streaming causal Conv1D / ConvTranspose1D
/// in the VibeVoice acoustic and semantic VAEs. Mirrors the Python
/// <c>VibeVoiceTokenizerStreamingCache</c> in
/// <c>vibevoice/modular/modular_vibevoice_tokenizer.py</c>.
///
/// <para>Each <see cref="SConv1d"/>-style wrapper stores its trailing
/// <c>(kernel-1) * dilation - (stride-1)</c> input samples (or, for the transpose path,
/// the trailing <c>kernel-1</c> input samples) keyed by <c>(layer_id, sample_index)</c>.
/// Subsequent chunks prepend this history to their inputs so the underlying conv sees the
/// same receptive field it would in a non-streaming forward.</para>
///
/// <para><strong>Layer IDs</strong> are deterministic strings assigned at construction time
/// by the parent module (e.g. <c>"enc.stage0.block1.mixer"</c>). The Python reference uses
/// <c>f"sconv1d_{id(self)}"</c> which is process-local and non-portable; we replace it with
/// a stable composition path so encode/decode passes can target the same cache keys across
/// invocations.</para>
///
/// <para><strong>Per-sample keys</strong> support batching independent streams (e.g. up to
/// 4 speakers in a podcast each producing audio in their own buffer). Internally we store
/// tensors per <c>(layer_id, sample_idx)</c> and reassemble batched outputs in
/// <see cref="Get"/> with left-padding to the longest entry.</para></summary>
public sealed class VibeVoiceTokenizerStreamingCache : IDisposable
{
    private readonly Dictionary<(string LayerId, int SampleIndex), Tensor> _cache = new();
    private bool _disposed;

    /// <summary>Looks up the cached history tensors for the given layer + sample indices and
    /// returns them stacked along a fresh batch axis. Tensors are left-padded with zeros to
    /// the longest entry so the result is rectangular.
    ///
    /// <para>Returns <c>null</c> if any of the requested sample indices is missing — the
    /// caller treats that as "first chunk, initialize to zeros" (the canonical first-chunk
    /// behavior in the Python reference).</para>
    ///
    /// <para>Output shape: <c>[len(sampleIndices), C, max_T]</c>. The caller passes
    /// <paramref name="channels"/> so we can verify and allocate the stacked buffer in F32
    /// without inspecting every entry's shape.</para></summary>
    public Tensor? Get(string layerId, ReadOnlySpan<int> sampleIndices, int channels)
    {
        ThrowIfDisposed();
        int batch = sampleIndices.Length;
        if (batch == 0)
            throw new ArgumentException("sampleIndices must contain at least one entry.", nameof(sampleIndices));

        int maxLength = 0;
        for (int i = 0; i < batch; i++)
        {
            if (!_cache.TryGetValue((layerId, sampleIndices[i]), out Tensor? state))
                return null;
            int len = (int)state.Shape[state.Shape.Rank - 1];
            if (len > maxLength) maxLength = len;
        }

        // Empty cache (e.g. when kernel==stride and there is no context to carry forward).
        if (maxLength == 0)
        {
            return new Tensor(new TensorShape(batch, channels, 0), DType.F32);
        }

        Tensor result = new(new TensorShape(batch, channels, maxLength), DType.F32);
        unsafe
        {
            float* dst = (float*)result.DataPointer;
            int stride = channels * maxLength;
            for (int i = 0; i < batch; i++)
            {
                Tensor src = _cache[(layerId, sampleIndices[i])];
                int srcLen = (int)src.Shape[src.Shape.Rank - 1];
                int leftPad = maxLength - srcLen;
                float* srcPtr = (float*)src.DataPointer;
                for (int c = 0; c < channels; c++)
                {
                    float* dstRow = dst + i * stride + c * maxLength;
                    float* srcRow = srcPtr + c * srcLen;
                    for (int k = 0; k < leftPad; k++) dstRow[k] = 0f;
                    for (int k = 0; k < srcLen; k++) dstRow[leftPad + k] = srcRow[k];
                }
            }
        }
        return result;
    }

    /// <summary>Stores the trailing-history slice produced by a conv layer for each sample
    /// in the batch. <paramref name="batched"/> has shape <c>[batch, C, T]</c>; we split
    /// along the batch axis and store one detached owning copy per <c>(layerId, sampleIndices[i])</c>.
    /// Any pre-existing entries under those keys are disposed first.</summary>
    public void Set(string layerId, ReadOnlySpan<int> sampleIndices, Tensor batched)
    {
        ThrowIfDisposed();
        if (batched.Shape.Rank != 3)
            throw new ArgumentException($"Expected rank-3 tensor [batch, channels, time], got {batched.Shape}.", nameof(batched));
        int batch = (int)batched.Shape[0];
        if (batch != sampleIndices.Length)
            throw new ArgumentException($"Batch dim ({batch}) does not match sampleIndices length ({sampleIndices.Length}).", nameof(sampleIndices));

        int channels = (int)batched.Shape[1];
        int t = (int)batched.Shape[2];
        int perSampleStride = channels * t;

        unsafe
        {
            float* src = (float*)batched.DataPointer;
            for (int i = 0; i < batch; i++)
            {
                (string, int) key = (layerId, sampleIndices[i]);
                if (_cache.TryGetValue(key, out Tensor? existing))
                {
                    existing.Dispose();
                    _cache.Remove(key);
                }
                Tensor slice = new(new TensorShape(channels, t), DType.F32);
                float* dst = (float*)slice.DataPointer;
                Buffer.MemoryCopy(src + i * perSampleStride, dst, perSampleStride * sizeof(float), perSampleStride * sizeof(float));
                _cache[key] = slice;
            }
        }
    }

    /// <summary>Zeros (in place) all cached entries for the given sample indices across all
    /// layers. Invoked by the inference loop on the <c>speech_end</c> token to reset the
    /// per-speaker streaming state without throwing away the layer-ID bookkeeping.</summary>
    public void SetToZero(ReadOnlySpan<int> sampleIndices)
    {
        ThrowIfDisposed();
        HashSet<int> targets = new();
        for (int i = 0; i < sampleIndices.Length; i++) targets.Add(sampleIndices[i]);

        foreach (KeyValuePair<(string LayerId, int SampleIndex), Tensor> kv in _cache)
        {
            if (!targets.Contains(kv.Key.SampleIndex)) continue;
            unsafe
            {
                float* p = (float*)kv.Value.DataPointer;
                long n = kv.Value.ElementCount;
                for (long k = 0; k < n; k++) p[k] = 0f;
            }
        }
    }

    /// <summary>Clears cache entries. With no args, drops everything. With
    /// <paramref name="layerId"/>, drops only that layer (any sample). With both
    /// <paramref name="layerId"/> and <paramref name="sampleIndices"/>, drops just those
    /// specific keys.</summary>
    public void Clear(string? layerId = null, ReadOnlySpan<int> sampleIndices = default)
    {
        ThrowIfDisposed();
        if (layerId is null && sampleIndices.IsEmpty)
        {
            foreach (Tensor t in _cache.Values) t.Dispose();
            _cache.Clear();
            return;
        }

        List<(string, int)> toRemove = new();
        if (layerId is not null && sampleIndices.IsEmpty)
        {
            foreach (KeyValuePair<(string LayerId, int SampleIndex), Tensor> kv in _cache)
                if (kv.Key.LayerId == layerId) toRemove.Add(kv.Key);
        }
        else if (layerId is not null)
        {
            for (int i = 0; i < sampleIndices.Length; i++)
                toRemove.Add((layerId, sampleIndices[i]));
        }

        foreach ((string, int) key in toRemove)
        {
            if (_cache.TryGetValue(key, out Tensor? t))
            {
                t.Dispose();
                _cache.Remove(key);
            }
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (Tensor t in _cache.Values) t.Dispose();
        _cache.Clear();
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(VibeVoiceTokenizerStreamingCache));
    }
}
