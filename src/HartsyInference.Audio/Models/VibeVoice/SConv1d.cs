using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.VibeVoice;

/// <summary>Causal Conv1D with built-in streaming-cache support: successive calls on contiguous chunks produce the same output as a single call on the concatenated input.</summary>
/// <remarks>Each instance carries a deterministic <c>LayerId</c> string assigned at
/// construction (e.g. <c>"enc.stage3.block2.mixer"</c>) that keys into the per-layer
/// <see cref="VibeVoiceTokenizerStreamingCache"/>. The cache buffer per (layer, sample)
/// holds the last <c>context_size = (kernel-1)*dilation - (stride-1)</c> INPUT samples;
/// when a new chunk arrives we prepend the cached history, run the conv with no extra
/// left padding (the history IS the padding), then stash the trailing
/// <c>context_size</c> samples back into the cache.
///
/// <para>Non-streaming mode routes through <see cref="IBackend.Conv1d"/> with the standard
/// left-pad budget, matching the published Python <c>SConv1d._forward_non_streaming</c>.</para></remarks>
internal sealed unsafe class SConv1d
{
    public string LayerId { get; }
    public int InChannels { get; }
    public int OutChannels { get; }
    public int KernelSize { get; }
    public int Stride { get; }
    public int Dilation { get; }
    public int Groups { get; }
    public bool Bias { get; }

    /// <summary><c>(kernel-1)*dilation - (stride-1)</c> — the per-call left-pad budget, also the streaming context size carried between chunks.</summary>
    public int ContextSize => (KernelSize - 1) * Dilation - (Stride - 1);

    private Tensor? _weight;
    private Tensor? _bias;

    public SConv1d(string layerId, int inChannels, int outChannels, int kernelSize, int stride = 1, int dilation = 1, int groups = 1, bool bias = true)
    {
        if (string.IsNullOrEmpty(layerId)) throw new ArgumentException("layerId must be non-empty.", nameof(layerId));
        LayerId = layerId;
        InChannels = inChannels;
        OutChannels = outChannels;
        KernelSize = kernelSize;
        Stride = stride;
        Dilation = dilation;
        Groups = groups;
        Bias = bias;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        // Python layout: SConv1d wraps NormConv1d which wraps nn.Conv1d. The state dict key is
        // f"{prefix}.conv.conv.weight" (the outer is SConv1d's `.conv` = NormConv1d, the inner
        // is NormConv1d's `.conv` = nn.Conv1d).
        _weight = WhisperOps.EnsureF32(w[$"{prefix}.conv.conv.weight"]);
        if (Bias) _bias = WhisperOps.EnsureF32(w[$"{prefix}.conv.conv.bias"]);
    }

    /// <summary>Non-streaming forward, routed to <see cref="IBackend.Conv1d"/>'s asymmetric <c>padLeft/padRight</c> for the causal left-pad and stride-alignment right-pad.</summary>
    /// <returns><c>[B, C_out, T_out]</c>; T_out depends on stride and the right-pad budget.</returns>
    public Tensor Forward(IBackend backend, Tensor input, int batch, int tIn)
    {
        if (_weight is null) throw new InvalidOperationException($"SConv1d '{LayerId}' weights not loaded.");
        int padTotal = ContextSize;
        int extraRight = VibeVoiceOps.GetExtraRightPadding(tIn, KernelSize, Stride, padTotal);
        int tPadded = tIn + padTotal + extraRight;
        int tOut = (tPadded - Dilation * (KernelSize - 1) - 1) / Stride + 1;

        Tensor output = new(new TensorShape(batch, OutChannels, tOut), DType.F32);
        backend.Conv1d(output, input, _weight!, _bias, Stride, padTotal, extraRight, Dilation, Groups);
        return output;
    }

    /// <summary>Streaming forward: prepends per-sample cached history, runs the conv with zero additional left pad, then stashes the new trailing context; output corresponds to the new input chunk only.</summary>
    /// <remarks>Uses NO right-pad (<c>extraRight = 0</c>) for chunk-stride alignment — the streaming forward
    /// intentionally drops a few trailing samples per chunk and lets the next chunk's cached history (plus its
    /// own new samples) recover them. This matches the published Python behavior where
    /// <c>self.conv(input_with_context)</c> is called directly without an explicit pad step.</remarks>
    public Tensor ForwardStreaming(IBackend backend, Tensor input, int batch, int tIn, VibeVoiceTokenizerStreamingCache cache, ReadOnlySpan<int> sampleIndices)
    {
        if (_weight is null) throw new InvalidOperationException($"SConv1d '{LayerId}' weights not loaded.");

        int ctx = ContextSize;
        Tensor? cached = cache.Get(LayerId, sampleIndices, InChannels);
        int cachedLen = cached is null ? ctx : (int)cached.Shape[cached.Shape.Rank - 1];

        // Build [B, C_in, cachedLen + tIn] on-device: the cached history (or a zero pad on the
        // first chunk) provides the left receptive field, so the conv itself runs with no
        // padding budget. GPU Concat keeps the large input tensor resident.
        int tCombined = cachedLen + tIn;
        Tensor combined = new(new TensorShape(batch, InChannels, tCombined), DType.F32);
        if (cached is not null)
        {
            backend.Concat(combined, [cached, input], dim: 2);
            cached.Dispose();
        }
        else
        {
            // First chunk — zero-pad the cache region (Python "cached_states = zeros(B, C, ctx)").
            Tensor zeros = new(new TensorShape(batch, InChannels, ctx), DType.F32);
            backend.Fill(zeros, 0f);
            backend.Concat(combined, [zeros, input], dim: 2);
            zeros.Dispose();
        }

        // Conv with NO further padding (padLeft=0, padRight=0) — the prefix IS the left pad.
        int tOut = (tCombined - Dilation * (KernelSize - 1) - 1) / Stride + 1;
        if (tOut <= 0)
        {
            combined.Dispose();
            cache.Set(LayerId, sampleIndices, new Tensor(new TensorShape(batch, InChannels, 0), DType.F32));
            return new Tensor(new TensorShape(batch, OutChannels, 0), DType.F32);
        }

        Tensor output = new(new TensorShape(batch, OutChannels, tOut), DType.F32);
        backend.Conv1d(output, combined, _weight!, _bias, Stride, 0, 0, Dilation, Groups);

        // Stash the trailing ContextSize input samples into the cache for the next chunk.
        int tKeep = Math.Min(ctx, tCombined);
        Tensor newTail = new(new TensorShape(batch, InChannels, tKeep), DType.F32);
        backend.SliceLastDim(newTail, combined, tCombined - tKeep);
        cache.Set(LayerId, sampleIndices, newTail);
        newTail.Dispose();
        combined.Dispose();
        return output;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_weight is not null) yield return _weight;
        if (_bias is not null) yield return _bias;
    }
}
