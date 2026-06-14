using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.VibeVoice;

/// <summary>Causal Conv1D with built-in streaming-cache support. Wraps
/// <see cref="VibeVoiceOps.CausalConv1d"/> so that successive calls on contiguous chunks
/// produce the same output as a single call on the concatenated input.
///
/// <para>Each instance carries a deterministic <c>LayerId</c> string assigned at
/// construction (e.g. <c>"enc.stage3.block2.mixer"</c>) that keys into the per-layer
/// <see cref="VibeVoiceTokenizerStreamingCache"/>. The cache buffer per (layer, sample)
/// holds the last <c>context_size = (kernel-1)*dilation - (stride-1)</c> INPUT samples;
/// when a new chunk arrives we prepend the cached history, run the conv with no extra
/// left padding (the history IS the padding), then stash the trailing
/// <c>context_size</c> samples back into the cache.</para>
///
/// <para>Non-streaming mode (<paramref name="useCache"/> = false): forwards to
/// <see cref="VibeVoiceOps.CausalConv1d"/> with the standard left-pad budget, matching
/// the published Python <c>SConv1d._forward_non_streaming</c>.</para></summary>
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

    /// <summary><c>(kernel-1)*dilation - (stride-1)</c> — the per-call left-pad budget,
    /// also the streaming context size carried between chunks.</summary>
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

    /// <summary>Non-streaming forward. Input <c>[B, C_in, T_in]</c>, output
    /// <c>[B, C_out, T_out]</c>. T_out depends on stride and the right-pad budget.</summary>
    public Tensor Forward(Tensor input, int batch, int tIn)
    {
        if (_weight is null) throw new InvalidOperationException($"SConv1d '{LayerId}' weights not loaded.");
        int padTotal = ContextSize;
        int extraRight = VibeVoiceOps.GetExtraRightPadding(tIn, KernelSize, Stride, padTotal);
        int tPadded = tIn + padTotal + extraRight;
        int tOut = (tPadded - Dilation * (KernelSize - 1) - 1) / Stride + 1;

        Tensor output = new(new TensorShape(batch, OutChannels, tOut), DType.F32);
        VibeVoiceOps.CausalConv1d(output, input, _weight!, _bias,
            batch, InChannels, OutChannels, tIn, KernelSize, Stride, Dilation, Groups, extraRight);
        return output;
    }

    /// <summary>Streaming forward. Prepends per-sample cached history, runs the conv with
    /// zero additional left pad, stashes the new trailing context. Output corresponds to
    /// the new input chunk only.
    ///
    /// <para>For chunk-stride alignment we use NO right-pad (<c>extraRight = 0</c>) — the
    /// streaming forward intentionally drops a few trailing samples per chunk and lets the
    /// next chunk's cached history (plus its own new samples) recover them. This matches
    /// the published Python behavior where <c>self.conv(input_with_context)</c> is called
    /// directly without an explicit pad step.</para></summary>
    public Tensor ForwardStreaming(Tensor input, int batch, int tIn, VibeVoiceTokenizerStreamingCache cache, ReadOnlySpan<int> sampleIndices)
    {
        if (_weight is null) throw new InvalidOperationException($"SConv1d '{LayerId}' weights not loaded.");

        int ctx = ContextSize;
        Tensor? cached = cache.Get(LayerId, sampleIndices, InChannels);
        int cachedLen = cached is null ? 0 : (int)cached.Shape[cached.Shape.Rank - 1];

        // Build the [B, C_in, cachedLen + tIn] working tensor by copying cached + input.
        int tCombined = cachedLen + tIn;
        Tensor combined = new(new TensorShape(batch, InChannels, tCombined), DType.F32);
        float* dst = (float*)combined.DataPointer;
        float* xPtr = (float*)input.DataPointer;
        if (cached is not null)
        {
            float* cPtr = (float*)cached.DataPointer;
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < InChannels; c++)
                {
                    int dstBase = (b * InChannels + c) * tCombined;
                    int srcBase = (b * InChannels + c) * cachedLen;
                    for (int t = 0; t < cachedLen; t++) dst[dstBase + t] = cPtr[srcBase + t];
                }
        }
        else
        {
            // First chunk — initialize the cache region with zeros (matches Python
            // "cached_states = torch.zeros(B, C, ctx)" fallback when no cache exists yet).
            int zeroLen = ctx;
            tCombined = zeroLen + tIn;
            combined.Dispose();
            combined = new(new TensorShape(batch, InChannels, tCombined), DType.F32);
            dst = (float*)combined.DataPointer;
            cachedLen = zeroLen;
            // Zero-init for [0, ctx).
            for (long i = 0; i < (long)batch * InChannels * zeroLen; i++) dst[i] = 0f;
        }
        cached?.Dispose();

        // Copy new input samples after the cached region.
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < InChannels; c++)
            {
                int dstBase = (b * InChannels + c) * tCombined + cachedLen;
                int srcBase = (b * InChannels + c) * tIn;
                for (int t = 0; t < tIn; t++) dst[dstBase + t] = xPtr[srcBase + t];
            }

        // Run the inner conv against the combined buffer with NO further padding
        // (padTotal=0, extraRight=0). The cached prefix already provides the left
        // receptive field.
        int tPadded = tCombined; // no extra pad
        int tOut = (tPadded - Dilation * (KernelSize - 1) - 1) / Stride + 1;
        if (tOut <= 0)
        {
            combined.Dispose();
            // Still store what we have for next chunk.
            cache.Set(LayerId, sampleIndices, BuildEmptyTail(batch));
            return new Tensor(new TensorShape(batch, OutChannels, 0), DType.F32);
        }

        Tensor output = new(new TensorShape(batch, OutChannels, tOut), DType.F32);
        RunConvNoPad(output, combined, _weight!, _bias, batch, InChannels, OutChannels, tCombined, KernelSize, Stride, Dilation, Groups);

        // Stash the trailing ContextSize input samples into the cache for the next chunk.
        Tensor newTail = SliceTail(combined, batch, InChannels, tCombined, ctx);
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

    private Tensor BuildEmptyTail(int batch)
    {
        return new Tensor(new TensorShape(batch, InChannels, 0), DType.F32);
    }

    /// <summary>Slices the trailing <paramref name="tail"/> samples from a channels-first
    /// <c>[B, C, T]</c> tensor, allocating a fresh owning tensor for the result.</summary>
    private static Tensor SliceTail(Tensor src, int batch, int channels, int tFull, int tail)
    {
        int tKeep = Math.Min(tail, tFull);
        Tensor result = new(new TensorShape(batch, channels, tKeep), DType.F32);
        float* sp = (float*)src.DataPointer;
        float* dp = (float*)result.DataPointer;
        int srcStart = tFull - tKeep;
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
            {
                int srcBase = (b * channels + c) * tFull + srcStart;
                int dstBase = (b * channels + c) * tKeep;
                for (int t = 0; t < tKeep; t++) dp[dstBase + t] = sp[srcBase + t];
            }
        return result;
    }

    /// <summary>Direct conv with zero left/right padding budget. Layout matches
    /// <see cref="VibeVoiceOps.CausalConv1d"/> but without the padding offset — used only
    /// from the streaming path where the cached history IS the left padding.</summary>
    private static void RunConvNoPad(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int batch, int cIn, int cOut, int tIn, int kernel, int stride, int dilation, int groups)
    {
        int inPerGroup = cIn / groups;
        int outPerGroup = cOut / groups;
        int tOut = (tIn - dilation * (kernel - 1) - 1) / stride + 1;

        float* ip = (float*)input.DataPointer;
        float* op = (float*)output.DataPointer;
        float* wp = (float*)weight.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int oc = 0; oc < cOut; oc++)
            {
                int group = oc / outPerGroup;
                int icStart = group * inPerGroup;
                float biasV = bp is null ? 0f : bp[oc];
                int wRow = oc * inPerGroup * kernel;
                int outBase = (b * cOut + oc) * tOut;

                for (int j = 0; j < tOut; j++)
                {
                    int jSrcStart = j * stride;
                    float acc = biasV;
                    for (int ic = 0; ic < inPerGroup; ic++)
                    {
                        int inCh = icStart + ic;
                        int inBase = (b * cIn + inCh) * tIn;
                        int wBase = wRow + ic * kernel;
                        for (int k = 0; k < kernel; k++)
                        {
                            int src = jSrcStart + k * dilation;
                            if ((uint)src < (uint)tIn)
                                acc += ip[inBase + src] * wp[wBase + k];
                        }
                    }
                    op[outBase + j] = acc;
                }
            }
        }
    }
}
