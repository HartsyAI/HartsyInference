using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.VibeVoice;

/// <summary>Causal ConvTranspose1D with streaming-cache support. Used only by the
/// acoustic decoder (6 upsample stages with strides <c>[8, 5, 5, 4, 2, 2]</c>).
///
/// <para>Streaming convention (matches Python <c>SConvTranspose1d._forward_streaming</c>):
/// the per-layer cache holds the last <c>context_size = kernel - 1</c> INPUT samples. Each
/// chunk re-runs the underlying transpose conv on <c>cache + new_input</c> and returns
/// only the trailing <c>T_new * stride</c> output samples (the "new" portion). The first
/// chunk has no cache and returns the full output.</para>
///
/// <para>Non-streaming forward applies the right-trim of <c>kernel - stride</c> samples
/// inside <see cref="VibeVoiceOps.CausalConvTranspose1d"/> directly (matches
/// <c>trim_right_ratio=1.0</c>).</para></summary>
internal sealed unsafe class SConvTranspose1d
{
    public string LayerId { get; }
    public int InChannels { get; }
    public int OutChannels { get; }
    public int KernelSize { get; }
    public int Stride { get; }
    public bool Bias { get; }

    /// <summary>Trailing input samples carried between chunks.</summary>
    public int ContextSize => KernelSize - 1;

    private Tensor? _weight;
    private Tensor? _bias;

    public SConvTranspose1d(string layerId, int inChannels, int outChannels, int kernelSize, int stride, bool bias = true)
    {
        if (string.IsNullOrEmpty(layerId)) throw new ArgumentException("layerId must be non-empty.", nameof(layerId));
        LayerId = layerId;
        InChannels = inChannels;
        OutChannels = outChannels;
        KernelSize = kernelSize;
        Stride = stride;
        Bias = bias;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        // Outer is SConvTranspose1d's .convtr = NormConvTranspose1d; inner is .convtr = nn.ConvTranspose1d.
        _weight = WhisperOps.EnsureF32(w[$"{prefix}.convtr.convtr.weight"]);
        if (Bias) _bias = WhisperOps.EnsureF32(w[$"{prefix}.convtr.convtr.bias"]);
    }

    /// <summary>Non-streaming forward. Input <c>[B, C_in, T_in]</c> →
    /// <c>[B, C_out, T_in * stride]</c> after the causal right-trim.</summary>
    public Tensor Forward(Tensor input, int batch, int tIn)
    {
        if (_weight is null) throw new InvalidOperationException($"SConvTranspose1d '{LayerId}' weights not loaded.");
        int tOut = tIn * Stride;
        Tensor output = new(new TensorShape(batch, OutChannels, tOut), DType.F32);
        VibeVoiceOps.CausalConvTranspose1d(output, input, _weight!, _bias, batch, InChannels, OutChannels, tIn, KernelSize, Stride);
        return output;
    }

    /// <summary>Streaming forward — returns only the output samples produced by this
    /// chunk's new input.</summary>
    public Tensor ForwardStreaming(Tensor input, int batch, int tIn, VibeVoiceTokenizerStreamingCache cache, ReadOnlySpan<int> sampleIndices)
    {
        if (_weight is null) throw new InvalidOperationException($"SConvTranspose1d '{LayerId}' weights not loaded.");

        Tensor? cached = cache.Get(LayerId, sampleIndices, InChannels);
        int cachedLen = cached is null ? 0 : (int)cached.Shape[cached.Shape.Rank - 1];

        // Build full_input = cached + input.
        int tFull = cachedLen + tIn;
        Tensor fullInput = new(new TensorShape(batch, InChannels, tFull), DType.F32);
        float* dst = (float*)fullInput.DataPointer;
        float* xPtr = (float*)input.DataPointer;
        if (cached is not null)
        {
            float* cPtr = (float*)cached.DataPointer;
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < InChannels; c++)
                {
                    int dstBase = (b * InChannels + c) * tFull;
                    int srcBase = (b * InChannels + c) * cachedLen;
                    for (int t = 0; t < cachedLen; t++) dst[dstBase + t] = cPtr[srcBase + t];
                }
        }
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < InChannels; c++)
            {
                int dstBase = (b * InChannels + c) * tFull + cachedLen;
                int srcBase = (b * InChannels + c) * tIn;
                for (int t = 0; t < tIn; t++) dst[dstBase + t] = xPtr[srcBase + t];
            }
        cached?.Dispose();

        // Run the full transposed conv with built-in right trim. Output T = tFull * stride.
        int tOutFull = tFull * Stride;
        Tensor fullOutput = new(new TensorShape(batch, OutChannels, tOutFull), DType.F32);
        VibeVoiceOps.CausalConvTranspose1d(fullOutput, fullInput, _weight!, _bias, batch, InChannels, OutChannels, tFull, KernelSize, Stride);

        Tensor output;
        if (cachedLen == 0)
        {
            // First chunk: return everything.
            output = fullOutput;
        }
        else
        {
            // Subsequent chunks: take only the last T_new * stride samples (the new portion).
            int expected = tIn * Stride;
            int take = Math.Min(expected, tOutFull);
            output = new(new TensorShape(batch, OutChannels, take), DType.F32);
            float* sp = (float*)fullOutput.DataPointer;
            float* dp = (float*)output.DataPointer;
            int srcStart = tOutFull - take;
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < OutChannels; c++)
                {
                    int srcBase = (b * OutChannels + c) * tOutFull + srcStart;
                    int dstBase = (b * OutChannels + c) * take;
                    for (int t = 0; t < take; t++) dp[dstBase + t] = sp[srcBase + t];
                }
            fullOutput.Dispose();
        }

        // Stash the trailing ContextSize input samples into the cache.
        Tensor newTail = SliceTail(fullInput, batch, InChannels, tFull, ContextSize);
        cache.Set(LayerId, sampleIndices, newTail);
        newTail.Dispose();
        fullInput.Dispose();

        return output;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_weight is not null) yield return _weight;
        if (_bias is not null) yield return _bias;
    }

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
}
