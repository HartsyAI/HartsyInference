using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.VibeVoice;

/// <summary>Causal ConvTranspose1D with streaming-cache support; used only by the acoustic decoder (6 upsample stages with strides <c>[8, 5, 5, 4, 2, 2]</c>).</summary>
/// <remarks>Streaming convention (matches Python <c>SConvTranspose1d._forward_streaming</c>):
/// the per-layer cache holds the last <c>context_size = kernel - 1</c> INPUT samples. Each
/// chunk re-runs the underlying transpose conv on <c>cache + new_input</c> and returns
/// only the trailing <c>T_new * stride</c> output samples (the "new" portion). The first
/// chunk has no cache and returns the full output.
///
/// <para>Non-streaming forward applies the right-trim of <c>kernel - stride</c> samples directly via
/// <see cref="IBackend.ConvTranspose1d"/> (matches <c>trim_right_ratio=1.0</c>).</para></remarks>
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

    /// <summary>Non-streaming forward, routed to <see cref="IBackend.ConvTranspose1d"/> with <c>padLeft=0, padRight=K-stride</c> (trim_right_ratio=1.0) so output is exactly <c>[B, C_out, T_in * stride]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor input, int batch, int tIn)
    {
        if (_weight is null) throw new InvalidOperationException($"SConvTranspose1d '{LayerId}' weights not loaded.");
        int tOut = tIn * Stride;
        Tensor output = new(new TensorShape(batch, OutChannels, tOut), DType.F32);
        backend.ConvTranspose1d(output, input, _weight!, _bias, Stride, 0, KernelSize - Stride, 1, 1);
        return output;
    }

    /// <summary>Streaming forward — returns only the output samples produced by this chunk's new input.</summary>
    public Tensor ForwardStreaming(IBackend backend, Tensor input, int batch, int tIn, VibeVoiceTokenizerStreamingCache cache, ReadOnlySpan<int> sampleIndices)
    {
        if (_weight is null) throw new InvalidOperationException($"SConvTranspose1d '{LayerId}' weights not loaded.");

        Tensor? cached = cache.Get(LayerId, sampleIndices, InChannels);
        int cachedLen = cached is null ? 0 : (int)cached.Shape[cached.Shape.Rank - 1];

        // Build full_input = cached + input on-device (no zero-pad prepend on the first chunk —
        // the transpose streaming convention runs on the raw chunk and returns the new portion).
        int tFull = cachedLen + tIn;
        Tensor fullInput;
        if (cached is not null)
        {
            fullInput = new(new TensorShape(batch, InChannels, tFull), DType.F32);
            backend.Concat(fullInput, [cached, input], dim: 2);
            cached.Dispose();
        }
        else
        {
            fullInput = new(input.Shape, DType.F32);
            backend.CopyInto(fullInput, input);
        }

        // Full transposed conv with built-in right trim (padRight = K - stride). T = tFull * stride.
        int tOutFull = tFull * Stride;
        Tensor fullOutput = new(new TensorShape(batch, OutChannels, tOutFull), DType.F32);
        backend.ConvTranspose1d(fullOutput, fullInput, _weight!, _bias, Stride, 0, KernelSize - Stride, 1, 1);

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
            backend.SliceLastDim(output, fullOutput, tOutFull - take);
            fullOutput.Dispose();
        }

        // Stash the trailing ContextSize input samples into the cache.
        int tKeep = Math.Min(ContextSize, tFull);
        Tensor newTail = new(new TensorShape(batch, InChannels, tKeep), DType.F32);
        backend.SliceLastDim(newTail, fullInput, tFull - tKeep);
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
}
