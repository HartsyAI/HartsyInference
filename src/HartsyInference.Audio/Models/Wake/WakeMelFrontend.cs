using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Wake;

/// <summary>The 32-bin mel front-end feeding <see cref="SpeechEmbeddingModel"/>, ported from openWakeWord's
/// <c>melspectrogram.onnx</c>. The STFT is a materialized DFT basis applied as a strided Conv1d (257 real +
/// 257 imaginary filters of length 512), not a radix-2 FFT, so the weights ARE the transform.
///
/// <para>Constants are taken from the graph, not from upstream docs (which are wrong on several): n_fft and
/// window are <b>512</b>, hop <b>160</b>, no centering, power spectrum, 32 mel bins, <c>10·log10</c>, then a
/// dynamic-range floor of <c>max − 80 dB</c> and the <c>x/10 + 2</c> transform openWakeWord applies to match
/// the original TensorFlow behaviour.</para>
///
/// <para><b>Input must be int16-scaled</b> (roughly ±32768), not normalized to ±1 — upstream requires 16-bit
/// PCM and converts to float without scaling. Feeding ±1 audio shifts the whole spectrum and the model
/// silently mis-scores.</para>
///
/// <para>The <c>max − 80 dB</c> floor is a reduction over the whole input buffer, so this front-end is not a
/// continuous streaming mel: callers must reproduce upstream's chunking (see
/// <c>WakeDetectionPipeline</c>) rather than feeding a rolling incremental STFT.</para></summary>
public sealed unsafe class WakeMelFrontend : IDisposable
{
    /// <summary>Samples consumed per mel frame.</summary>
    public const int HopLength = 160;
    /// <summary>FFT size and window length; the DFT basis filters are this long.</summary>
    public const int NFft = 512;
    /// <summary>Mel bins produced per frame.</summary>
    public const int MelBins = 32;
    /// <summary>Spectrogram bins before the mel projection (<c>NFft/2 + 1</c>).</summary>
    public const int SpecBins = NFft / 2 + 1;

    private const float LogFloor = 1e-10f;
    private const float DynamicRangeDb = 80f;

    private Tensor? _convReal;
    private Tensor? _convImag;
    private Tensor? _melW;
    private int _disposed;

    /// <summary>Mel frames produced for <paramref name="sampleCount"/> samples: <c>ceil(n/160 − 3)</c>.</summary>
    public static int FrameCount(int sampleCount) => Math.Max(0, (int)Math.Ceiling(sampleCount / (double)HopLength) - 3);

    /// <summary>Loads the DFT basis and mel matrix from <c>melspectrogram.onnx</c>'s initializers.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _convReal = Get(weights, "0.stft.conv_real.weight");
        _convImag = Get(weights, "0.stft.conv_imag.weight");
        _melW = Get(weights, "1.melW");

        if (_convReal.Shape[0] != SpecBins || _convReal.Shape[2] != NFft)
            throw new HartsyInferenceException($"Wake mel STFT basis must be [{SpecBins},1,{NFft}], got {_convReal.Shape}.");
        if (_melW.Shape[0] != SpecBins || _melW.Shape[1] != MelBins)
            throw new HartsyInferenceException($"Wake mel filterbank must be [{SpecBins},{MelBins}], got {_melW.Shape}.");
    }

    private static Tensor Get(IReadOnlyDictionary<string, Tensor> weights, string name) =>
        WakeWeights.Require(weights, name, "openWakeWord's melspectrogram.onnx");

    /// <summary>Computes mel frames for <paramref name="samples"/> (int16-scaled, 16 kHz mono) into
    /// <paramref name="output"/>, which must hold <c>FrameCount(samples.Length) * 32</c> floats. Returns the frame count.</summary>
    public int Compute(IBackend backend, ReadOnlySpan<float> samples, Span<float> output)
    {
        if (_convReal is null || _convImag is null || _melW is null)
            throw new InvalidOperationException("WakeMelFrontend weights not loaded.");

        int frames = FrameCount(samples.Length);
        if (frames <= 0)
            throw new ArgumentException($"Wake mel needs at least {4 * HopLength} samples to produce a frame, got {samples.Length}.", nameof(samples));
        if (output.Length < frames * MelBins)
            throw new ArgumentException($"Wake mel output needs {frames * MelBins} floats, got {output.Length}.", nameof(output));

        using Tensor input = new(new TensorShape(1, 1, samples.Length), DType.F32);
        samples.CopyTo(input.AsSpan<float>());

        // Conv1d output length is (n - 512)/160 + 1, which can exceed `frames`; the extra tail frames are
        // the ones upstream's ceil(n/160 - 3) discards.
        int convFrames = (samples.Length - NFft) / HopLength + 1;
        if (convFrames < frames)
            throw new HartsyInferenceException($"Wake mel expected at least {frames} STFT frames, got {convFrames}.");

        using Tensor real = new(new TensorShape(1, SpecBins, convFrames), DType.F32);
        using Tensor imag = new(new TensorShape(1, SpecBins, convFrames), DType.F32);
        backend.Conv1d(real, input, _convReal, null, HopLength, 0, 0, 1, 1);
        backend.Conv1d(imag, input, _convImag, null, HopLength, 0, 0, 1, 1);

        // Power spectrum, transposed to frame-major so the mel projection is one [F,257]×[257,32] GEMM.
        using Tensor power = new(new TensorShape(frames, SpecBins), DType.F32);
        float* re = (float*)real.DataPointer, im = (float*)imag.DataPointer, po = (float*)power.DataPointer;
        for (int b = 0; b < SpecBins; b++)
        {
            long src = (long)b * convFrames;
            for (int f = 0; f < frames; f++)
            {
                float r = re[src + f], i = im[src + f];
                po[(long)f * SpecBins + b] = r * r + i * i;
            }
        }

        using Tensor mel = new(new TensorShape(frames, MelBins), DType.F32);
        backend.MatMul(mel, power, _melW);

        float* mp = (float*)mel.DataPointer;
        int count = frames * MelBins;
        float max = float.NegativeInfinity;
        for (int i = 0; i < count; i++)
        {
            float db = 10f * MathF.Log10(MathF.Max(mp[i], LogFloor));
            mp[i] = db;
            if (db > max) max = db;
        }

        // Upstream clamps to (global max − 80 dB) across the whole buffer, then applies x/10 + 2.
        float floor = max - DynamicRangeDb;
        for (int i = 0; i < count; i++)
            output[i] = MathF.Max(mp[i], floor) / 10f + 2f;

        return frames;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _convReal?.Dispose();
        _convImag?.Dispose();
        _melW?.Dispose();
    }
}
