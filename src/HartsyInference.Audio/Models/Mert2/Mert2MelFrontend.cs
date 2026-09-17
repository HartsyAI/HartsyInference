using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Mert2;

/// <summary>MERT-v2's mel frontend: <c>torch.stft(n_fft=2048, hop=240, center=True)</c> power spectrum through a
/// filterbank, to decibels, then per-bin standardization. Everything it needs — the analysis window, the
/// filterbank, and the mean/std — is stored in the checkpoint, so nothing here is regenerated from mel-scale
/// formulas; a rebuilt filterbank would be close but not equal, and the standardization would amplify the
/// difference.
///
/// <para>Host-side by necessity: no GPU backend in this engine serves FFT. It runs once per 300-second window and
/// its scratch is allocated with the instance, so the per-frame loop allocates nothing.</para></summary>
public sealed class Mert2MelFrontend
{
    private const float LogFloor = 1e-10f;
    private const float StdFloor = 1e-5f;

    private readonly Mert2Config _config;
    private readonly int _bins;
    private readonly float[] _window;
    private readonly float[] _filterbank;   // [bins, mels], the checkpoint's own orientation
    private readonly float[] _mean;         // [mels]
    private readonly float[] _inverseStd;   // [mels]
    private readonly float[] _frameReal;    // [nFft] FFT scratch
    private readonly float[] _frameImaginary;
    private readonly float[] _power;        // [bins]
    private readonly float[] _accumulator;  // [mels]
    private bool _loaded;

    /// <summary>Allocates the frontend's scratch; call <see cref="LoadWeights"/> before <see cref="Compute"/>.</summary>
    public Mert2MelFrontend(Mert2Config config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
        _bins = config.NFft / 2 + 1;
        _window = new float[config.NFft];
        _filterbank = new float[_bins * config.MelBins];
        _mean = new float[config.MelBins];
        _inverseStd = new float[config.MelBins];
        _frameReal = new float[config.NFft];
        _frameImaginary = new float[config.NFft];
        _power = new float[_bins];
        _accumulator = new float[config.MelBins];
    }

    /// <summary>Copies the window, filterbank and standardization statistics out of the checkpoint.</summary>
    /// <param name="prefix">Module path of the frontend; the released single file stores it under
    /// <c>encoder.feature_extractor</c>.</param>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "encoder.feature_extractor")
    {
        ArgumentNullException.ThrowIfNull(weights);
        CopyF32(weights, $"{prefix}.spectrogram.window", _window, [_config.NFft]);
        // [bins, mels] and [mels, bins] hold the same number of floats, so only the full shape can tell a
        // transposed filterbank from the right one.
        CopyF32(weights, $"{prefix}.mel_scale.fb", _filterbank, [_bins, _config.MelBins]);
        CopyF32(weights, $"{prefix}.mel_mean", _mean, [_config.MelBins]);
        CopyF32(weights, $"{prefix}.mel_std", _inverseStd, [_config.MelBins]);
        for (int m = 0; m < _inverseStd.Length; m++)
        {
            _inverseStd[m] = 1f / MathF.Max(_inverseStd[m], StdFloor);
        }
        _loaded = true;
    }

    /// <summary>Turns a clip into the encoder's mel input <c>[frames, melBins]</c>, zero-padding it out to a full
    /// window first. Samples past <paramref name="clip"/> are treated as silence rather than materialized, so a
    /// 12-second clip does not cost a 29 MB copy of mostly zeros.
    ///
    /// <para>Token-major and rank 2, which is the layout every op in the subsampling stack wants; the released
    /// module carries a leading batch axis that is always 1.</para></summary>
    /// <param name="clip">Mono 24 kHz samples; longer than one window is rejected rather than truncated.</param>
    public unsafe Tensor Compute(ReadOnlySpan<float> clip)
    {
        if (!_loaded) throw new InvalidOperationException("Call LoadWeights before Compute.");
        int windowSamples = _config.WindowSeconds * _config.SampleRate;
        if (clip.Length > windowSamples)
            throw new HartsyInferenceException($"MERT2 takes at most one {_config.WindowSeconds} s window ({windowSamples} samples); got {clip.Length}.");

        int frames = _config.FramesPerWindow;
        int pad = _config.NFft / 2;
        int mels = _config.MelBins;
        Tensor mel = new(new TensorShape(frames, mels), DType.F32);
        float* output = (float*)mel.DataPointer;

        for (int t = 0; t < frames; t++)
        {
            int start = t * _config.HopLength - pad;
            for (int i = 0; i < _config.NFft; i++)
            {
                _frameReal[i] = Sample(clip, start + i, windowSamples) * _window[i];
                _frameImaginary[i] = 0f;
            }
            Fft.Transform(_frameReal, _frameImaginary, _config.NFft);
            for (int k = 0; k < _bins; k++)
            {
                _power[k] = _frameReal[k] * _frameReal[k] + _frameImaginary[k] * _frameImaginary[k];
            }

            ApplyFilterbank();
            float* row = output + (long)t * mels;
            for (int m = 0; m < mels; m++)
            {
                float decibels = 10f * MathF.Log10(MathF.Max(_accumulator[m], LogFloor));
                row[m] = (decibels - _mean[m]) * _inverseStd[m];
            }
        }
        return mel;
    }

    /// <summary>Projects the power spectrum onto the mel bins as a bin-major accumulation, so the innermost loop
    /// walks one contiguous filterbank row and vectorizes.</summary>
    private void ApplyFilterbank()
    {
        Span<float> accumulator = _accumulator;
        accumulator.Clear();
        int mels = _config.MelBins;
        int lanes = Vector<float>.Count;
        for (int k = 0; k < _bins; k++)
        {
            float magnitude = _power[k];
            if (magnitude == 0f) continue;
            ReadOnlySpan<float> filter = _filterbank.AsSpan(k * mels, mels);
            Vector<float> broadcast = new(magnitude);
            int m = 0;
            for (; m <= mels - lanes; m += lanes)
            {
                Vector<float> mixed = new Vector<float>(accumulator.Slice(m, lanes))
                    + broadcast * new Vector<float>(filter.Slice(m, lanes));
                mixed.CopyTo(accumulator.Slice(m, lanes));
            }
            for (; m < mels; m++)
            {
                accumulator[m] += magnitude * filter[m];
            }
        }
    }

    /// <summary>One sample of the centered, reflect-padded window — the same edge-excluding convention as
    /// <see cref="SignalPadding.Reflect"/>, resolved by index because the array is never built.</summary>
    private static float Sample(ReadOnlySpan<float> clip, int index, int length)
    {
        if (length > 1)
        {
            int period = 2 * (length - 1);
            int folded = ((index % period) + period) % period;
            index = folded < length ? folded : period - folded;
        }
        else
        {
            index = 0;
        }
        return index < clip.Length ? clip[index] : 0f;
    }

    private static void CopyF32(IReadOnlyDictionary<string, Tensor> weights, string key, float[] destination,
        ReadOnlySpan<int> expected)
    {
        if (!weights.TryGetValue(key, out Tensor? source))
            throw new HartsyInferenceException($"MERT2 mel frontend is missing '{key}'.");
        Mert2Ops.RequireShape(source, key, expected);
        // EnsureF32 hands back the caller's own tensor when it is already F32, so only a conversion is ours to free.
        Tensor f32 = TensorCasts.EnsureF32(source);
        try
        {
            if (f32.ElementCount != destination.Length)
                throw new HartsyInferenceException($"MERT2 '{key}' has {f32.ElementCount} elements, expected {destination.Length}.");
            f32.AsReadOnlySpan<float>().CopyTo(destination);
        }
        finally
        {
            if (!ReferenceEquals(f32, source)) f32.Dispose();
        }
    }
}
