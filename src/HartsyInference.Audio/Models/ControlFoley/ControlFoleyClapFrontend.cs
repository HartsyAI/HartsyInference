using HartsyInference.Audio.Preprocessing;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>The torchlibrosa front end of the CLAP HTS-AT tower: reflect-padded Hann STFT, power spectrogram, Slaney mel bank and
/// <c>10 * log10(max(1e-10, x))</c> (reference 1, no top-dB clamp), then the repeat-pad waveform conditioning of
/// <c>laion_clap.training.data.get_audio_features</c>.</summary>
internal sealed class ControlFoleyClapFrontend
{
    private const float AmplitudeFloor = 1e-10f;

    private readonly ControlFoleyClapConfig _config;
    private readonly float[] _window;
    private readonly float[,] _melBasis;

    internal ControlFoleyClapFrontend(ControlFoleyClapConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
        _window = HannWindow.Get(config.NFft);
        _melBasis = MelFilterbank.Get(config.SampleRate, config.NFft, config.MelBins, config.Fmin, config.Fmax);
    }

    /// <summary>Repeat-pads (<c>data_filling='repeatpad'</c>) a short clip to <see cref="ControlFoleyClapConfig.ClipSamples"/>.
    /// A longer clip keeps its first window; the official <c>rand_trunc</c> picks a random one.</summary>
    internal float[] ConditionWaveform(ReadOnlySpan<float> audio)
    {
        int target = _config.ClipSamples;
        if (audio.Length == 0)
        {
            throw new ArgumentException("Reference audio is empty.", nameof(audio));
        }

        float[] result = new float[target];
        if (audio.Length >= target)
        {
            audio[..target].CopyTo(result);
            return result;
        }

        int repeats = target / audio.Length;
        for (int r = 0; r < repeats; r++)
        {
            audio.CopyTo(result.AsSpan(r * audio.Length));
        }

        return result;
    }

    /// <summary>Log-mel of a conditioned waveform as <c>[frames, MelBins]</c> row-major, <c>frames = 1 + length / hop</c>.</summary>
    internal float[] LogMel(ReadOnlySpan<float> wave)
    {
        int nFft = _config.NFft, hop = _config.Hop, mels = _config.MelBins, bins = nFft / 2 + 1;
        float[] padded = SignalPadding.Reflect(wave, nFft / 2);
        int frames = 1 + wave.Length / hop;
        float[] mel = new float[frames * mels];
        float[] frame = new float[nFft];
        float[] re = new float[bins];
        float[] im = new float[bins];
        float[] power = new float[bins];
        for (int f = 0; f < frames; f++)
        {
            int start = f * hop;
            for (int i = 0; i < nFft; i++)
            {
                frame[i] = padded[start + i] * _window[i];
            }

            Fft.RealTransform(frame, re, im, nFft);
            for (int k = 0; k < bins; k++)
            {
                power[k] = re[k] * re[k] + im[k] * im[k];
            }

            for (int m = 0; m < mels; m++)
            {
                double acc = 0.0;
                for (int k = 0; k < bins; k++)
                {
                    acc += (double)_melBasis[m, k] * power[k];
                }

                mel[f * mels + m] = 10f * MathF.Log10(MathF.Max((float)acc, AmplitudeFloor));
            }
        }

        return mel;
    }
}
