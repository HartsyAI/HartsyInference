using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Vocoders;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Numerics;

namespace HartsyInference.Audio.Models.Separation;

/// <summary>The MDX-Net separation loop around a spectrogram network, matching LongCat-Video's <c>separate_vocals</c> over
/// <c>audio-separator</c> 0.30.2: peak-normalize the 44.1 kHz stereo mix, pad it, run overlapping chunks (STFT, network, inverse STFT),
/// blend them with symmetric Hann windows, apply the compensation gain, peak-normalize the stem, then mono-mix and resample.
/// <para>Stereo only; a mono source is duplicated by the caller. The network runs one chunk at a time, which changes nothing
/// numerically against the reference's batch of two.</para></summary>
public sealed class MdxSeparator
{
    private readonly MdxSettings _settings;
    private readonly IMdxSpectrogramModel _model;
    private readonly FftPlan _plan;
    private readonly float[] _window;

    /// <summary>Creates a separator that sends each chunk's spectrogram through <paramref name="model"/>.</summary>
    public MdxSeparator(MdxSettings settings, IMdxSpectrogramModel model)
    {
        _settings = settings ?? throw new ArgumentNullException(nameof(settings));
        _model = model ?? throw new ArgumentNullException(nameof(model));
        if (settings.DimF > settings.NFft / 2 + 1) throw new ArgumentException("DimF exceeds the one-sided spectrum.", nameof(settings));
        if (settings.Overlap is < 0 or >= 1) throw new ArgumentOutOfRangeException(nameof(settings), "Overlap must be in [0, 1).");
        _plan = new FftPlan(settings.NFft);
        _window = HannWindow.Get(settings.NFft);
    }

    /// <summary>Separates the stem from a 44.1 kHz stereo mix; returns two channels of <paramref name="left"/>'s length (or
    /// shorter when <paramref name="durationSeconds"/> crops the work), normalized to the settings' peak.</summary>
    /// <param name="durationSeconds">When set, only the first <c>duration</c> seconds plus one chunk are processed, as the reference does.</param>
    /// <param name="progress">Called with (chunks done, chunks total).</param>
    public float[][] Separate(float[] left, float[] right, double? durationSeconds = null, Action<int, int>? progress = null,
        CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        if (left.Length != right.Length) throw new ArgumentException("The two channels must have the same length.", nameof(right));
        if (left.Length == 0) throw new ArgumentException("The audio to separate has no samples.", nameof(left));

        int length = left.Length;
        if (durationSeconds is not null)
        {
            long target = Math.Min((long)Math.Ceiling(durationSeconds.Value * _settings.SampleRate), length);
            length = (int)Math.Min(target + _settings.ChunkSize, length);
        }
        float[][] mix = [left.AsSpan(0, length).ToArray(), right.AsSpan(0, length).ToArray()];
        PeakNormalize(mix, _settings.NormalizationThreshold);
        float[][] stem = Demix(mix, progress, cancel);
        PeakNormalize(stem, _settings.NormalizationThreshold);
        return stem;
    }

    /// <summary>LongCat's 16 kHz mono input from the float stem, used by the video pre-step and the parity harness: averages the channels and
    /// resamples from <paramref name="sourceRate"/> to <paramref name="targetRate"/> with the SoXR-HQ stand-in. Cropping to a clip's length is the recipe's.</summary>
    public static float[] ToMono(float[][] stereo, int sourceRate, int targetRate)
    {
        ArgumentNullException.ThrowIfNull(stereo);
        float[] mono = new WavFile.DecodedAudio(stereo, sourceRate).ToMono();
        return SincResampler.ResampleSoxrHqStandIn(mono, sourceRate, targetRate);
    }

    /// <summary>Overlap-add over chunks. <paramref name="mix"/> is <c>[2][N]</c>; the result is <c>[2][N]</c> already scaled by the compensation gain.</summary>
    internal float[][] Demix(float[][] mix, Action<int, int>? progress, CancellationToken cancel)
    {
        int n = mix[0].Length;
        int chunk = _settings.ChunkSize;
        int trim = _settings.Trim;
        int gen = chunk - 2 * trim;
        int pad = gen + trim - (n % gen);
        int total = trim + n + pad;
        float[][] padded = [new float[total], new float[total]];
        for (int c = 0; c < 2; c++) mix[c].CopyTo(padded[c], trim);

        int step = (int)((1.0 - _settings.Overlap) * chunk);
        float[][] result = [new float[total], new float[total]];
        float[] divider = new float[total];
        int chunks = (total + step - 1) / step;
        float[][] part = [new float[chunk], new float[chunk]];
        float[][] predicted = [new float[chunk], new float[chunk]];
        int done = 0;
        for (int start = 0; start < total; start += step)
        {
            cancel.ThrowIfCancellationRequested();
            int end = Math.Min(start + chunk, total);
            int actual = end - start;
            for (int c = 0; c < 2; c++)
            {
                Array.Clear(part[c]);
                padded[c].AsSpan(start, actual).CopyTo(part[c]);
            }
            RunChunk(part, predicted);
            double[] weights = _settings.Overlap == 0 ? Ones(actual) : SymmetricHann(actual);
            for (int i = 0; i < actual; i++)
            {
                float w = (float)weights[i];
                divider[start + i] += w;
                result[0][start + i] += predicted[0][i] * w;
                result[1][start + i] += predicted[1][i] * w;
            }
            progress?.Invoke(++done, chunks);
        }

        float[][] source = [new float[n], new float[n]];
        for (int i = 0; i < n; i++)
        {
            float d = divider[trim + i];
            // A zero only occurs on the outermost sample of the first and last window, which the trim removes.
            float inv = d == 0f ? 0f : _settings.Compensate / d;
            source[0][i] = result[0][trim + i] * inv;
            source[1][i] = result[1][trim + i] * inv;
        }
        return source;
    }

    /// <summary>One chunk: <c>[2][chunk]</c> samples in, <c>[2][chunk]</c> separated samples out.</summary>
    private void RunChunk(float[][] chunk, float[][] output)
    {
        int bins = _settings.NFft / 2 + 1;
        int frames = _settings.Frames;
        int dimF = _settings.DimF;
        float[] spec = new float[4 * dimF * frames];
        Stft(chunk, spec);
        for (int c = 0; c < 4; c++)
        {
            for (int k = 0; k < _settings.ZeroLowBins; k++) Array.Clear(spec, (c * dimF + k) * frames, frames);
        }
        float[] pred = new float[spec.Length];
        _model.Run(spec, pred);
        float[] re = new float[(long)frames * bins];
        float[] im = new float[(long)frames * bins];
        for (int c = 0; c < 2; c++)
        {
            Array.Clear(re);
            Array.Clear(im);
            for (int k = 0; k < dimF; k++)
            {
                int reRow = ((2 * c) * dimF + k) * frames;
                int imRow = ((2 * c + 1) * dimF + k) * frames;
                for (int f = 0; f < frames; f++)
                {
                    re[f * bins + k] = pred[reRow + f];
                    im[f * bins + k] = pred[imRow + f];
                }
            }
            IStft.Apply(re, im, frames, _settings.NFft, _settings.Hop).CopyTo(output[c], 0);
        }
    }

    /// <summary><c>torch.stft(n_fft, hop, periodic Hann, center=True)</c> per channel into <c>[4, DimF, Frames]</c> (real, imaginary per channel).</summary>
    private void Stft(float[][] chunk, float[] spec)
    {
        int nFft = _settings.NFft;
        int hop = _settings.Hop;
        int frames = _settings.Frames;
        int dimF = _settings.DimF;
        int bins = nFft / 2 + 1;
        float[][] padded = [SignalPadding.Reflect(chunk[0], nFft / 2), SignalPadding.Reflect(chunk[1], nFft / 2)];
        int expected = 1 + (padded[0].Length - nFft) / hop;
        if (expected != frames) throw new InvalidOperationException($"A chunk yields {expected} frames; the model takes {frames}.");
        CpuParallel.For(2 * frames, 2L * frames * nFft * 64, job =>
        {
            int c = job / frames;
            int f = job % frames;
            float[] frame = new float[nFft];
            float[] re = new float[bins];
            float[] im = new float[bins];
            float[] work = new float[_plan.WorkLength];
            ReadOnlySpan<float> source = padded[c].AsSpan(f * hop, nFft);
            for (int i = 0; i < nFft; i++) frame[i] = source[i] * _window[i];
            _plan.ForwardReal(frame, re, im, work);
            for (int k = 0; k < dimF; k++)
            {
                spec[((2 * c) * dimF + k) * frames + f] = re[k];
                spec[((2 * c + 1) * dimF + k) * frames + f] = im[k];
            }
        });
    }

    /// <summary>No blending window, as the reference uses when chunks do not overlap: a Hann window would zero the outermost sample of each chunk.</summary>
    private static double[] Ones(int m)
    {
        double[] w = new double[m];
        Array.Fill(w, 1d);
        return w;
    }

    /// <summary><c>numpy.hanning(m)</c>: the symmetric window, <c>0.5 − 0.5·cos(2πn/(m−1))</c>.</summary>
    internal static double[] SymmetricHann(int m)
    {
        double[] w = new double[m];
        if (m == 1) { w[0] = 1d; return w; }
        for (int i = 0; i < m; i++) w[i] = 0.5 - 0.5 * Math.Cos(2d * Math.PI * i / (m - 1));
        return w;
    }

    /// <summary>Scales all channels together so the loudest sample reaches <paramref name="peak"/>, when it exceeds it (<c>spec_utils.normalize</c>).</summary>
    internal static void PeakNormalize(float[][] channels, float peak)
    {
        float max = 0f;
        foreach (float[] channel in channels)
        {
            foreach (float v in channel) max = Math.Max(max, Math.Abs(v));
        }
        if (max <= peak) return;
        float scale = peak / max;
        foreach (float[] channel in channels)
        {
            for (int i = 0; i < channel.Length; i++) channel[i] *= scale;
        }
    }
}
