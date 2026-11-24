using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Numerics;

namespace HartsyInference.Audio.Models.VoiceIsolator;

/// <summary>Host STFT and inverse of the isolator at <c>torch.stft</c> / <c>torch.istft</c> semantics: periodic Hann window,
/// centred frames, <c>1 + n / hop</c> frames, overlap-add divided by the squared-window envelope. The transform size must be
/// a power of two or a size <see cref="FftPlan"/> factors; it runs on the host because <c>IBackend.Stft</c> has no GPU path.</summary>
public sealed class VoiceIsolatorSpectral
{
    private readonly FftPlan _plan;
    private readonly float[] _window;

    /// <summary>Creates the transform for one window size and hop.</summary>
    public VoiceIsolatorSpectral(int nFft, int hopLength)
    {
        if (nFft < 4 || nFft % 2 != 0 || hopLength < 1 || hopLength > nFft / 2)
        {
            throw new ArgumentException("n_fft must be even and hop_length within (0, n_fft / 2].");
        }
        NFft = nFft;
        HopLength = hopLength;
        _plan = new FftPlan(nFft);
        _window = new float[nFft];
        for (int i = 0; i < nFft; i++)
        {
            _window[i] = (float)(0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / nFft));
        }
    }

    /// <summary>Window and transform size.</summary>
    public int NFft { get; }

    /// <summary>Samples between frames.</summary>
    public int HopLength { get; }

    /// <summary>Non-negative-frequency bins per frame.</summary>
    public int Bins => NFft / 2 + 1;

    /// <summary>Frames of a clip of <paramref name="samples"/> samples.</summary>
    public int FrameCount(int samples) => 1 + samples / HopLength;

    /// <summary>The periodic Hann window.</summary>
    public ReadOnlySpan<float> Window => _window;

    /// <summary>Analyses <paramref name="audio"/> into <paramref name="re"/> and <paramref name="im"/> (frame-major,
    /// <c>[frames, bins]</c>). <paramref name="reflect"/> pads the edges by reflection as <c>torch.stft</c> does for a clip longer
    /// than half a window; false pads with zeros, as a live stream sees the start and end of its input.</summary>
    public void Analyze(ReadOnlySpan<float> audio, float[] re, float[] im, bool reflect)
    {
        int frames = FrameCount(audio.Length);
        if (audio.Length == 0 || re.Length < (long)frames * Bins || im.Length < (long)frames * Bins)
        {
            throw new ArgumentException("Analysis needs audio and output buffers of frames x bins.");
        }
        int half = NFft / 2;
        float[] padded = new float[audio.Length + NFft];
        bool mirror = reflect && audio.Length > half;
        audio.CopyTo(padded.AsSpan(half));
        if (mirror)
        {
            for (int i = 1; i <= half; i++)
            {
                padded[half - i] = audio[i];
                padded[half + audio.Length - 1 + i] = audio[audio.Length - 1 - i];
            }
        }
        int bins = Bins;
        CpuParallel.For(frames, (long)frames * NFft * 16, frame =>
        {
            float[] windowed = new float[NFft];
            float[] work = new float[_plan.WorkLength];
            float[] fr = new float[bins];
            float[] fi = new float[bins];
            int start = frame * HopLength;
            for (int i = 0; i < NFft; i++)
            {
                windowed[i] = padded[start + i] * _window[i];
            }
            _plan.ForwardReal(windowed, fr, fi, work);
            Array.Copy(fr, 0, re, (long)frame * bins, bins);
            Array.Copy(fi, 0, im, (long)frame * bins, bins);
        });
    }

    /// <summary>Inverse of <see cref="Analyze"/>: <c>length</c> samples from <paramref name="frames"/> spectra
    /// (frame-major <c>[frames, bins]</c>), the overlap-add divided by the summed squared window.</summary>
    public float[] Synthesize(float[] re, float[] im, int frames, int length)
    {
        int bins = Bins;
        if (frames < 1 || length < 1 || re.Length < (long)frames * bins || im.Length < (long)frames * bins)
        {
            throw new ArgumentException("Synthesis needs frames x bins spectra and a positive length.");
        }
        float[][] time = new float[frames][];
        CpuParallel.For(frames, (long)frames * NFft * 16, frame => time[frame] = InverseFrame(re, im, frame));
        int total = NFft + HopLength * (frames - 1);
        float[] ola = new float[total];
        float[] envelope = new float[total];
        for (int frame = 0; frame < frames; frame++)
        {
            int at = frame * HopLength;
            float[] row = time[frame];
            for (int i = 0; i < NFft; i++)
            {
                ola[at + i] += row[i] * _window[i];
                envelope[at + i] += _window[i] * _window[i];
            }
        }
        float[] output = new float[length];
        int offset = NFft / 2;
        int available = Math.Min(length, total - offset);
        for (int i = 0; i < available; i++)
        {
            output[i] = ola[offset + i] / envelope[offset + i];
        }
        return output;
    }

    /// <summary>One frame's real inverse transform (unwindowed): <c>irfft</c> of bins taken from <paramref name="re"/>/<paramref name="im"/>.</summary>
    public float[] InverseFrame(float[] re, float[] im, int frame)
    {
        int bins = Bins;
        float[] inRe = new float[NFft];
        float[] inIm = new float[NFft];
        long baseIndex = (long)frame * bins;
        inRe[0] = re[baseIndex];
        inRe[NFft / 2] = re[baseIndex + NFft / 2];
        for (int k = 1; k < NFft / 2; k++)
        {
            inRe[k] = re[baseIndex + k];
            inIm[k] = -im[baseIndex + k];
            inRe[NFft - k] = re[baseIndex + k];
            inIm[NFft - k] = im[baseIndex + k];
        }
        // FftPlan.Forward shares one work buffer, so concurrent frames take the stateless transform
        Fft.Transform(inRe, inIm, NFft);
        float scale = 1f / NFft;
        for (int i = 0; i < NFft; i++)
        {
            inRe[i] *= scale;
        }
        return inRe;
    }
}
