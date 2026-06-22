using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.OpenVoice;

/// <summary>Linear-spectrogram extractor (STFT magnitude, <b>not</b> mel) — the front-end OpenVoice's tone
/// converter (a VITS-derived posterior encoder) consumes. Produces the channels-first tensor
/// <c>[1, n_fft/2 + 1, T]</c> of magnitude spectra, matching the VITS <c>spectrogram_torch</c> path
/// (<c>center=True</c> reflection padding, Hann window, magnitude = sqrt(re² + im² + 1e-6)).
///
/// <para>Reuses <see cref="HannWindow"/> + <see cref="Fft"/>. No mel filterbank is applied — the tone
/// converter's posterior encoder expects raw linear bins.</para></summary>
public static unsafe class LinearSpectrogram
{
    /// <summary>Computes the magnitude STFT of <paramref name="pcm"/> → <c>[1, nFft/2 + 1, T]</c>. Uses
    /// <c>center=True</c> reflection padding so <c>T = pcm.Length / hop + 1</c>, matching torch.stft.</summary>
    public static Tensor Extract(float[] pcm, int nFft, int hop)
    {
        if (pcm is null) throw new ArgumentNullException(nameof(pcm));
        if (nFft <= 0 || hop <= 0) throw new ArgumentException("nFft and hop must be positive.");

        int half = nFft / 2;
        int numBins = half + 1;
        float[] window = HannWindow.Get(nFft);

        // center=True reflection padding by nFft/2 on each side.
        int padded = pcm.Length + nFft;
        float[] buf = new float[padded];
        for (int i = 0; i < pcm.Length; i++) buf[half + i] = pcm[i];
        for (int i = 0; i < half; i++)
        {
            int mirror = half - i;
            if (mirror < pcm.Length) buf[i] = pcm[mirror];
            int tailSrc = pcm.Length - 2 - i;
            if (tailSrc >= 0) buf[half + pcm.Length + i] = pcm[tailSrc];
        }

        int frames = pcm.Length / hop + 1;
        Tensor outT = new(new TensorShape(1, numBins, frames), DType.F32);
        float* op = (float*)outT.DataPointer;

        float[] frame = new float[nFft];
        float[] re = new float[numBins];
        float[] im = new float[numBins];
        for (int t = 0; t < frames; t++)
        {
            int start = t * hop;
            for (int i = 0; i < nFft; i++)
            {
                float sample = (start + i) < padded ? buf[start + i] : 0f;
                frame[i] = sample * window[i];
            }
            Fft.RealTransform(frame, re, im, nFft);
            for (int k = 0; k < numBins; k++)
            {
                float mag = MathF.Sqrt(re[k] * re[k] + im[k] * im[k] + 1e-6f);
                op[(long)k * frames + t] = mag;
            }
        }
        return outT;
    }
}
