using HartsyInference.Audio.Preprocessing;

namespace HartsyInference.Audio.Models.Vocoders;

/// <summary>Inverse Short-Time Fourier Transform — converts a complex spectrogram back
/// to a real time-domain waveform via per-frame IFFT + windowing + overlap-add. Matches
/// <c>torch.istft(window=hann_window, center=True, normalized=False)</c> within float32
/// rounding.
///
/// <para>Used by Vocos, iSTFTNet (Kokoro / StyleTTS2 decoder), and any future
/// frequency-domain vocoder we add. The implementation is pure scalar C# — for the
/// typical vocoder shapes (n_fft=1024, ~500 frames) the FFT and overlap-add together
/// run in milliseconds, so SIMD-vectorization can wait.</para></summary>
public static class IStft
{
    /// <summary>Runs iSTFT on a complex spectrogram. Input is supplied as separate
    /// real / imaginary buffers of shape <c>[frames, n_fft/2 + 1]</c> (the
    /// non-negative-frequency half — the conjugate-symmetric upper half is reconstructed
    /// here).
    ///
    /// <para>Output is a float[] of length <c>(frames - 1) * hopLength</c> with center
    /// padding (the leading and trailing <c>n_fft/2</c> samples of the raw overlap-add
    /// result are trimmed — same as <c>torch.istft(center=True)</c>).</para>
    ///
    /// <para><paramref name="edgePad"/> overrides how many samples are trimmed from each
    /// end. When negative (default) it is <c>n_fft/2</c> (center padding). Vocos "same"
    /// padding — used by NeuCodec — passes <c>(n_fft - hopLength)/2</c>, yielding
    /// <c>frames * hopLength</c> output samples.</para></summary>
    public static float[] Apply(float[] spectReal, float[] spectImag, int frames, int nFft, int hopLength, int edgePad = -1)
    {
        int half = nFft / 2;
        int numBins = half + 1;
        if (spectReal.Length < (long)frames * numBins || spectImag.Length < (long)frames * numBins)
            throw new ArgumentException("spectrogram buffers too small for the requested frame count.");

        float[] window = HannWindow.Get(nFft);
        // Sum of squared windows at each output position for normalization (overlap-add denominator).
        // torch.istft uses this to undo the overlap-add bias.
        long rawLen = (long)(frames - 1) * hopLength + nFft;
        float[] outRaw = new float[rawLen];
        float[] winSq = new float[rawLen];

        // The per-frame inverse transforms are independent, so a block of frames runs across the cores into a
        // [block, nFft] scratch; the overlap-add that follows stays sequential in frame order, so the result is
        // exactly the single-threaded sum. The block bound keeps the scratch small for long, wide spectrograms.
        int blockFrames = Math.Max(1, Math.Min(frames, (1 << 22) / nFft));
        float[] block = new float[(long)blockFrames * nFft];
        for (int blockStart = 0; blockStart < frames; blockStart += blockFrames)
        {
            int blockEnd = Math.Min(frames, blockStart + blockFrames);
            Parallel.For(blockStart, blockEnd, () => (Re: new float[nFft], Im: new float[nFft]), (f, _, scratch) =>
            {
                InverseFrame(spectReal, spectImag, f, nFft, numBins, half, scratch.Re, scratch.Im);
                Array.Copy(scratch.Re, 0, block, (long)(f - blockStart) * nFft, nFft);
                return scratch;
            }, _ => { });

            for (int f = blockStart; f < blockEnd; f++)
            {
                // Apply synthesis window and overlap-add.
                long start = (long)f * hopLength;
                long row = (long)(f - blockStart) * nFft;
                for (int i = 0; i < nFft; i++)
                {
                    outRaw[start + i] += block[row + i] * window[i];
                    winSq[start + i] += window[i] * window[i];
                }
            }
        }

        // Normalize by sum-of-squared-windows to undo the overlap-add bias. Where the
        // sum is zero (edges before any window has landed) we skip — those samples are
        // about to be trimmed by the center-padding crop anyway.
        for (long i = 0; i < rawLen; i++)
        {
            if (winSq[i] > 1e-11f) outRaw[i] /= winSq[i];
        }

        // Trim the padded edges: drop `pad` samples from each end (center: n_fft/2; Vocos "same": (n_fft-hop)/2).
        int pad = edgePad >= 0 ? edgePad : half;
        long trimmedLen = rawLen - 2L * pad;
        if (trimmedLen < 0) trimmedLen = 0;
        float[] result = new float[trimmedLen];
        Array.Copy(outRaw, pad, result, 0, trimmedLen);
        return result;
    }

    /// <summary>One frame's inverse transform: rebuilds the conjugate-symmetric spectrum in
    /// <paramref name="frameRe"/>/<paramref name="frameIm"/>, then leaves the time-domain frame in <paramref name="frameRe"/>.</summary>
    private static void InverseFrame(float[] spectReal, float[] spectImag, int f, int nFft, int numBins, int half,
        float[] frameRe, float[] frameIm)
    {
        // Reconstruct the full-length conjugate-symmetric spectrum.
        long rowOff = (long)f * numBins;
        for (int k = 0; k < numBins; k++)
        {
            frameRe[k] = spectReal[rowOff + k];
            frameIm[k] = spectImag[rowOff + k];
        }
        for (int k = 1; k < half; k++)
        {
            frameRe[nFft - k] = spectReal[rowOff + k];
            frameIm[nFft - k] = -spectImag[rowOff + k];  // conjugate
        }
        // The DC (k=0) and Nyquist (k=half) bins are real-valued; their imaginary
        // parts are already 0 by construction.

        // Inverse FFT: trick — IFFT(X) = conj(FFT(conj(X))) / N. We negate the
        // imaginary part, run the forward FFT, negate again, and divide by N.
        for (int i = 0; i < nFft; i++) frameIm[i] = -frameIm[i];
        Fft.Transform(frameRe, frameIm, nFft);
        float invN = 1f / nFft;
        for (int i = 0; i < nFft; i++)
        {
            frameRe[i] *= invN;
            // frameIm should be ~0 after the inverse — discard it.
        }
    }
}
