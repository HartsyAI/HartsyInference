using System.Buffers;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Numerics;

namespace HartsyInference.Audio.Models.Vocoders;

/// <summary>Inverse Short-Time Fourier Transform — converts a complex spectrogram back
/// to a real time-domain waveform via per-frame IFFT + windowing + overlap-add. Matches
/// <c>torch.istft(window=hann_window, center=True, normalized=False)</c> within float32
/// rounding.
///
/// <para>Used by Vocos, iSTFTNet (Kokoro / StyleTTS2 decoder), and any future
/// frequency-domain vocoder we add. The per-frame inverse transforms fan out through
/// <see cref="CpuParallel"/>; the overlap-add stays sequential in frame order.</para></summary>
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
        int rawLen = checked((frames - 1) * hopLength + nFft);
        // The per-frame inverse transforms are independent, so fixed-size blocks of frames run through CpuParallel
        // into a [frames, nFft] buffer; the overlap-add that follows stays sequential in frame order, so the result is
        // exactly the single-threaded sum. Spans of whole blocks bound that buffer for long, wide spectrograms.
        int framesPerBlock = FramePartition.FramesPerBlock(nFft, FramePartition.TransformBudget);
        int spanFrames = Math.Max(framesPerBlock, Math.Min(frames, (1 << 20) / nFft / framesPerBlock * framesPerBlock));
        float[] outRaw = ArrayPool<float>.Shared.Rent(Math.Max(0, rawLen));
        float[] winSq = ArrayPool<float>.Shared.Rent(Math.Max(0, rawLen));
        float[] frameBuffer = ArrayPool<float>.Shared.Rent(checked(spanFrames * nFft));
        try
        {
            Array.Clear(outRaw, 0, Math.Max(0, rawLen));
            Array.Clear(winSq, 0, Math.Max(0, rawLen));
            for (int spanStart = 0; spanStart < frames; spanStart += spanFrames)
            {
                int first = spanStart;
                int last = Math.Min(frames, spanStart + spanFrames);
                int blocks = FramePartition.BlockCount(last - first, framesPerBlock);
                CpuParallel.For(blocks, (last - first) * FramePartition.TransformWork(nFft), b =>
                {
                    float[] scratch = ArrayPool<float>.Shared.Rent(2 * nFft);
                    try
                    {
                        Span<float> re = scratch.AsSpan(0, nFft);
                        Span<float> im = scratch.AsSpan(nFft, nFft);
                        int blockEnd = Math.Min(last, first + (b + 1) * framesPerBlock);
                        for (int f = first + b * framesPerBlock; f < blockEnd; f++)
                        {
                            InverseFrame(spectReal, spectImag, f, nFft, numBins, half, re, im);
                            re.CopyTo(frameBuffer.AsSpan((f - first) * nFft, nFft));
                        }
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(scratch);
                    }
                });

                for (int f = first; f < last; f++)
                {
                    // Apply synthesis window and overlap-add.
                    int start = f * hopLength;
                    int row = (f - first) * nFft;
                    for (int i = 0; i < nFft; i++)
                    {
                        outRaw[start + i] += frameBuffer[row + i] * window[i];
                        winSq[start + i] += window[i] * window[i];
                    }
                }
            }

            // Normalize by sum-of-squared-windows to undo the overlap-add bias. Where the
            // sum is zero (edges before any window has landed) we skip — those samples are
            // about to be trimmed by the center-padding crop anyway.
            for (int i = 0; i < rawLen; i++)
            {
                if (winSq[i] > 1e-11f) outRaw[i] /= winSq[i];
            }

            // Trim the padded edges: drop `pad` samples from each end (center: n_fft/2; Vocos "same": (n_fft-hop)/2).
            int pad = edgePad >= 0 ? edgePad : half;
            int trimmedLen = Math.Max(0, rawLen - 2 * pad);
            float[] result = new float[trimmedLen];
            Array.Copy(outRaw, pad, result, 0, trimmedLen);
            return result;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(outRaw);
            ArrayPool<float>.Shared.Return(winSq);
            ArrayPool<float>.Shared.Return(frameBuffer);
        }
    }

    /// <summary>One frame's inverse transform: rebuilds the conjugate-symmetric spectrum in
    /// <paramref name="frameRe"/>/<paramref name="frameIm"/>, then leaves the time-domain frame in <paramref name="frameRe"/>.</summary>
    private static void InverseFrame(float[] spectReal, float[] spectImag, int f, int nFft, int numBins, int half,
        Span<float> frameRe, Span<float> frameIm)
    {
        // An odd nFft leaves one mirrored bin unwritten below; clearing keeps a reused scratch from carrying a
        // previous frame into it.
        frameRe.Clear();
        frameIm.Clear();
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
