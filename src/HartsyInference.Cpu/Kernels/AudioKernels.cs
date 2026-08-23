using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;

namespace HartsyInference.Cpu.Kernels;

/// <summary>CPU audio processing kernels: FFT, STFT, and mel filterbank application.</summary>
public static class AudioKernels
{
    /// <summary>Cooley-Tukey radix-2 FFT. Input is real-valued [N], output is complex [N] (interleaved real/imag). N must be a power of 2.</summary>
    public static unsafe void Fft(Tensor output, Tensor input)
    {
        long n = input.Shape[input.Shape.Rank - 1];
        if ((n & (n - 1)) != 0)
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException($"FFT size must be power of 2, got {n}.");

        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        // Copy real input into complex output (real, imag=0)
        for (int i = 0; i < (int)n; i++)
        {
            outPtr[2 * i] = inPtr[i];
            outPtr[2 * i + 1] = 0f;
        }

        FftInPlaceComplex(outPtr, (int)n);
    }

    /// <summary>In-place Cooley-Tukey radix-2 FFT over interleaved complex data [n·2]. n must be a power of 2.</summary>
    private static unsafe void FftInPlaceComplex(float* data, int n)
    {
        // Bit-reversal permutation
        int bits = BitOperations.Log2((uint)n);
        for (int i = 0; i < n; i++)
        {
            int j = ReverseBits(i, bits);
            if (j > i)
            {
                // Swap complex pairs
                float tempReal = data[2 * i];
                float tempImag = data[2 * i + 1];
                data[2 * i] = data[2 * j];
                data[2 * i + 1] = data[2 * j + 1];
                data[2 * j] = tempReal;
                data[2 * j + 1] = tempImag;
            }
        }

        // Butterfly passes
        for (int size = 2; size <= n; size *= 2)
        {
            int halfSize = size / 2;
            float angleStep = -2.0f * MathF.PI / size;

            for (int i = 0; i < n; i += size)
            {
                for (int k = 0; k < halfSize; k++)
                {
                    float angle = angleStep * k;
                    float twiddleReal = MathF.Cos(angle);
                    float twiddleImag = MathF.Sin(angle);

                    int evenIdx = 2 * (i + k);
                    int oddIdx = 2 * (i + k + halfSize);

                    float oddReal = data[oddIdx] * twiddleReal - data[oddIdx + 1] * twiddleImag;
                    float oddImag = data[oddIdx] * twiddleImag + data[oddIdx + 1] * twiddleReal;

                    data[oddIdx] = data[evenIdx] - oddReal;
                    data[oddIdx + 1] = data[evenIdx + 1] - oddImag;
                    data[evenIdx] += oddReal;
                    data[evenIdx + 1] += oddImag;
                }
            }
        }
    }

    /// <summary>Short-time Fourier transform. Applies windowed FFT to overlapping frames of the input signal. Input: [T] (time-domain signal). Output: [NumFrames, FftSize] (complex interleaved).</summary>
    public static unsafe void Stft(Tensor output, Tensor input, int fftSize, int hopLength, Tensor window)
    {
        float* signal = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        float* windowPtr = (float*)window.DataPointer;

        long signalLength = input.Shape[input.Shape.Rank - 1];
        int numFrames = (int)((signalLength - fftSize) / hopLength) + 1;

        // Temporary buffer for windowed frame — heap allocate for large FFT sizes to avoid stack overflow
        const int MaxStackAllocFloats = 4096;
        float* frameAlloc = fftSize > MaxStackAllocFloats
            ? (float*)NativeMemory.Alloc((nuint)(fftSize * sizeof(float)))
            : null;
        Span<float> frame = frameAlloc != null
            ? new Span<float>(frameAlloc, fftSize)
            : stackalloc float[fftSize];

        try
        {
            for (int f = 0; f < numFrames; f++)
            {
                int frameStart = f * hopLength;

                // Apply window
                for (int i = 0; i < fftSize; i++)
                {
                    int sampleIdx = frameStart + i;
                    frame[i] = sampleIdx < (int)signalLength ? signal[sampleIdx] * windowPtr[i] : 0f;
                }

                // In-place FFT of the frame (write to output row)
                float* frameOut = outPtr + f * fftSize * 2;

                // Copy to output as complex (real, 0)
                for (int i = 0; i < fftSize; i++)
                {
                    frameOut[2 * i] = frame[i];
                    frameOut[2 * i + 1] = 0f;
                }

                FftInPlaceComplex(frameOut, fftSize);
            }
        }
        finally
        {
            if (frameAlloc != null)
                NativeMemory.Free(frameAlloc);
        }
    }

    /// <summary>Applies a mel filterbank matrix to a magnitude spectrogram. Input: [NumFrames, FreqBins], Filters: [NumMels, FreqBins], Output: [NumFrames, NumMels].</summary>
    public static unsafe void MelFilterbank(Tensor output, Tensor input, Tensor filters)
    {
        float* inPtr = (float*)input.DataPointer;
        float* filterPtr = (float*)filters.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        int numFrames = (int)input.Shape[0];
        int freqBins = (int)input.Shape[1];
        int numMels = (int)filters.Shape[0];

        // Pre-allocate the NEON reduction buffer outside the loop to avoid CA2014
        float* neonTmp = stackalloc float[Vector128<float>.Count];

        for (int f = 0; f < numFrames; f++)
        {
            float* frameIn = inPtr + f * freqBins;
            float* frameOut = outPtr + f * numMels;

            for (int m = 0; m < numMels; m++)
            {
                float* filterRow = filterPtr + m * freqBins;
                float sum = 0f;

                int i = 0;
                if (Avx2.IsSupported)
                {
                    Vector256<float> vSum = Vector256<float>.Zero;
                    for (; i <= freqBins - 8; i += 8)
                    {
                        Vector256<float> vIn = Avx.LoadVector256(frameIn + i);
                        Vector256<float> vFilter = Avx.LoadVector256(filterRow + i);
                        vSum = Fma.IsSupported
                            ? Fma.MultiplyAdd(vIn, vFilter, vSum)
                            : Avx.Add(vSum, Avx.Multiply(vIn, vFilter));
                    }
                    sum += SimdDispatch.HorizontalSum(vSum);
                }
                else if (AdvSimd.IsSupported)
                {
                    Vector128<float> vSum = Vector128<float>.Zero;
                    for (; i <= freqBins - 4; i += 4)
                    {
                        Vector128<float> vIn = AdvSimd.LoadVector128(frameIn + i);
                        Vector128<float> vFilter = AdvSimd.LoadVector128(filterRow + i);
                        vSum = AdvSimd.Add(vSum, AdvSimd.Multiply(vIn, vFilter));
                    }
                    AdvSimd.Store(neonTmp, vSum);
                    for (int j = 0; j < Vector128<float>.Count; j++)
                        sum += neonTmp[j];
                }

                for (; i < freqBins; i++)
                {
                    sum += frameIn[i] * filterRow[i];
                }

                frameOut[m] = sum;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ReverseBits(int value, int numBits)
    {
        int result = 0;
        for (int i = 0; i < numBits; i++)
        {
            result = (result << 1) | (value & 1);
            value >>= 1;
        }
        return result;
    }
}
