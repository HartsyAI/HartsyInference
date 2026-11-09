using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Cpu.Kernels;

/// <summary>CPU audio processing kernels: FFT, STFT, and mel filterbank application.</summary>
public static class AudioKernels
{

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
        float* frameAlloc = fftSize > MaxStackAllocFloats ? (float*)NativeMemory.Alloc((nuint)(fftSize * sizeof(float)))
            : null;
        Span<float> frame = frameAlloc != null ? new Span<float>(frameAlloc, fftSize) : stackalloc float[fftSize];

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
