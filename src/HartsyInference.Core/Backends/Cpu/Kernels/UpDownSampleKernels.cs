using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Cpu.Kernels;

/// <summary>Provides upsampling and downsampling CPU compute kernels for NCHW format F32 tensors. Nearest-neighbor only.</summary>
public static class UpDownSampleKernels
{
    /// <summary>Performs nearest-neighbor 2D upsampling on an NCHW tensor. Each input pixel is replicated into a <paramref name="scaleH"/> x <paramref name="scaleW"/> block. Output shape is [N, C, H * scaleH, W * scaleW].</summary>
    /// <param name="output">Output tensor with shape [N, C, H*scaleH, W*scaleW].</param>
    /// <param name="input">Input tensor with shape [N, C, H, W].</param>
    /// <param name="scaleH">Vertical upsampling factor.</param>
    /// <param name="scaleW">Horizontal upsampling factor.</param>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW)
    {
        if (input.DType != DType.F32 || output.DType != DType.F32)
            throw new NotSupportedException($"CPU UpsampleNearest2D is F32-only; got input {input.DType}, output {output.DType}.");
        long N = input.Shape[0];
        long C = input.Shape[1];
        long inH = input.Shape[2];
        long inW = input.Shape[3];
        long outH = inH * scaleH;
        long outW = inW * scaleW;

        float* pIn = (float*)input.DataPointer;
        float* pOut = (float*)output.DataPointer;

        for (long n = 0; n < N; n++)
        {
            for (long c = 0; c < C; c++)
            {
                long inChannelOffset = (n * C + c) * inH * inW;
                long outChannelOffset = (n * C + c) * outH * outW;

                for (long oh = 0; oh < outH; oh++)
                {
                    long ih = oh / scaleH;
                    float* inRow = pIn + inChannelOffset + ih * inW;
                    float* outRow = pOut + outChannelOffset + oh * outW;

                    if (scaleW == 1)
                    {
                        // No horizontal scaling: direct copy
                        long byteCount = inW * sizeof(float);
                        Buffer.MemoryCopy(inRow, outRow, byteCount, byteCount);
                    }
                    else
                    {
                        if (Avx2.IsSupported && scaleW >= 2)
                        {
                            // For each input pixel, broadcast to scaleW output pixels
                            for (long iw = 0; iw < inW; iw++)
                            {
                                float val = inRow[iw];
                                Vector256<float> vVal = Vector256.Create(val);
                                int remaining = scaleW;
                                int outIdx = (int)(iw * scaleW);

                                while (remaining >= Vector256<float>.Count)
                                {
                                    Avx.Store(outRow + outIdx, vVal);
                                    outIdx += Vector256<float>.Count;
                                    remaining -= Vector256<float>.Count;
                                }

                                for (int r = 0; r < remaining; r++)
                                {
                                    outRow[outIdx + r] = val;
                                }
                            }
                        }
                        else
                        {
                            for (long iw = 0; iw < inW; iw++)
                            {
                                float val = inRow[iw];
                                long baseOw = iw * scaleW;

                                for (int sw = 0; sw < scaleW; sw++)
                                {
                                    outRow[baseOw + sw] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

}
