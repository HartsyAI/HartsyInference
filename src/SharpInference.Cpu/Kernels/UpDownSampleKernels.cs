using System.Runtime.InteropServices;
using SharpInference.Core.Tensors;

namespace SharpInference.Cpu.Kernels;

/// <summary>Provides upsampling and downsampling CPU compute kernels for NCHW format F32 tensors. Supports nearest-neighbor and bilinear interpolation modes.</summary>
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

    /// <summary>Performs bilinear 2D upsampling on an NCHW tensor. Uses bilinear interpolation to compute output values from the four nearest input pixels. Output shape is [N, C, H * scaleH, W * scaleW]. Coordinate mapping uses align_corners=false convention: src = (dst + 0.5) / scale - 0.5.</summary>
    /// <param name="output">Output tensor with shape [N, C, H*scaleH, W*scaleW].</param>
    /// <param name="input">Input tensor with shape [N, C, H, W].</param>
    /// <param name="scaleH">Vertical upsampling factor.</param>
    /// <param name="scaleW">Horizontal upsampling factor.</param>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void UpsampleBilinear2D(Tensor output, Tensor input, int scaleH, int scaleW)
    {
        long N = input.Shape[0];
        long C = input.Shape[1];
        long inH = input.Shape[2];
        long inW = input.Shape[3];
        long outH = inH * scaleH;
        long outW = inW * scaleW;

        float* pIn = (float*)input.DataPointer;
        float* pOut = (float*)output.DataPointer;

        float scaleFactorH = (float)scaleH;
        float scaleFactorW = (float)scaleW;

        // Heap-allocate precomputed tables to avoid stack overflow with large dimensions
        nuint yTableBytes = (nuint)(outH * sizeof(int));
        nuint yWeightBytes = (nuint)(outH * sizeof(float));
        nuint xTableBytes = (nuint)(outW * sizeof(int));
        nuint xWeightBytes = (nuint)(outW * sizeof(float));

        int* srcY0 = (int*)NativeMemory.Alloc(yTableBytes);
        int* srcY1 = (int*)NativeMemory.Alloc(yTableBytes);
        float* yWeights = (float*)NativeMemory.Alloc(yWeightBytes);
        int* srcX0 = (int*)NativeMemory.Alloc(xTableBytes);
        int* srcX1 = (int*)NativeMemory.Alloc(xTableBytes);
        float* xWeights = (float*)NativeMemory.Alloc(xWeightBytes);

        try
        {
            // Precompute vertical mapping
            for (long oh = 0; oh < outH; oh++)
            {
                float srcY = ((float)oh + 0.5f) / scaleFactorH - 0.5f;
                int y0 = (int)MathF.Floor(srcY);
                int y1Clamped = y0 + 1;

                if (y0 < 0) y0 = 0;
                if (y1Clamped >= (int)inH) y1Clamped = (int)inH - 1;
                if (y0 >= (int)inH) y0 = (int)inH - 1;

                float wy = srcY - MathF.Floor(srcY);
                if (srcY < 0.0f) wy = 0.0f;

                srcY0[(int)oh] = y0;
                srcY1[(int)oh] = y1Clamped;
                yWeights[(int)oh] = wy;
            }

            // Precompute horizontal mapping
            for (long ow = 0; ow < outW; ow++)
            {
                float srcX = ((float)ow + 0.5f) / scaleFactorW - 0.5f;
                int x0 = (int)MathF.Floor(srcX);
                int x1Clamped = x0 + 1;

                if (x0 < 0) x0 = 0;
                if (x1Clamped >= (int)inW) x1Clamped = (int)inW - 1;
                if (x0 >= (int)inW) x0 = (int)inW - 1;

                float wx = srcX - MathF.Floor(srcX);
                if (srcX < 0.0f) wx = 0.0f;

                srcX0[(int)ow] = x0;
                srcX1[(int)ow] = x1Clamped;
                xWeights[(int)ow] = wx;
            }

            for (long n = 0; n < N; n++)
            {
                for (long c = 0; c < C; c++)
                {
                    long inChannelOffset = (n * C + c) * inH * inW;
                    long outChannelOffset = (n * C + c) * outH * outW;
                    float* inChannel = pIn + inChannelOffset;
                    float* outChannel = pOut + outChannelOffset;

                    for (long oh = 0; oh < outH; oh++)
                    {
                        int y0 = srcY0[(int)oh];
                        int y1 = srcY1[(int)oh];
                        float wy = yWeights[(int)oh];
                        float wy0 = 1.0f - wy;

                        float* inRowTop = inChannel + y0 * inW;
                        float* inRowBot = inChannel + y1 * inW;
                        float* outRow = outChannel + oh * outW;

                        for (long ow = 0; ow < outW; ow++)
                        {
                            int x0 = srcX0[(int)ow];
                            int x1 = srcX1[(int)ow];
                            float wx = xWeights[(int)ow];
                            float wx0 = 1.0f - wx;

                            float topLeft = inRowTop[x0];
                            float topRight = inRowTop[x1];
                            float botLeft = inRowBot[x0];
                            float botRight = inRowBot[x1];

                            float top = topLeft * wx0 + topRight * wx;
                            float bot = botLeft * wx0 + botRight * wx;
                            outRow[ow] = top * wy0 + bot * wy;
                        }
                    }
                }
            }
        }
        finally
        {
            NativeMemory.Free(srcY0);
            NativeMemory.Free(srcY1);
            NativeMemory.Free(yWeights);
            NativeMemory.Free(srcX0);
            NativeMemory.Free(srcX1);
            NativeMemory.Free(xWeights);
        }
    }
}
