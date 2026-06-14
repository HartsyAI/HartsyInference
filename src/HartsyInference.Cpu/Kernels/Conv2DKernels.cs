using HartsyInference.Core.Tensors;

namespace HartsyInference.Cpu.Kernels;

/// <summary>Provides 2D convolution CPU compute kernels using the im2col + GEMM approach. Input format is NCHW, weight format is [Cout, Cin, KH, KW].</summary>
public static class Conv2DKernels
{
    /// <summary>Performs 2D convolution: output = conv2d(input, weight) + bias. Uses the im2col approach to convert convolution into matrix multiplication. For 1x1 convolutions (KH=KW=1, stride=1, pad=0), im2col is skipped and the input is treated directly as a matrix.</summary>
    /// <param name="output">Output tensor with shape [N, Cout, outH, outW].</param>
    /// <param name="input">Input tensor with shape [N, Cin, H, W].</param>
    /// <param name="weight">Weight tensor with shape [Cout, Cin, KH, KW].</param>
    /// <param name="bias">Optional bias tensor with shape [Cout], or null.</param>
    /// <param name="strideH">Vertical stride.</param>
    /// <param name="strideW">Horizontal stride.</param>
    /// <param name="padH">Vertical padding.</param>
    /// <param name="padW">Horizontal padding.</param>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Conv2D(
        Tensor output,
        Tensor input,
        Tensor weight,
        Tensor? bias,
        int strideH,
        int strideW,
        int padH,
        int padW)
    {
        long N = input.Shape[0];
        long Cin = input.Shape[1];
        long H = input.Shape[2];
        long W = input.Shape[3];

        long Cout = weight.Shape[0];
        long KH = weight.Shape[2];
        long KW = weight.Shape[3];

        long outH = output.Shape[2];
        long outW = output.Shape[3];

        float* pInput = (float*)input.DataPointer;
        float* pWeight = (float*)weight.DataPointer;
        float* pOutput = (float*)output.DataPointer;
        float* pBias = bias != null ? (float*)bias.DataPointer : null;

        long colRows = Cin * KH * KW;
        long colCols = outH * outW;

        // Check for 1x1 convolution shortcut
        bool is1x1 = KH == 1 && KW == 1 && strideH == 1 && strideW == 1 && padH == 0 && padW == 0;

        for (long n = 0; n < N; n++)
        {
            float* inputSlice = pInput + n * Cin * H * W;
            float* outputSlice = pOutput + n * Cout * outH * outW;

            if (is1x1)
            {
                // For 1x1 conv, the im2col is an identity: input is [Cin, H*W], weight is [Cout, Cin]
                // output = weight @ input_reshaped
                TensorShape weightMatShape = new TensorShape(Cout, Cin);
                TensorShape inputMatShape = new TensorShape(Cin, H * W);
                TensorShape outputMatShape = new TensorShape(Cout, outH * outW);

                Tensor weightMat = new Tensor(pWeight, weightMatShape, DType.F32);
                Tensor inputMat = new Tensor(inputSlice, inputMatShape, DType.F32);
                Tensor outputMat = new Tensor(outputSlice, outputMatShape, DType.F32);

                MatMulKernels.MatMul(outputMat, weightMat, inputMat);
            }
            else
            {
                // Allocate im2col buffer: [Cin*KH*KW, outH*outW]
                long colBufferSize = colRows * colCols;
                float* colBuffer = (float*)NativeMemory.AllocZeroed((nuint)colBufferSize, (nuint)sizeof(float));

                try
                {
                    // Build the im2col matrix
                    Im2Col(inputSlice, colBuffer, Cin, H, W, KH, KW, outH, outW, strideH, strideW, padH, padW);

                    // GEMM: output[Cout, outH*outW] = weight[Cout, Cin*KH*KW] @ colBuffer[Cin*KH*KW, outH*outW]
                    TensorShape weightMatShape = new TensorShape(Cout, colRows);
                    TensorShape colMatShape = new TensorShape(colRows, colCols);
                    TensorShape outputMatShape = new TensorShape(Cout, colCols);

                    Tensor weightMat = new Tensor(pWeight, weightMatShape, DType.F32);
                    Tensor colMat = new Tensor(colBuffer, colMatShape, DType.F32);
                    Tensor outputMat = new Tensor(outputSlice, outputMatShape, DType.F32);

                    MatMulKernels.MatMul(outputMat, weightMat, colMat);
                }
                finally
                {
                    NativeMemory.Free(colBuffer);
                }
            }

            // Add bias if present
            if (pBias != null)
            {
                long spatialSize = outH * outW;
                for (long c = 0; c < Cout; c++)
                {
                    float biasVal = pBias[c];
                    float* channelOut = outputSlice + c * spatialSize;

                    int s = 0;

                    if (Avx2.IsSupported)
                    {
                        Vector256<float> vBias = Vector256.Create(biasVal);
                        int vectorEnd = (int)spatialSize - Vector256<float>.Count + 1;

                        for (; s < vectorEnd; s += Vector256<float>.Count)
                        {
                            Vector256<float> vOut = Avx.LoadVector256(channelOut + s);
                            Avx.Store(channelOut + s, Avx.Add(vOut, vBias));
                        }
                    }
                    else if (AdvSimd.IsSupported)
                    {
                        Vector128<float> vBias = Vector128.Create(biasVal);
                        int vectorEnd = (int)spatialSize - Vector128<float>.Count + 1;

                        for (; s < vectorEnd; s += Vector128<float>.Count)
                        {
                            Vector128<float> vOut = AdvSimd.LoadVector128(channelOut + s);
                            AdvSimd.Store(channelOut + s, AdvSimd.Add(vOut, vBias));
                        }
                    }

                    for (; s < (int)spatialSize; s++)
                    {
                        channelOut[s] += biasVal;
                    }
                }
            }
        }
    }

    /// <summary>Converts an input image into a column matrix for GEMM-based convolution. Each column corresponds to one output spatial location and contains the flattened receptive field from all input channels.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Im2Col(
        float* input,
        float* colBuffer,
        long Cin,
        long H,
        long W,
        long KH,
        long KW,
        long outH,
        long outW,
        int strideH,
        int strideW,
        int padH,
        int padW)
    {
        long colIdx = 0;

        for (long c = 0; c < Cin; c++)
        {
            float* channelIn = input + c * H * W;

            for (long kh = 0; kh < KH; kh++)
            {
                for (long kw = 0; kw < KW; kw++)
                {
                    for (long oh = 0; oh < outH; oh++)
                    {
                        long ih = oh * strideH - padH + kh;

                        for (long ow = 0; ow < outW; ow++)
                        {
                            long iw = ow * strideW - padW + kw;

                            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            {
                                colBuffer[colIdx] = channelIn[ih * W + iw];
                            }
                            else
                            {
                                colBuffer[colIdx] = 0.0f;
                            }

                            colIdx++;
                        }
                    }
                }
            }
        }
    }
}
