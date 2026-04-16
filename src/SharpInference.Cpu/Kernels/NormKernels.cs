using SharpInference.Core.Tensors;

namespace SharpInference.Cpu.Kernels;

/// <summary>Provides normalization CPU compute kernels (GroupNorm, LayerNorm, RmsNorm) with SIMD-accelerated sum and sum-of-squares reductions.</summary>
public static class NormKernels
{
    /// <summary>Applies Group Normalization to a 4D tensor with shape [N, C, H, W]. Channels are split into <paramref name="groups"/> groups, and each group is independently normalized with mean/variance, then scaled by weight and bias.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void GroupNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps)
    {
        long N = input.Shape[0];
        long C = input.Shape[1];
        long H = input.Shape[2];
        long W = input.Shape[3];
        long channelsPerGroup = C / groups;
        long spatialSize = H * W;
        long groupSize = channelsPerGroup * spatialSize;

        float* pIn = (float*)input.DataPointer;
        float* pOut = (float*)output.DataPointer;
        float* pWeight = (float*)weight.DataPointer;
        float* pBias = (float*)bias.DataPointer;

        for (long n = 0; n < N; n++)
        {
            for (int g = 0; g < groups; g++)
            {
                long channelStart = g * channelsPerGroup;

                // Compute mean and variance for this group
                float sum = 0.0f;
                float sumSq = 0.0f;

                for (long c = channelStart; c < channelStart + channelsPerGroup; c++)
                {
                    long baseIdx = (n * C + c) * spatialSize;
                    VectorizedSumAndSumSq(pIn + baseIdx, (int)spatialSize, out float partialSum, out float partialSumSq);
                    sum += partialSum;
                    sumSq += partialSumSq;
                }

                float mean = sum / groupSize;
                float variance = sumSq / groupSize - mean * mean;
                float invStd = 1.0f / MathF.Sqrt(variance + eps);

                // Normalize and apply affine transform
                for (long c = channelStart; c < channelStart + channelsPerGroup; c++)
                {
                    long baseIdx = (n * C + c) * spatialSize;
                    float w = pWeight[c];
                    float b = pBias[c];

                    for (long s = 0; s < spatialSize; s++)
                    {
                        long idx = baseIdx + s;
                        pOut[idx] = (pIn[idx] - mean) * invStd * w + b;
                    }
                }
            }
        }
    }

    /// <summary>Applies Layer Normalization over the last dimension of the input tensor. For each element along the outer dimensions, the last dimension is independently normalized, then scaled by weight and shifted by bias.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void LayerNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, float eps)
    {
        int rank = input.Shape.Rank;
        long lastDim = input.Shape[rank - 1];
        long outerSize = input.ElementCount / lastDim;

        float* pIn = (float*)input.DataPointer;
        float* pOut = (float*)output.DataPointer;
        float* pWeight = (float*)weight.DataPointer;
        float* pBias = (float*)bias.DataPointer;

        for (long outer = 0; outer < outerSize; outer++)
        {
            long baseIdx = outer * lastDim;

            VectorizedSumAndSumSq(pIn + baseIdx, (int)lastDim, out float sum, out float sumSq);

            float mean = sum / lastDim;
            float variance = sumSq / lastDim - mean * mean;
            float invStd = 1.0f / MathF.Sqrt(variance + eps);

            int i = 0;

            if (Avx2.IsSupported)
            {
                Vector256<float> vMean = Vector256.Create(mean);
                Vector256<float> vInvStd = Vector256.Create(invStd);
                int vectorEnd = (int)lastDim - Vector256<float>.Count + 1;

                for (; i < vectorEnd; i += Vector256<float>.Count)
                {
                    Vector256<float> vIn = Avx.LoadVector256(pIn + baseIdx + i);
                    Vector256<float> vW = Avx.LoadVector256(pWeight + i);
                    Vector256<float> vB = Avx.LoadVector256(pBias + i);
                    Vector256<float> normalized = Avx.Multiply(Avx.Subtract(vIn, vMean), vInvStd);
                    Vector256<float> result = Avx.Add(Avx.Multiply(normalized, vW), vB);
                    Avx.Store(pOut + baseIdx + i, result);
                }
            }
            else if (AdvSimd.IsSupported)
            {
                Vector128<float> vMean = Vector128.Create(mean);
                Vector128<float> vInvStd = Vector128.Create(invStd);
                int vectorEnd = (int)lastDim - Vector128<float>.Count + 1;

                for (; i < vectorEnd; i += Vector128<float>.Count)
                {
                    Vector128<float> vIn = AdvSimd.LoadVector128(pIn + baseIdx + i);
                    Vector128<float> vW = AdvSimd.LoadVector128(pWeight + i);
                    Vector128<float> vB = AdvSimd.LoadVector128(pBias + i);
                    Vector128<float> normalized = AdvSimd.Multiply(AdvSimd.Subtract(vIn, vMean), vInvStd);
                    Vector128<float> result = AdvSimd.Add(AdvSimd.Multiply(normalized, vW), vB);
                    AdvSimd.Store(pOut + baseIdx + i, result);
                }
            }

            for (; i < (int)lastDim; i++)
            {
                long idx = baseIdx + i;
                pOut[idx] = (pIn[idx] - mean) * invStd * pWeight[i] + pBias[i];
            }
        }
    }

    /// <summary>Applies Root Mean Square Normalization over the last dimension of the input tensor. RMSNorm omits the mean centering step: output = (x / rms(x)) * weight, where rms(x) = sqrt(mean(x^2) + eps).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void RmsNorm(Tensor output, Tensor input, Tensor weight, float eps)
    {
        int rank = input.Shape.Rank;
        long lastDim = input.Shape[rank - 1];
        long outerSize = input.ElementCount / lastDim;

        float* pIn = (float*)input.DataPointer;
        float* pOut = (float*)output.DataPointer;
        float* pWeight = (float*)weight.DataPointer;

        for (long outer = 0; outer < outerSize; outer++)
        {
            long baseIdx = outer * lastDim;

            float sumSq = VectorizedSumOfSquares(pIn + baseIdx, (int)lastDim);
            float rms = MathF.Sqrt(sumSq / lastDim + eps);
            float invRms = 1.0f / rms;

            int i = 0;

            if (Avx2.IsSupported)
            {
                Vector256<float> vInvRms = Vector256.Create(invRms);
                int vectorEnd = (int)lastDim - Vector256<float>.Count + 1;

                for (; i < vectorEnd; i += Vector256<float>.Count)
                {
                    Vector256<float> vIn = Avx.LoadVector256(pIn + baseIdx + i);
                    Vector256<float> vW = Avx.LoadVector256(pWeight + i);
                    Vector256<float> result = Avx.Multiply(Avx.Multiply(vIn, vInvRms), vW);
                    Avx.Store(pOut + baseIdx + i, result);
                }
            }
            else if (AdvSimd.IsSupported)
            {
                Vector128<float> vInvRms = Vector128.Create(invRms);
                int vectorEnd = (int)lastDim - Vector128<float>.Count + 1;

                for (; i < vectorEnd; i += Vector128<float>.Count)
                {
                    Vector128<float> vIn = AdvSimd.LoadVector128(pIn + baseIdx + i);
                    Vector128<float> vW = AdvSimd.LoadVector128(pWeight + i);
                    Vector128<float> result = AdvSimd.Multiply(AdvSimd.Multiply(vIn, vInvRms), vW);
                    AdvSimd.Store(pOut + baseIdx + i, result);
                }
            }

            for (; i < (int)lastDim; i++)
            {
                long idx = baseIdx + i;
                pOut[idx] = pIn[idx] * invRms * pWeight[i];
            }
        }
    }

    /// <summary>Computes vectorized sum and sum-of-squares of a float buffer.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void VectorizedSumAndSumSq(float* ptr, int count, out float sum, out float sumSq)
    {
        sum = 0.0f;
        sumSq = 0.0f;
        int i = 0;

        if (Avx2.IsSupported)
        {
            Vector256<float> vSum = Vector256<float>.Zero;
            Vector256<float> vSumSq = Vector256<float>.Zero;
            int vectorEnd = count - Vector256<float>.Count + 1;

            for (; i < vectorEnd; i += Vector256<float>.Count)
            {
                Vector256<float> v = Avx.LoadVector256(ptr + i);
                vSum = Avx.Add(vSum, v);
                vSumSq = Fma.IsSupported
                    ? Fma.MultiplyAdd(v, v, vSumSq)
                    : Avx.Add(vSumSq, Avx.Multiply(v, v));
            }

            // Horizontal reduction
            float* tmpSum = stackalloc float[Vector256<float>.Count];
            float* tmpSumSq = stackalloc float[Vector256<float>.Count];
            Avx.Store(tmpSum, vSum);
            Avx.Store(tmpSumSq, vSumSq);

            for (int j = 0; j < Vector256<float>.Count; j++)
            {
                sum += tmpSum[j];
                sumSq += tmpSumSq[j];
            }
        }
        else if (AdvSimd.IsSupported)
        {
            Vector128<float> vSum = Vector128<float>.Zero;
            Vector128<float> vSumSq = Vector128<float>.Zero;
            int vectorEnd = count - Vector128<float>.Count + 1;

            for (; i < vectorEnd; i += Vector128<float>.Count)
            {
                Vector128<float> v = AdvSimd.LoadVector128(ptr + i);
                vSum = AdvSimd.Add(vSum, v);
                vSumSq = AdvSimd.Add(vSumSq, AdvSimd.Multiply(v, v));
            }

            float* tmpSum = stackalloc float[Vector128<float>.Count];
            float* tmpSumSq = stackalloc float[Vector128<float>.Count];
            AdvSimd.Store(tmpSum, vSum);
            AdvSimd.Store(tmpSumSq, vSumSq);

            for (int j = 0; j < Vector128<float>.Count; j++)
            {
                sum += tmpSum[j];
                sumSq += tmpSumSq[j];
            }
        }

        for (; i < count; i++)
        {
            float val = ptr[i];
            sum += val;
            sumSq += val * val;
        }
    }

    /// <summary>Computes vectorized sum of squares of a float buffer.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float VectorizedSumOfSquares(float* ptr, int count)
    {
        float sumSq = 0.0f;
        int i = 0;

        if (Avx2.IsSupported)
        {
            Vector256<float> vSumSq = Vector256<float>.Zero;
            int vectorEnd = count - Vector256<float>.Count + 1;

            for (; i < vectorEnd; i += Vector256<float>.Count)
            {
                Vector256<float> v = Avx.LoadVector256(ptr + i);
                vSumSq = Fma.IsSupported
                    ? Fma.MultiplyAdd(v, v, vSumSq)
                    : Avx.Add(vSumSq, Avx.Multiply(v, v));
            }

            float* tmp = stackalloc float[Vector256<float>.Count];
            Avx.Store(tmp, vSumSq);
            for (int j = 0; j < Vector256<float>.Count; j++)
            {
                sumSq += tmp[j];
            }
        }
        else if (AdvSimd.IsSupported)
        {
            Vector128<float> vSumSq = Vector128<float>.Zero;
            int vectorEnd = count - Vector128<float>.Count + 1;

            for (; i < vectorEnd; i += Vector128<float>.Count)
            {
                Vector128<float> v = AdvSimd.LoadVector128(ptr + i);
                vSumSq = AdvSimd.Add(vSumSq, AdvSimd.Multiply(v, v));
            }

            float* tmp = stackalloc float[Vector128<float>.Count];
            AdvSimd.Store(tmp, vSumSq);
            for (int j = 0; j < Vector128<float>.Count; j++)
            {
                sumSq += tmp[j];
            }
        }

        for (; i < count; i++)
        {
            sumSq += ptr[i] * ptr[i];
        }

        return sumSq;
    }
}
