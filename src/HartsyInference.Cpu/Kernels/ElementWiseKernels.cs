using HartsyInference.Core.Tensors;

namespace HartsyInference.Cpu.Kernels;

/// <summary>Provides element-wise CPU compute kernels with SIMD acceleration for F32 tensors.</summary>
public static class ElementWiseKernels
{
    /// <summary>Performs element-wise addition: output = a + b. Uses AVX2/NEON vectorization with scalar fallback.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Add(Tensor output, Tensor a, Tensor b)
    {
        int count = (int)output.ElementCount;
        float* pOut = (float*)output.DataPointer;
        float* pA = (float*)a.DataPointer;
        float* pB = (float*)b.DataPointer;

        int i = 0;

        if (Avx2.IsSupported)
        {
            int vectorEnd = count - Vector256<float>.Count + 1;
            for (; i < vectorEnd; i += Vector256<float>.Count)
            {
                Vector256<float> va = Avx.LoadVector256(pA + i);
                Vector256<float> vb = Avx.LoadVector256(pB + i);
                Vector256<float> result = Avx.Add(va, vb);
                Avx.Store(pOut + i, result);
            }
        }
        else if (AdvSimd.IsSupported)
        {
            int vectorEnd = count - Vector128<float>.Count + 1;
            for (; i < vectorEnd; i += Vector128<float>.Count)
            {
                Vector128<float> va = AdvSimd.LoadVector128(pA + i);
                Vector128<float> vb = AdvSimd.LoadVector128(pB + i);
                Vector128<float> result = AdvSimd.Add(va, vb);
                AdvSimd.Store(pOut + i, result);
            }
        }

        for (; i < count; i++)
        {
            pOut[i] = pA[i] + pB[i];
        }
    }

    /// <summary>Performs element-wise multiplication: output = a * b. Uses AVX2/NEON vectorization with scalar fallback.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Mul(Tensor output, Tensor a, Tensor b)
    {
        int count = (int)output.ElementCount;
        float* pOut = (float*)output.DataPointer;
        float* pA = (float*)a.DataPointer;
        float* pB = (float*)b.DataPointer;

        int i = 0;

        if (Avx2.IsSupported)
        {
            int vectorEnd = count - Vector256<float>.Count + 1;
            for (; i < vectorEnd; i += Vector256<float>.Count)
            {
                Vector256<float> va = Avx.LoadVector256(pA + i);
                Vector256<float> vb = Avx.LoadVector256(pB + i);
                Vector256<float> result = Avx.Multiply(va, vb);
                Avx.Store(pOut + i, result);
            }
        }
        else if (AdvSimd.IsSupported)
        {
            int vectorEnd = count - Vector128<float>.Count + 1;
            for (; i < vectorEnd; i += Vector128<float>.Count)
            {
                Vector128<float> va = AdvSimd.LoadVector128(pA + i);
                Vector128<float> vb = AdvSimd.LoadVector128(pB + i);
                Vector128<float> result = AdvSimd.Multiply(va, vb);
                AdvSimd.Store(pOut + i, result);
            }
        }

        for (; i < count; i++)
        {
            pOut[i] = pA[i] * pB[i];
        }
    }

    /// <summary>Performs scalar multiplication: output = input * scalar. Uses AVX2/NEON vectorization with scalar fallback.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Scale(Tensor output, Tensor input, float scalar)
    {
        int count = (int)output.ElementCount;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;

        int i = 0;

        if (Avx2.IsSupported)
        {
            Vector256<float> vScalar = Vector256.Create(scalar);
            int vectorEnd = count - Vector256<float>.Count + 1;
            for (; i < vectorEnd; i += Vector256<float>.Count)
            {
                Vector256<float> va = Avx.LoadVector256(pIn + i);
                Vector256<float> result = Avx.Multiply(va, vScalar);
                Avx.Store(pOut + i, result);
            }
        }
        else if (AdvSimd.IsSupported)
        {
            Vector128<float> vScalar = Vector128.Create(scalar);
            int vectorEnd = count - Vector128<float>.Count + 1;
            for (; i < vectorEnd; i += Vector128<float>.Count)
            {
                Vector128<float> va = AdvSimd.LoadVector128(pIn + i);
                Vector128<float> result = AdvSimd.Multiply(va, vScalar);
                AdvSimd.Store(pOut + i, result);
            }
        }

        for (; i < count; i++)
        {
            pOut[i] = pIn[i] * scalar;
        }
    }

    /// <summary>Performs element-wise clamp: output = clamp(input, min, max). Uses AVX2/NEON vectorization with scalar fallback.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Clamp(Tensor output, Tensor input, float min, float max)
    {
        int count = (int)output.ElementCount;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;

        int i = 0;

        if (Avx2.IsSupported)
        {
            Vector256<float> vMin = Vector256.Create(min);
            Vector256<float> vMax = Vector256.Create(max);
            int vectorEnd = count - Vector256<float>.Count + 1;
            for (; i < vectorEnd; i += Vector256<float>.Count)
            {
                Vector256<float> va = Avx.LoadVector256(pIn + i);
                Vector256<float> result = Avx.Max(Avx.Min(va, vMax), vMin);
                Avx.Store(pOut + i, result);
            }
        }
        else if (AdvSimd.IsSupported)
        {
            Vector128<float> vMin = Vector128.Create(min);
            Vector128<float> vMax = Vector128.Create(max);
            int vectorEnd = count - Vector128<float>.Count + 1;
            for (; i < vectorEnd; i += Vector128<float>.Count)
            {
                Vector128<float> va = AdvSimd.LoadVector128(pIn + i);
                Vector128<float> result = AdvSimd.Max(AdvSimd.Min(va, vMax), vMin);
                AdvSimd.Store(pOut + i, result);
            }
        }

        for (; i < count; i++)
        {
            float val = pIn[i];
            pOut[i] = val < min ? min : (val > max ? max : val);
        }
    }

    /// <summary>Concatenates multiple tensors along the specified dimension. Currently optimized for dim 0 (batch concatenation) using block memory copies.</summary>
    public static unsafe void Concat(Tensor output, ReadOnlySpan<Tensor> inputs, int dim)
    {
        if (dim == 0)
        {
            float* pOut = (float*)output.DataPointer;
            long offset = 0;

            for (int t = 0; t < inputs.Length; t++)
            {
                float* pIn = (float*)inputs[t].DataPointer;
                long elementCount = inputs[t].ElementCount;
                long byteCount = elementCount * sizeof(float);

                Buffer.MemoryCopy(pIn, pOut + offset, byteCount, byteCount);
                offset += elementCount;
            }
        }
        else
        {
            // General case: iterate over the outer dimensions and copy slices.
            // For dim > 0, we compute the block sizes before and after the concat dimension.
            long outerSize = 1;
            for (int d = 0; d < dim; d++)
            {
                outerSize *= output.Shape[d];
            }

            long innerSize = 1;
            for (int d = dim + 1; d < output.Shape.Rank; d++)
            {
                innerSize *= output.Shape[d];
            }

            float* pOut = (float*)output.DataPointer;
            long outDimStride = output.Shape[dim] * innerSize;

            for (long outer = 0; outer < outerSize; outer++)
            {
                long outOffset = outer * outDimStride;
                long dimOffset = 0;

                for (int t = 0; t < inputs.Length; t++)
                {
                    long inputDimSize = inputs[t].Shape[dim];
                    long sliceSize = inputDimSize * innerSize;
                    float* pIn = (float*)inputs[t].DataPointer;

                    long inDimStride = inputDimSize * innerSize;
                    long inOffset = outer * inDimStride;

                    long byteCount = sliceSize * sizeof(float);
                    Buffer.MemoryCopy(pIn + inOffset, pOut + outOffset + dimOffset, byteCount, byteCount);

                    dimOffset += sliceSize;
                }
            }
        }
    }

    /// <summary>Splits a tensor into multiple output tensors along the specified dimension. Currently optimized for dim 0 (batch splitting) using block memory copies.</summary>
    public static unsafe void Split(ReadOnlySpan<Tensor> outputs, Tensor input, int dim)
    {
        if (dim == 0)
        {
            float* pIn = (float*)input.DataPointer;
            long offset = 0;

            for (int t = 0; t < outputs.Length; t++)
            {
                float* pOut = (float*)outputs[t].DataPointer;
                long elementCount = outputs[t].ElementCount;
                long byteCount = elementCount * sizeof(float);

                Buffer.MemoryCopy(pIn + offset, pOut, byteCount, byteCount);
                offset += elementCount;
            }
        }
        else
        {
            long outerSize = 1;
            for (int d = 0; d < dim; d++)
            {
                outerSize *= input.Shape[d];
            }

            long innerSize = 1;
            for (int d = dim + 1; d < input.Shape.Rank; d++)
            {
                innerSize *= input.Shape[d];
            }

            float* pIn = (float*)input.DataPointer;
            long inDimStride = input.Shape[dim] * innerSize;

            for (long outer = 0; outer < outerSize; outer++)
            {
                long inOffset = outer * inDimStride;
                long dimOffset = 0;

                for (int t = 0; t < outputs.Length; t++)
                {
                    long outputDimSize = outputs[t].Shape[dim];
                    long sliceSize = outputDimSize * innerSize;
                    float* pOut = (float*)outputs[t].DataPointer;

                    long outDimStride = outputDimSize * innerSize;
                    long outOffset = outer * outDimStride;

                    long byteCount = sliceSize * sizeof(float);
                    Buffer.MemoryCopy(pIn + inOffset + dimOffset, pOut + outOffset, byteCount, byteCount);

                    dimOffset += sliceSize;
                }
            }
        }
    }
}
