using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;

namespace HartsyInference.Cpu.Kernels;

/// <summary>Provides scaled dot-product attention CPU compute kernels. Supports multi-head attention with optional masking for F32 tensors.</summary>
public static class AttentionKernels
{
    /// <summary>Computes scaled dot-product attention: output = softmax(Q @ K^T * scale + mask) @ V. Query has shape [B, H, Sq, D], Key/Value have shape [B, H, Skv, D]. Supports different sequence lengths for Q and K/V (cross-attention). Uses O(Skv) auxiliary memory per row.</summary>
    /// <param name="output">Output tensor with shape [B, H, Sq, D].</param>
    /// <param name="query">Query tensor with shape [B, H, Sq, D].</param>
    /// <param name="key">Key tensor with shape [B, H, Skv, D].</param>
    /// <param name="value">Value tensor with shape [B, H, Skv, D].</param>
    /// <param name="mask">Optional additive mask tensor with shape broadcastable to [B, H, Sq, Skv], or null.</param>
    /// <param name="scale">Scale factor, typically 1/sqrt(D).</param>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void ScaledDotProductAttention(
        Tensor output,
        Tensor query,
        Tensor key,
        Tensor value,
        Tensor? mask,
        float scale)
    {
        long B = query.Shape[0];
        long H = query.Shape[1];
        long Sq = query.Shape[2];   // Query sequence length
        long D = query.Shape[3];
        long Skv = key.Shape[2];    // Key/Value sequence length (may differ from Sq for cross-attention)

        float* pQ = (float*)query.DataPointer;
        float* pK = (float*)key.DataPointer;
        float* pV = (float*)value.DataPointer;
        float* pOut = (float*)output.DataPointer;
        float* pMask = mask != null ? (float*)mask.DataPointer : null;

        long qHeadStride = Sq * D;
        long kvHeadStride = Skv * D;
        long qBatchStride = H * qHeadStride;
        long kvBatchStride = H * kvHeadStride;

        // Allocate a single row of attention scores: [Skv] floats
        float* attnRow = (float*)NativeMemory.Alloc((nuint)(Skv * sizeof(float)));

        try
        {
            for (long b = 0; b < B; b++)
            {
                for (long h = 0; h < H; h++)
                {
                    float* qHead = pQ + b * qBatchStride + h * qHeadStride;
                    float* kHead = pK + b * kvBatchStride + h * kvHeadStride;
                    float* vHead = pV + b * kvBatchStride + h * kvHeadStride;
                    float* oHead = pOut + b * qBatchStride + h * qHeadStride;

                    // Determine mask offset for this (b, h) pair. The mask rank decides how to broadcast:
                    //   rank 2  [Sq, Skv]            → shared across all B and H (the causal-mask case)
                    //   rank 3  [H|1, Sq, Skv]       → per-head, shared across B
                    //   rank 4  [B|1, H|1, Sq, Skv]  → per-batch / per-head
                    // The previous code assumed rank 4 unconditionally and, for a rank-2 mask, mis-read
                    // Shape[0]/Shape[1] as B/H — producing an out-of-bounds offset (h·Sq·Skv) for every head > 0.
                    // A mask whose query axis is 1 stores one [Skv] row broadcast over every query (a bias that
                    // depends only on the key); maskSq is that axis, so it is also the per-block row count.
                    long maskBhOffset = 0;
                    long maskSq = Sq;
                    if (pMask != null && mask != null)
                    {
                        long maskRank = mask.Shape.Rank;
                        maskSq = maskRank == 2 ? mask.Shape[0] : maskRank == 3 ? mask.Shape[1] : mask.Shape[2];
                        if (maskRank == 2)
                        {
                            maskBhOffset = 0;
                        }
                        else if (maskRank == 3)
                        {
                            long maskH = mask.Shape[0] > 1 ? h : 0;
                            maskBhOffset = maskH * maskSq * Skv;
                        }
                        else
                        {
                            long maskB = mask.Shape[0] > 1 ? b : 0;
                            long maskH = mask.Shape[1] > 1 ? h : 0;
                            maskBhOffset = maskB * mask.Shape[1] * maskSq * Skv + maskH * maskSq * Skv;
                        }
                    }

                    for (long qi = 0; qi < Sq; qi++)
                    {
                        float* qRow = qHead + qi * D;

                        // Step 1: Compute Q[qi] @ K^T -> attnRow[Skv], scaled
                        for (long ki = 0; ki < Skv; ki++)
                        {
                            float* kRow = kHead + ki * D;
                            float dot = VectorizedDot(qRow, kRow, (int)D);
                            attnRow[ki] = dot * scale;
                        }

                        // Step 2: Apply additive mask if present
                        if (pMask != null)
                        {
                            float* maskRow = pMask + maskBhOffset + (maskSq == 1 ? 0 : qi) * Skv;
                            for (long ki = 0; ki < Skv; ki++)
                            {
                                attnRow[ki] += maskRow[ki];
                            }
                        }

                        // Step 3: Softmax over attnRow
                        SoftmaxInPlace(attnRow, (int)Skv);

                        // Step 4: attnRow @ V -> output row
                        float* oRow = oHead + qi * D;

                        // Zero the output row
                        NativeMemory.Clear(oRow, (nuint)(D * sizeof(float)));

                        for (long vi = 0; vi < Skv; vi++)
                        {
                            float attnVal = attnRow[vi];
                            float* vRow = vHead + vi * D;

                            int d = 0;

                            if (Avx2.IsSupported)
                            {
                                Vector256<float> vAttn = Vector256.Create(attnVal);
                                int vectorEnd = (int)D - Vector256<float>.Count + 1;

                                for (; d < vectorEnd; d += Vector256<float>.Count)
                                {
                                    Vector256<float> vV = Avx.LoadVector256(vRow + d);
                                    Vector256<float> vO = Avx.LoadVector256(oRow + d);
                                    Vector256<float> result = Fma.IsSupported
                                        ? Fma.MultiplyAdd(vAttn, vV, vO)
                                        : Avx.Add(vO, Avx.Multiply(vAttn, vV));
                                    Avx.Store(oRow + d, result);
                                }
                            }
                            else if (AdvSimd.IsSupported)
                            {
                                Vector128<float> vAttn = Vector128.Create(attnVal);
                                int vectorEnd = (int)D - Vector128<float>.Count + 1;

                                for (; d < vectorEnd; d += Vector128<float>.Count)
                                {
                                    Vector128<float> vV = AdvSimd.LoadVector128(vRow + d);
                                    Vector128<float> vO = AdvSimd.LoadVector128(oRow + d);
                                    Vector128<float> result = AdvSimd.Add(vO, AdvSimd.Multiply(vAttn, vV));
                                    AdvSimd.Store(oRow + d, result);
                                }
                            }

                            for (; d < (int)D; d++)
                            {
                                oRow[d] += attnVal * vRow[d];
                            }
                        }
                    }
                }
            }
        }
        finally
        {
            NativeMemory.Free(attnRow);
        }
    }

    /// <summary>Computes the dot product of two float vectors using SIMD acceleration.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float VectorizedDot(float* a, float* b, int count)
    {
        float result = 0.0f;
        int i = 0;

        if (Avx2.IsSupported)
        {
            Vector256<float> vSum = Vector256<float>.Zero;
            int vectorEnd = count - Vector256<float>.Count + 1;

            for (; i < vectorEnd; i += Vector256<float>.Count)
            {
                Vector256<float> va = Avx.LoadVector256(a + i);
                Vector256<float> vb = Avx.LoadVector256(b + i);
                vSum = Fma.IsSupported
                    ? Fma.MultiplyAdd(va, vb, vSum)
                    : Avx.Add(vSum, Avx.Multiply(va, vb));
            }

            result += SimdDispatch.HorizontalSum(vSum);
        }
        else if (AdvSimd.IsSupported)
        {
            Vector128<float> vSum = Vector128<float>.Zero;
            int vectorEnd = count - Vector128<float>.Count + 1;

            for (; i < vectorEnd; i += Vector128<float>.Count)
            {
                Vector128<float> va = AdvSimd.LoadVector128(a + i);
                Vector128<float> vb = AdvSimd.LoadVector128(b + i);
                vSum = AdvSimd.Add(vSum, AdvSimd.Multiply(va, vb));
            }

            float* tmp = stackalloc float[Vector128<float>.Count];
            AdvSimd.Store(tmp, vSum);
            for (int j = 0; j < Vector128<float>.Count; j++)
            {
                result += tmp[j];
            }
        }

        for (; i < count; i++)
        {
            result += a[i] * b[i];
        }

        return result;
    }

    /// <summary>Applies numerically stable softmax in-place over a float buffer.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void SoftmaxInPlace(float* data, int count)
    {
        // Find max for numerical stability
        float maxVal = float.NegativeInfinity;
        for (int i = 0; i < count; i++)
        {
            if (data[i] > maxVal)
            {
                maxVal = data[i];
            }
        }

        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int i = 0; i < count; i++)
        {
            float expVal = MathF.Exp(data[i] - maxVal);
            data[i] = expVal;
            sum += expVal;
        }

        // Normalize
        float invSum = 1.0f / sum;
        for (int i = 0; i < count; i++)
        {
            data[i] *= invSum;
        }
    }
}
