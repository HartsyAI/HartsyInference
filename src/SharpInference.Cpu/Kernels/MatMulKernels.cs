using SharpInference.Core.Tensors;

namespace SharpInference.Cpu.Kernels;

/// <summary>Provides matrix multiplication CPU compute kernels with SIMD-accelerated inner loops. Supports 2D GEMM and 3D batched matrix multiplication for F32 tensors.</summary>
public static class MatMulKernels
{
    private const int TileSize = 32;

    /// <summary>Performs 2D general matrix multiplication: output[M,N] = a[M,K] @ b[K,N]. Uses a tiled approach with AVX2 vectorization for the inner accumulation loop. Both input matrices are expected in row-major layout.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void MatMul(Tensor output, Tensor a, Tensor b)
    {
        int M = (int)a.Shape[0];
        int K = (int)a.Shape[1];
        int N = (int)b.Shape[1];

        float* pA = (float*)a.DataPointer;
        float* pB = (float*)b.DataPointer;
        float* pOut = (float*)output.DataPointer;

        // Zero the output buffer
        NativeMemory.Clear(pOut, (nuint)(M * N * sizeof(float)));

        // Tiled GEMM: iterate over tiles to improve cache locality
        for (int ii = 0; ii < M; ii += TileSize)
        {
            int iEnd = Math.Min(ii + TileSize, M);

            for (int kk = 0; kk < K; kk += TileSize)
            {
                int kEnd = Math.Min(kk + TileSize, K);

                for (int jj = 0; jj < N; jj += TileSize)
                {
                    int jEnd = Math.Min(jj + TileSize, N);

                    for (int i = ii; i < iEnd; i++)
                    {
                        float* rowA = pA + i * K;
                        float* rowOut = pOut + i * N;

                        for (int k = kk; k < kEnd; k++)
                        {
                            float aVal = rowA[k];
                            float* rowB = pB + k * N;

                            int j = jj;

                            if (Avx2.IsSupported)
                            {
                                Vector256<float> vA = Vector256.Create(aVal);
                                int vectorEnd = jEnd - Vector256<float>.Count + 1;

                                for (; j < vectorEnd; j += Vector256<float>.Count)
                                {
                                    Vector256<float> vB = Avx.LoadVector256(rowB + j);
                                    Vector256<float> vOut = Avx.LoadVector256(rowOut + j);
                                    Vector256<float> result = Fma.IsSupported
                                        ? Fma.MultiplyAdd(vA, vB, vOut)
                                        : Avx.Add(vOut, Avx.Multiply(vA, vB));
                                    Avx.Store(rowOut + j, result);
                                }
                            }
                            else if (AdvSimd.IsSupported)
                            {
                                Vector128<float> vA = Vector128.Create(aVal);
                                int vectorEnd = jEnd - Vector128<float>.Count + 1;

                                for (; j < vectorEnd; j += Vector128<float>.Count)
                                {
                                    Vector128<float> vB = AdvSimd.LoadVector128(rowB + j);
                                    Vector128<float> vOut = AdvSimd.LoadVector128(rowOut + j);
                                    Vector128<float> result = AdvSimd.Add(vOut, AdvSimd.Multiply(vA, vB));
                                    AdvSimd.Store(rowOut + j, result);
                                }
                            }

                            for (; j < jEnd; j++)
                            {
                                rowOut[j] += aVal * rowB[j];
                            }
                        }
                    }
                }
            }
        }
    }

    /// <summary>Performs 3D batched matrix multiplication: output[B,M,N] = a[B,M,K] @ b[K,N] or b[B,K,N]. When b is 2D, it is broadcast across the batch dimension. Iterates over the batch dimension and delegates each slice to <see cref="MatMul"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void BatchedMatMul(Tensor output, Tensor a, Tensor b)
    {
        long batchSize = a.Shape[0];
        long M = a.Shape[1];
        long K = a.Shape[2];

        // Handle 2D right operand: b is [K, N] broadcast across batch
        bool bIs2D = b.Shape.Rank == 2;
        long N = bIs2D ? b.Shape[1] : b.Shape[2];

        long aSliceSize = M * K;
        long bSliceSize = bIs2D ? 0 : K * N; // 0 = reuse same pointer for all batches
        long outSliceSize = M * N;

        float* pA = (float*)a.DataPointer;
        float* pB = (float*)b.DataPointer;
        float* pOut = (float*)output.DataPointer;

        for (long batch = 0; batch < batchSize; batch++)
        {
            // Create views into the batch slices using pointer arithmetic
            TensorShape sliceShapeA = new TensorShape(M, K);
            TensorShape sliceShapeB = new TensorShape(K, N);
            TensorShape sliceShapeOut = new TensorShape(M, N);

            Tensor sliceA = new Tensor(pA + batch * aSliceSize, sliceShapeA, DType.F32);
            Tensor sliceB = new Tensor(pB + batch * bSliceSize, sliceShapeB, DType.F32);
            Tensor sliceOut = new Tensor(pOut + batch * outSliceSize, sliceShapeOut, DType.F32);

            MatMul(sliceOut, sliceA, sliceB);
        }
    }
}
