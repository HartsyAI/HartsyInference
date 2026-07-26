using HartsyInference.Core.Tensors;

namespace HartsyInference.Cpu.Kernels;

/// <summary>Provides matrix multiplication CPU compute kernels with SIMD-accelerated inner loops. Supports 2D GEMM and 3D batched matrix multiplication for F32 tensors.</summary>
public static class MatMulKernels
{
    private const int TileSize = 32;

    /// <summary>These kernels read tensor data as <c>float*</c>. A non-F32 tensor (e.g. bf16/f16 weights loaded
    /// straight from a checkpoint) has half the bytes per element, so casting its pointer to <c>float*</c> reads
    /// off the end of the allocation and corrupts the heap. Fail loudly instead: the CPU backend is F32-only, so
    /// callers must cast weights to F32 (e.g. <see cref="Tensor.CastTo"/>) before dispatching here.</summary>
    private static void RequireF32(Tensor output, Tensor a, Tensor b, Tensor? bias, string op)
    {
        if (output.DType != DType.F32 || a.DType != DType.F32 || b.DType != DType.F32 ||
            (bias is not null && bias.DType != DType.F32))
            throw new ArgumentException(
                $"{op}: the CPU backend is F32-only (got output={output.DType}, a={a.DType}, b={b.DType}" +
                $"{(bias is not null ? $", bias={bias.DType}" : "")}). Cast non-F32 weights to F32 first.");
    }

    /// <summary>Performs 2D general matrix multiplication: output[M,N] = a[M,K] @ b[K,N]. Uses a tiled approach with AVX2 vectorization for the inner accumulation loop. Both input matrices are expected in row-major layout.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void MatMul(Tensor output, Tensor a, Tensor b)
    {
        RequireF32(output, a, b, null, nameof(MatMul));
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

    /// <summary>Linear layer: output[M,N] = input[M,K] × weight^T[K,N] + bias[N]. Weight is [N, K] row-major (PyTorch convention). Uses tiled GEMM with AVX2 for the matmul, then adds bias.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void LinearTransB(Tensor output, Tensor input, Tensor weight, Tensor? bias)
    {
        // Tolerate a non-F32 weight by casting it up for the call (fp16→f32 is exact). The MusicGen/AudioGen
        // decoder + T5 now keep their big projection weights in native fp16 to halve host RAM; the GPU GEMM casts
        // them itself, and on CPU we cast here so those models still run on the F32-only CPU kernels. output/input/
        // bias must already be F32.
        Tensor? weightF32 = weight.DType != DType.F32 ? weight.CastTo(DType.F32) : null;
        Tensor w = weightF32 ?? weight;
        try
        {
        RequireF32(output, input, w, bias, nameof(LinearTransB));
        int N = (int)w.Shape[0]; // outDim
        int K = (int)w.Shape[1]; // inDim
        int M = (int)(input.ElementCount / K); // batch*seqLen

        // Fail fast on a weight/output mismatch: the tiled loop below writes M·N floats derived from the
        // WEIGHT's outDim, so an undersized output buffer is silent native heap corruption, not an exception.
        if (output.ElementCount != (long)M * N)
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException(
                $"LinearTransB shape mismatch: output has {output.ElementCount} elements but input rows {M} × weight outDim {N} = {(long)M * N} (weight {w.Shape}, input {input.Shape}).");

        float* pIn = (float*)input.DataPointer;
        float* pW = (float*)w.DataPointer;
        float* pOut = (float*)output.DataPointer;

        // Zero the output buffer
        NativeMemory.Clear(pOut, (nuint)(M * N * sizeof(float)));

        // Tiled GEMM: C[M,N] = A[M,K] × B^T[K,N] where B is stored as [N, K]
        // Access pattern: for each (i, j), sum_k A[i,k] * B[j,k]
        for (int ii = 0; ii < M; ii += TileSize)
        {
            int iEnd = Math.Min(ii + TileSize, M);

            for (int jj = 0; jj < N; jj += TileSize)
            {
                int jEnd = Math.Min(jj + TileSize, N);

                for (int kk = 0; kk < K; kk += TileSize)
                {
                    int kEnd = Math.Min(kk + TileSize, K);

                    for (int i = ii; i < iEnd; i++)
                    {
                        float* rowA = pIn + i * K;
                        float* rowOut = pOut + i * N;

                        for (int j = jj; j < jEnd; j++)
                        {
                            float* rowW = pW + j * K;
                            float sum = 0f;

                            int k = kk;
                            if (Avx2.IsSupported)
                            {
                                Vector256<float> vSum = Vector256<float>.Zero;
                                int vectorEnd = kEnd - Vector256<float>.Count + 1;
                                for (; k < vectorEnd; k += Vector256<float>.Count)
                                {
                                    Vector256<float> vA = Avx.LoadVector256(rowA + k);
                                    Vector256<float> vW = Avx.LoadVector256(rowW + k);
                                    vSum = Fma.IsSupported
                                        ? Fma.MultiplyAdd(vA, vW, vSum)
                                        : Avx.Add(vSum, Avx.Multiply(vA, vW));
                                }
                                // Horizontal sum
                                Vector128<float> hi = Avx.ExtractVector128(vSum, 1);
                                Vector128<float> lo = vSum.GetLower();
                                Vector128<float> v4 = Sse.Add(lo, hi);
                                Vector128<float> v2 = Sse.Add(v4, Sse.MoveHighToLow(v4, v4));
                                Vector128<float> v1 = Sse.AddScalar(v2, Sse.Shuffle(v2, v2, 1));
                                sum = v1.ToScalar();
                            }

                            for (; k < kEnd; k++)
                            {
                                sum += rowA[k] * rowW[k];
                            }

                            rowOut[j] += sum;
                        }
                    }
                }
            }
        }

        if (bias is not null)
        {
            float* bPtr = (float*)bias.DataPointer;
            for (int m = 0; m < M; m++)
            {
                int rowOffset = m * N;
                for (int n = 0; n < N; n++)
                {
                    pOut[rowOffset + n] += bPtr[n];
                }
            }
        }
        }
        finally
        {
            weightF32?.Dispose();
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
