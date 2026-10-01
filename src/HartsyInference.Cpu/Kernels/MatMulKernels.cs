using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cpu.Kernels;

/// <summary>Provides matrix multiplication CPU compute kernels with SIMD-accelerated inner loops. Supports 2D GEMM and 3D batched matrix multiplication for F32 tensors.</summary>
public static class MatMulKernels
{
    private const int TileSize = 32;

    // Held in a field so an inline call allocates no delegate by construction. The C# 11+ compiler also caches a
    // static method-group conversion, but the language permits that rather than requiring it.
    // VoiceFrontendAllocationTests is what checks the whole path allocates nothing.
    private static readonly unsafe Action<int, LinearTiles> LinearTileBody = LinearTile;

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

        NativeMemory.Clear(pOut, (nuint)(M * N * sizeof(float)));

        // Tiled GEMM, fanned out over (row tile x column tile). Every (i, j) belongs to exactly one such pair,
        // so the split is race-free; the K tiles stay inside a task because they accumulate into the same
        // element. Splitting on both axes rather than on rows alone is what keeps a single-row call — an LLM
        // decoding one token, or a 1x1 convolution routed here by Conv2D — from running on one core.
        int iTiles = (M + TileSize - 1) / TileSize;
        int jTiles = (N + TileSize - 1) / TileSize;
        CpuParallel.For(iTiles * jTiles, (long)M * N * K, tile =>
        {
            int ii = tile / jTiles * TileSize;
            int iEnd = Math.Min(ii + TileSize, M);
            int jj = tile % jTiles * TileSize;
            int jEnd = Math.Min(jj + TileSize, N);

            for (int kk = 0; kk < K; kk += TileSize)
            {
                int kEnd = Math.Min(kk + TileSize, K);

                {
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
                                    Vector256<float> result = Fma.IsSupported ? Fma.MultiplyAdd(vA, vB, vOut)
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
        });
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

            NativeMemory.Clear(pOut, (nuint)(M * N * sizeof(float)));

            // Tiled GEMM: C[M,N] = A[M,K] × B^T[K,N] where B is stored as [N, K]
            // Access pattern: for each (i, j), sum_k A[i,k] * B[j,k]
            //
            // Fanned out over (row tile x column tile) for the reason spelled out in MatMul: token-by-token decode
            // calls this with M = 1, so a row-only split would leave every projection and the vocabulary
            // projection running on a single core.
            int iTiles = (M + TileSize - 1) / TileSize;
            int jTiles = (N + TileSize - 1) / TileSize;
            CpuParallel.For(iTiles * jTiles, (long)M * N * K, new LinearTiles(pIn, pW, pOut, M, N, K, jTiles),
                LinearTileBody);

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

    /// <summary>One (row tile, column tile) of <see cref="LinearTransB"/>. A static method over passed-in state
    /// rather than a capturing lambda, so its delegate is built once, in a static field, and an inline call allocates
    /// nothing.</summary>
    private static unsafe void LinearTile(int tile, LinearTiles s)
    {
        float* pIn = s.Input;
        float* pW = s.Weight;
        float* pOut = s.Output;
        int M = s.M, N = s.N, K = s.K, jTiles = s.JTiles;
        int ii = tile / jTiles * TileSize;
        int iEnd = Math.Min(ii + TileSize, M);
        int jj = tile % jTiles * TileSize;
        int jEnd = Math.Min(jj + TileSize, N);

        for (int kk = 0; kk < K; kk += TileSize)
        {
            int kEnd = Math.Min(kk + TileSize, K);

            for (int i = ii; i < iEnd; i++)
            {
                float* rowA = pIn + i * K;
                float* rowOut = pOut + i * N;

                int j = jj;
                if (Avx2.IsSupported)
                {
                    // Four weight rows at a time. One row's sum is a dependent chain that leaves the core
                    // mostly idle; four independent ones overlap. Each row still runs exactly the
                    // single-row sequence below — same products, horizontal sum and scalar tail, in the
                    // same order — so the output is bit-identical, only faster.
                    for (; j + 4 <= jEnd; j += 4)
                    {
                        float* w0 = pW + j * K;
                        float* w1 = w0 + K;
                        float* w2 = w1 + K;
                        float* w3 = w2 + K;
                        Vector256<float> s0 = Vector256<float>.Zero, s1 = s0, s2 = s0, s3 = s0;
                        int k = kk;
                        int vectorEnd = kEnd - Vector256<float>.Count + 1;
                        for (; k < vectorEnd; k += Vector256<float>.Count)
                        {
                            Vector256<float> vA = Avx.LoadVector256(rowA + k);
                            s0 = MultiplyAccumulate(vA, Avx.LoadVector256(w0 + k), s0);
                            s1 = MultiplyAccumulate(vA, Avx.LoadVector256(w1 + k), s1);
                            s2 = MultiplyAccumulate(vA, Avx.LoadVector256(w2 + k), s2);
                            s3 = MultiplyAccumulate(vA, Avx.LoadVector256(w3 + k), s3);
                        }
                        float sum0 = HorizontalSum(s0), sum1 = HorizontalSum(s1);
                        float sum2 = HorizontalSum(s2), sum3 = HorizontalSum(s3);
                        for (; k < kEnd; k++)
                        {
                            float a = rowA[k];
                            sum0 += a * w0[k];
                            sum1 += a * w1[k];
                            sum2 += a * w2[k];
                            sum3 += a * w3[k];
                        }
                        rowOut[j] += sum0;
                        rowOut[j + 1] += sum1;
                        rowOut[j + 2] += sum2;
                        rowOut[j + 3] += sum3;
                    }
                }

                for (; j < jEnd; j++)
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
                            vSum = MultiplyAccumulate(vA, vW, vSum);
                        }
                        sum = HorizontalSum(vSum);
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

    /// <summary>What <see cref="LinearTile"/> needs from the call that fans it out.</summary>
    private readonly unsafe struct LinearTiles(float* input, float* weight, float* output, int m, int n, int k,
        int jTiles)
    {
        public readonly float* Input = input;
        public readonly float* Weight = weight;
        public readonly float* Output = output;
        public readonly int M = m;
        public readonly int N = n;
        public readonly int K = k;
        public readonly int JTiles = jTiles;
    }

    /// <summary><c>acc + a·b</c>, fused where the CPU has FMA. Shared by every <see cref="LinearTransB"/> row path so
    /// the single-row and four-row loops round identically.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> MultiplyAccumulate(Vector256<float> a, Vector256<float> b, Vector256<float> acc) =>
        Fma.IsSupported ? Fma.MultiplyAdd(a, b, acc) : Avx.Add(acc, Avx.Multiply(a, b));

    /// <summary>Sums the eight lanes in <see cref="LinearTransB"/>'s fixed order: halves, then pairs, then the last two.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalSum(Vector256<float> v)
    {
        Vector128<float> v4 = Sse.Add(v.GetLower(), Avx.ExtractVector128(v, 1));
        Vector128<float> v2 = Sse.Add(v4, Sse.MoveHighToLow(v4, v4));
        return Sse.AddScalar(v2, Sse.Shuffle(v2, v2, 1)).ToScalar();
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
