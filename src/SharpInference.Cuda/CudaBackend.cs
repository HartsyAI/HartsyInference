using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Cuda;

/// <summary>CUDA GPU backend implementing IBackend. Routes operations to cuBLAS SGEMM for matmul and PTX kernels for element-wise/normalization ops.</summary>
public sealed class CudaBackend : IBackend
{
    private readonly CudaContext _context;
    private readonly CudaStream _stream;
    private readonly CudaKernels? _kernels;
    private nint _cublasHandle;
    private bool _disposed;

    /// <summary>The device this backend targets.</summary>
    public DeviceKind Device { get; }

    /// <summary>Capabilities of this CUDA backend.</summary>
    public BackendCapabilities Capabilities { get; }

    /// <summary>The CUDA context used by this backend.</summary>
    public CudaContext Context => _context;

    /// <summary>The default compute stream.</summary>
    public CudaStream Stream => _stream;

    /// <summary>The cuBLAS handle for GEMM operations.</summary>
    public nint CublasHandle => _cublasHandle;

    /// <summary>Creates a CUDA backend for the specified device ordinal. If ptxDir is provided, loads PTX kernels from that directory.</summary>
    public CudaBackend(int deviceOrdinal = 0, string? ptxDir = null)
    {
        _context = new CudaContext(deviceOrdinal);
        _stream = new CudaStream(nonBlocking: true);
        Device = DeviceKind.Cuda(deviceOrdinal);

        // Initialize cuBLAS
        CublasApi.cublasCreate(out _cublasHandle).ThrowOnCublasError();
        CublasApi.cublasSetStream(_cublasHandle, _stream.Handle).ThrowOnCublasError();

        // Load PTX kernels if directory provided
        if (ptxDir != null && Directory.Exists(ptxDir))
        {
            _kernels = new CudaKernels(ptxDir);
        }

        Capabilities = new BackendCapabilities
        {
            Name = $"CUDA ({_context.DeviceName}, SM {_context.ComputeCapabilityMajor}.{_context.ComputeCapabilityMinor})",
            SupportsF32 = true,
            SupportsF16 = true,
            SupportsBF16 = _context.ComputeCapabilityMajor >= 8,
            SupportsQuantized = true,
            SupportsConv2D = true,
            SupportsSdpa = true,
            SupportsFft = false,
            MaxRank = 6,
        };
    }

    // ── Linear Algebra ──────────────────────────────────────────────────

    /// <summary>Matrix multiply via cuBLAS SGEMM: output = a @ b</summary>
    public unsafe void MatMul(Tensor output, Tensor a, Tensor b)
    {
        int M = (int)a.Shape[0];
        int K = (int)a.Shape[1];
        int N = (int)b.Shape[1];

        float alpha = 1.0f;
        float beta = 0.0f;

        ulong pA = DevPtr(a);
        ulong pB = DevPtr(b);
        ulong pC = DevPtr(output);

        CublasApi.cublasSgemm(
            _cublasHandle,
            CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
            N, M, K,
            &alpha,
            pB, N,
            pA, K,
            &beta,
            pC, N).ThrowOnCublasError();
    }

    /// <summary>Batched matrix multiply via cuBLAS strided batched GEMM.</summary>
    public unsafe void BatchedMatMul(Tensor output, Tensor a, Tensor b)
    {
        long batchSize = a.Shape[0];
        int M = (int)a.Shape[1];
        int K = (int)a.Shape[2];

        bool bIs2D = b.Shape.Rank == 2;
        int N = bIs2D ? (int)b.Shape[1] : (int)b.Shape[2];

        long strideA = M * K;
        long strideB = bIs2D ? 0 : K * N;
        long strideC = M * N;

        float alpha = 1.0f;
        float beta = 0.0f;

        ulong pA = DevPtr(a);
        ulong pB = DevPtr(b);
        ulong pC = DevPtr(output);

        CublasApi.cublasGemmStridedBatchedEx(
            _cublasHandle,
            CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
            N, M, K,
            &alpha,
            pB, CublasApi.CUDA_R_32F, N, strideB,
            pA, CublasApi.CUDA_R_32F, K, strideA,
            &beta,
            pC, CublasApi.CUDA_R_32F, N, strideC,
            (int)batchSize,
            CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
    }

    // ── Convolution ─────────────────────────────────────────────────────

    /// <summary>2D convolution via im2col + cuBLAS SGEMM. Supports arbitrary stride, padding, and kernel sizes.</summary>
    public unsafe void Conv2D(Tensor output, Tensor input, Tensor weight, Tensor? bias, int strideH, int strideW, int padH, int padW)
    {
        EnsureKernels();

        int batch = (int)input.Shape[0];
        int inCh = (int)input.Shape[1];
        int inH = (int)input.Shape[2];
        int inW = (int)input.Shape[3];

        int outCh = (int)weight.Shape[0];
        int kH = (int)weight.Shape[2];
        int kW = (int)weight.Shape[3];

        int outH = (inH + 2 * padH - kH) / strideH + 1;
        int outW = (inW + 2 * padW - kW) / strideW + 1;

        int colRows = inCh * kH * kW;
        int colCols = outH * outW;

        // 1x1 optimization: skip im2col when kernel is 1x1 with stride 1 and no padding
        bool is1x1 = kH == 1 && kW == 1 && strideH == 1 && strideW == 1 && padH == 0 && padW == 0;

        // Allocate column buffer for im2col (reused per batch)
        ulong colBuf = 0;
        if (!is1x1)
        {
            colBuf = CudaMemory.Allocate((nuint)(colRows * colCols * sizeof(float)));
        }

        ulong pWeight = DevPtr(weight);
        ulong pOutput = DevPtr(output);
        ulong pInput = DevPtr(input);

        float alpha = 1.0f;
        float beta = 0.0f;

        try
        {
            for (int b = 0; b < batch; b++)
            {
                int inputBatchOffset = b * inCh; // batch offset in channels for im2col

                ulong colPtr;
                if (is1x1)
                {
                    // For 1x1 conv, input IS the column matrix (just reshape)
                    colPtr = pInput + (ulong)(b * inCh * inH * inW * sizeof(float));
                }
                else
                {
                    // Im2Col: extract patches → col[colRows, colCols]
                    _kernels!.LaunchIm2Col(
                        colBuf, pInput,
                        inCh, inH, inW, kH, kW,
                        padH, padW, strideH, strideW,
                        outH, outW, inputBatchOffset,
                        _stream.Handle);
                    colPtr = colBuf;
                }

                // GEMM: weight[outCh, colRows] @ col[colRows, colCols] → out[outCh, colCols]
                // In cuBLAS column-major: C = B^T @ A^T
                ulong outBatchPtr = pOutput + (ulong)(b * outCh * outH * outW * sizeof(float));

                CublasApi.cublasSgemm(
                    _cublasHandle,
                    CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
                    colCols, outCh, colRows,
                    &alpha,
                    colPtr, colCols,
                    pWeight, colRows,
                    &beta,
                    outBatchPtr, colCols).ThrowOnCublasError();
            }

            // Add bias if present
            if (bias is not null)
            {
                int totalElements = batch * outCh * outH * outW;
                _kernels!.LaunchBiasAdd(
                    pOutput, DevPtr(bias),
                    outCh, outH * outW, totalElements,
                    _stream.Handle);
            }
        }
        finally
        {
            if (colBuf != 0)
            {
                CudaMemory.Free(colBuf);
            }
        }
    }

    // ── Normalization ───────────────────────────────────────────────────

    public void GroupNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps)
    {
        EnsureKernels();

        int batch = (int)input.Shape[0];
        int channels = (int)input.Shape[1];
        int spatial = 1;
        for (int d = 2; d < input.Shape.Rank; d++)
        {
            spatial *= (int)input.Shape[d];
        }

        _kernels!.LaunchGroupNorm(
            DevPtr(output), DevPtr(input),
            DevPtr(weight), DevPtr(bias),
            batch, channels, spatial, groups, eps,
            _stream.Handle);
    }

    public void LayerNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, float eps)
    {
        EnsureKernels();

        int normDim = (int)input.Shape[input.Shape.Rank - 1];
        int totalRows = (int)(input.ElementCount / normDim);

        _kernels!.LaunchLayerNorm(
            DevPtr(output), DevPtr(input),
            DevPtr(weight), DevPtr(bias),
            normDim, totalRows, eps,
            _stream.Handle);
    }

    public void RmsNorm(Tensor output, Tensor input, Tensor weight, float eps)
    {
        throw new NotImplementedException("CUDA RmsNorm not yet implemented");
    }

    // ── Attention ───────────────────────────────────────────────────────

    /// <summary>Scaled dot-product attention via cuBLAS batched GEMM: softmax(Q @ K^T * scale) @ V. Mask support via elementwise kernel.</summary>
    public unsafe void ScaledDotProductAttention(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale)
    {
        EnsureKernels();

        long B = query.Shape[0];
        long H = query.Shape[1];
        long Sq = query.Shape[2];
        long D = query.Shape[3];
        long Skv = key.Shape[2];

        long totalHeads = B * H;

        // Step 1: QK^T → scores [B*H, Sq, Skv]
        // Q is [B, H, Sq, D] contiguous → treat as [B*H, Sq, D]
        // K is [B, H, Skv, D] contiguous → need K^T [B*H, D, Skv]
        // scores = Q @ K^T = [B*H, Sq, Skv]

        nuint scoresBytes = (nuint)(totalHeads * Sq * Skv * sizeof(float));
        ulong scoresBuf = CudaMemory.Allocate(scoresBytes);

        try
        {
            ulong pQ = DevPtr(query);
            ulong pK = DevPtr(key);
            ulong pV = DevPtr(value);
            ulong pOut = DevPtr(output);

            float alpha = scale;
            float beta = 0.0f;

            // Batched GEMM: scores[bh] = Q[bh] @ K[bh]^T
            // Q[bh] is [Sq, D], K[bh] is [Skv, D], K^T is [D, Skv]
            // In cuBLAS col-major: C = op(A) @ op(B)
            // We need: scores = Q @ K^T → cuBLAS(K, Q^T, scores) with transA=T, transB=N
            // Actually: row-major C[M,N] = A[M,K] @ B[K,N]
            // → cuBLAS col-major: C = B^T @ A^T
            // Here: A = Q[Sq, D], B = K^T[D, Skv], C = scores[Sq, Skv]
            // B^T = K[Skv, D], A^T = Q^T[D, Sq]
            // cuBLAS: C[Skv, Sq] = K[Skv, D] @ Q^T[D, Sq]
            // But we want row-major scores[Sq, Skv]
            // Trick: cublasSgemm(N, T, Skv, Sq, D, alpha, K, Skv, Q, D??)
            // Wait, let me think again with the standard row-major trick:
            // Row-major: C = A @ B where A[M,K], B[K,N], C[M,N]
            // cuBLAS: we compute C^T = B^T @ A^T, all in column-major
            // So: cublasSgemm(N, N, N, M, K, alpha, B, N, A, K, beta, C, N)
            //
            // For QK^T: A=Q[Sq,D], B=K^T[D,Skv]
            // M=Sq, K=D, N=Skv
            // But B is K transposed. K is stored [Skv,D]. K^T[D,Skv] is not contiguous.
            // Instead: use cublas transpose flags.
            // cublasSgemm(transa=T, transb=N, N=Skv, M=Sq, K=D, alpha, K, D_stride_for_K, Q, D_stride_for_Q, ...)
            //
            // K is [Skv, D] in row-major → col-major: [D, Skv], ld=D?? No.
            // Row-major K[Skv,D]: element at (i,j) is at offset i*D+j. Leading dimension of col-major view = D? No, = Skv...
            //
            // Actually, for row-major data:
            // K stored as [Skv, D] means K_colmajor is a [D, Skv] matrix with ld = D? No.
            // A row-major [R,C] matrix, when viewed as col-major, has ld=C.
            // K[Skv, D] row-major → col-major K^T[D, Skv] with ld=D.
            // Wait no. Row-major [Skv, D] storage is the same bits as col-major [D, Skv] with ld=D.
            // So if I want K^T in col-major layout, I can use K's storage directly!
            // K^T is [D, Skv] in row-major = [Skv, D] in col-major with ld=Skv.
            // Hmm, let me just use the transpose flags.
            //
            // Standard approach:
            // scores = Q @ K^T, both row-major.
            // cuBLAS: scores^T = (K^T)^T @ Q^T = K @ Q^T
            // cublasSgemm(N, T, Skv, Sq, D, alpha, K_ptr, Skv_or_D?, Q_ptr, ?, beta, scores, Skv)
            //
            // K is row-major [Skv, D]. In cuBLAS col-major view, ld = D (the number of columns).
            // Wait, cuBLAS ld is the leading dimension of the column-major matrix.
            // Row-major [rows, cols] → col-major view: ld = cols.
            //
            // K row-major [Skv, D] → col-major ld = D
            // Q row-major [Sq, D] → col-major ld = D
            // scores row-major [Sq, Skv] → col-major ld = Skv
            //
            // We want scores = Q @ K^T in row-major.
            // Equivalently in col-major: scores_cm = K_cm @ Q_cm^T
            //   where K_cm is the col-major view of K, Q_cm is the col-major view of Q
            //
            // K_cm has ld = D, shape in col-major is [D, Skv] (since row-major [Skv, D])
            //   We want K_cm without transpose: it's a [D, Skv] col-major matrix
            // Q_cm has ld = D, shape in col-major is [D, Sq]
            //   We want Q_cm^T: transpose of [D, Sq] = [Sq, D]
            //
            // cublasSgemm(transa=N, transb=T, m=Skv, n=Sq, k=D,
            //             alpha, K_ptr, lda=D, Q_ptr, ldb=D, beta, scores_ptr, ldc=Skv)
            //
            // Wait, m should be rows of op(A) and cols of C.
            // op(A) = A (no transpose) = K_cm [D, Skv] → rows=D? No...
            //
            // OK let me just use the simple approach. For row-major matrices:
            // C[M,N] = A[M,K] @ B[K,N]
            // cuBLAS: cublasSgemm(N, N, N, M, K, alpha, B_ptr, N, A_ptr, K, beta, C_ptr, N)
            //
            // For C = Q @ K^T: M=Sq, K_dim=D, N=Skv, B=K^T
            // But K^T is not stored contiguously. K is [Skv, D].
            // K^T[D, Skv]: element (i,j) of K^T = K[j,i] = stored at j*D+i.
            // This IS col-major [D, Skv] with ld=D. Wait no, K^T row-major [D,Skv] stored as...
            // We don't have K^T stored. We have K[Skv,D].
            //
            // Let me try a different cuBLAS approach:
            // cublasSgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            // computes C = op(A) @ op(B) where C is col-major [m, n] with ldc.
            //
            // We want row-major scores[Sq, Skv] = row-major Q[Sq,D] @ (row-major K[Skv,D])^T
            //
            // Row-major scores[Sq, Skv] is col-major [Skv, Sq] with ld=Skv
            // Row-major Q[Sq, D] is col-major [D, Sq] with ld=D
            // Row-major K[Skv, D] is col-major [D, Skv] with ld=D
            //
            // Col-major: C[Skv, Sq] = op(A) @ op(B)
            // We need result [Skv, Sq].
            // K in col-major is [D, Skv]. K^T in col-major is... we need [Skv, D] or equiv.
            //
            // OK: C[Skv,Sq] = K_cm^T [Skv,D] @ Q_cm [D, Sq]
            //   transa=T, transb=N, m=Skv, n=Sq, k=D
            //   A=K_ptr, lda=D (since K_cm is [D,Skv] with ld=D)
            //   B=Q_ptr, ldb=D (since Q_cm is [D,Sq] with ld=D)
            //   C=scores_ptr, ldc=Skv
            //
            // This should work! Let me verify:
            // op(A) = A^T = transpose of K_cm [D,Skv] = [Skv, D], rows=Skv ✓ m=Skv
            // op(B) = B = Q_cm [D, Sq], rows=D, cols=Sq ✓ k=D, n=Sq
            // C = [Skv, Sq] = [m, n] ✓

            long strideQ = Sq * D;
            long strideK = Skv * D;
            long strideScores = Sq * Skv;

            // Batched SGEMM for QK^T
            for (long bh = 0; bh < totalHeads; bh++)
            {
                ulong qPtr = pQ + (ulong)(bh * strideQ * sizeof(float));
                ulong kPtr = pK + (ulong)(bh * strideK * sizeof(float));
                ulong sPtr = scoresBuf + (ulong)(bh * strideScores * sizeof(float));

                CublasApi.cublasSgemm(
                    _cublasHandle,
                    CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                    (int)Skv, (int)Sq, (int)D,
                    &alpha,
                    kPtr, (int)D,
                    qPtr, (int)D,
                    &beta,
                    sPtr, (int)Skv).ThrowOnCublasError();
            }

            // Step 2: Apply mask if present (elementwise add)
            if (mask is not null)
            {
                // For now, handle simple broadcast: mask [1, 1, Sq, Skv] or matching
                long maskElements = mask.ElementCount;
                long scoreElements = totalHeads * Sq * Skv;

                if (maskElements == Sq * Skv)
                {
                    // Broadcast: add same mask to each head
                    ulong maskPtr = DevPtr(mask);
                    for (long bh = 0; bh < totalHeads; bh++)
                    {
                        ulong sPtr = scoresBuf + (ulong)(bh * strideScores * sizeof(float));
                        _kernels!.LaunchAdd(sPtr, sPtr, maskPtr, (int)(Sq * Skv), _stream.Handle);
                    }
                }
                else if (maskElements == scoreElements)
                {
                    _kernels!.LaunchAdd(scoresBuf, scoresBuf, DevPtr(mask), (int)scoreElements, _stream.Handle);
                }
            }

            // Step 3: Softmax over last dimension (Skv) for each row — GPU kernel
            _kernels!.LaunchSoftmax(scoresBuf, (int)Skv, (int)(totalHeads * Sq), _stream.Handle);

            // Step 4: attn_weights @ V → output [B*H, Sq, D]
            // attn_weights is [Sq, Skv], V is [Skv, D], result is [Sq, D]
            // Row-major: C[Sq,D] = attn[Sq,Skv] @ V[Skv,D]
            // cuBLAS col-major: C_cm[D,Sq] = V_cm[D,Skv] @ attn_cm[Skv,Sq]
            //   transA=N, transB=N, m=D, n=Sq, k=Skv
            //   A=V_ptr, lda=D; B=attn_ptr, ldb=Skv; C=out_ptr, ldc=D

            long strideV = Skv * D;
            long strideOut = Sq * D;
            float one = 1.0f;
            float zero = 0.0f;

            for (long bh = 0; bh < totalHeads; bh++)
            {
                ulong sPtr = scoresBuf + (ulong)(bh * strideScores * sizeof(float));
                ulong vPtr = pV + (ulong)(bh * strideV * sizeof(float));
                ulong oPtr = pOut + (ulong)(bh * strideOut * sizeof(float));

                CublasApi.cublasSgemm(
                    _cublasHandle,
                    CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
                    (int)D, (int)Sq, (int)Skv,
                    &one,
                    vPtr, (int)D,
                    sPtr, (int)Skv,
                    &zero,
                    oPtr, (int)D).ThrowOnCublasError();
            }
        }
        finally
        {
            CudaMemory.Free(scoresBuf);
        }
    }

    // ── Activations ─────────────────────────────────────────────────────

    public void Gelu(Tensor output, Tensor input)
    {
        EnsureKernels();
        _kernels!.LaunchGelu(
            DevPtr(output),
            DevPtr(input),
            (int)input.ElementCount,
            _stream.Handle);
    }

    public void Silu(Tensor output, Tensor input)
    {
        EnsureKernels();
        _kernels!.LaunchSilu(
            DevPtr(output),
            DevPtr(input),
            (int)input.ElementCount,
            _stream.Handle);
    }

    // ── Element-wise ────────────────────────────────────────────────────

    public void Add(Tensor output, Tensor a, Tensor b)
    {
        EnsureKernels();
        _kernels!.LaunchAdd(
            DevPtr(output),
            DevPtr(a),
            DevPtr(b),
            (int)a.ElementCount,
            _stream.Handle);
    }

    public void Mul(Tensor output, Tensor a, Tensor b)
    {
        EnsureKernels();
        _kernels!.LaunchMul(
            DevPtr(output),
            DevPtr(a),
            DevPtr(b),
            (int)a.ElementCount,
            _stream.Handle);
    }

    public void Scale(Tensor output, Tensor input, float scalar)
    {
        EnsureKernels();
        _kernels!.LaunchScale(
            DevPtr(output),
            DevPtr(input),
            scalar,
            (int)input.ElementCount,
            _stream.Handle);
    }

    public void Clamp(Tensor output, Tensor input, float min, float max)
    {
        EnsureKernels();
        _kernels!.LaunchClamp(
            DevPtr(output),
            DevPtr(input),
            min, max,
            (int)input.ElementCount,
            _stream.Handle);
    }

    private void EnsureKernels()
    {
        if (_kernels == null)
            throw new InvalidOperationException("PTX kernels not loaded. Provide a ptxDir to the CudaBackend constructor.");
    }

    /// <summary>Gets the device pointer as a ulong for CUDA API calls.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe ulong DevPtr(Tensor tensor) => (ulong)(nint)tensor.DataPointer;

    // ── Shape Operations ────────────────────────────────────────────────

    /// <summary>Concatenates tensors along the specified dimension using device-to-device copies.</summary>
    public unsafe void Concat(Tensor output, ReadOnlySpan<Tensor> inputs, int dim)
    {
        if (dim == 0)
        {
            // Simple case: just copy contiguously
            ulong offset = 0;
            ulong pOut = DevPtr(output);
            for (int t = 0; t < inputs.Length; t++)
            {
                nuint byteSize = (nuint)(inputs[t].ElementCount * sizeof(float));
                CudaMemory.CopyDeviceToDevice(pOut + offset, DevPtr(inputs[t]), byteSize);
                offset += (ulong)byteSize;
            }
        }
        else
        {
            // General case: compute outer/inner strides and copy slices
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

            ulong pOut = DevPtr(output);
            long outDimStride = output.Shape[dim] * innerSize;

            for (long outer = 0; outer < outerSize; outer++)
            {
                long dimOffset = 0;
                for (int t = 0; t < inputs.Length; t++)
                {
                    long inputDimSize = inputs[t].Shape[dim];
                    long sliceSize = inputDimSize * innerSize;
                    nuint sliceBytes = (nuint)(sliceSize * sizeof(float));

                    long inDimStride = inputDimSize * innerSize;
                    ulong srcOffset = (ulong)((outer * inDimStride) * sizeof(float));
                    ulong dstOffset = (ulong)((outer * outDimStride + dimOffset) * sizeof(float));

                    CudaMemory.CopyDeviceToDevice(pOut + dstOffset, DevPtr(inputs[t]) + srcOffset, sliceBytes);
                    dimOffset += sliceSize;
                }
            }
        }
    }

    public void Split(ReadOnlySpan<Tensor> outputs, Tensor input, int dim)
    {
        throw new NotImplementedException("CUDA Split not yet implemented");
    }

    // ── Sampling ────────────────────────────────────────────────────────

    public void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW)
    {
        EnsureKernels();

        int batch = (int)input.Shape[0];
        int channels = (int)input.Shape[1];
        int inH = (int)input.Shape[2];
        int inW = (int)input.Shape[3];
        int outH = inH * scaleH;
        int outW = inW * scaleW;

        _kernels!.LaunchUpsampleNearest2D(
            DevPtr(output), DevPtr(input),
            batch, channels, inH, inW, outH, outW, scaleH, scaleW,
            _stream.Handle);
    }

    public void UpsampleBilinear2D(Tensor output, Tensor input, int scaleH, int scaleW)
    {
        throw new NotImplementedException("CUDA UpsampleBilinear2D not yet implemented");
    }

    // ── Data Movement ───────────────────────────────────────────────────

    /// <summary>Copies tensor data between host and device or device to device.</summary>
    public unsafe void CopyTo(Tensor destination, Tensor source)
    {
        nuint byteSize = (nuint)(source.ElementCount * source.DType.SizeInBytes);

        bool srcGpu = source.Device.IsCuda;
        bool dstGpu = destination.Device.IsCuda;

        if (srcGpu && dstGpu)
        {
            CudaMemory.CopyDeviceToDevice(
                DevPtr(destination),
                DevPtr(source),
                byteSize);
        }
        else if (!srcGpu && dstGpu)
        {
            CudaMemory.CopyHostToDevice(
                DevPtr(destination),
                source.DataPointer,
                byteSize);
        }
        else if (srcGpu && !dstGpu)
        {
            CudaMemory.CopyDeviceToHost(
                destination.DataPointer,
                DevPtr(source),
                byteSize);
        }
        else
        {
            Buffer.MemoryCopy(source.DataPointer, destination.DataPointer, (long)byteSize, (long)byteSize);
        }
    }

    /// <summary>Fills a tensor with a constant float value.</summary>
    public unsafe void Fill(Tensor tensor, float value)
    {
        uint pattern = *(uint*)&value;
        nuint count = (nuint)tensor.ElementCount;

        if (tensor.Device.IsCuda)
        {
            CudaMemory.Fill32(DevPtr(tensor), pattern, count);
        }
        else
        {
            float* ptr = (float*)tensor.DataPointer;
            for (long i = 0; i < tensor.ElementCount; i++)
            {
                ptr[i] = value;
            }
        }
    }

    // ── Audio ───────────────────────────────────────────────────────────

    public void Fft(Tensor output, Tensor input)
    {
        throw new NotSupportedException("CUDA FFT not supported — use CPU backend for audio");
    }

    public void Stft(Tensor output, Tensor input, int fftSize, int hopLength, Tensor window)
    {
        throw new NotSupportedException("CUDA STFT not supported — use CPU backend for audio");
    }

    public void MelFilterbank(Tensor output, Tensor input, Tensor filters)
    {
        throw new NotSupportedException("CUDA MelFilterbank not supported — use CPU backend for audio");
    }

    // ── Disposal ────────────────────────────────────────────────────────

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;

            _kernels?.Dispose();

            if (_cublasHandle != 0)
            {
                CublasApi.cublasDestroy(_cublasHandle);
                _cublasHandle = 0;
            }

            _stream.Dispose();
            _context.Dispose();
        }
        GC.SuppressFinalize(this);
    }

    ~CudaBackend()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_cublasHandle != 0)
            {
                CublasApi.cublasDestroy(_cublasHandle);
                _cublasHandle = 0;
            }
        }
    }
}
