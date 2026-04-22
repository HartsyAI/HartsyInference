using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Cuda;

/// <summary>CUDA GPU backend implementing IBackend. Routes operations to cuBLAS SGEMM for matmul and PTX kernels for element-wise/normalization ops. Transparently handles CPU tensors via auto-transfer (H2D before each op, D2H after, sync per op).</summary>
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

    // ── Linear Algebra -------------------------------------------------------

    /// <summary>Matrix multiply via cuBLAS SGEMM: output = a @ b</summary>
    public unsafe void MatMul(Tensor output, Tensor a, Tensor b)
    {
        int M = (int)a.Shape[0];
        int K = (int)a.Shape[1];
        int N = (int)b.Shape[1];

        ulong pA = 0, pB = 0, pC = 0;
        try
        {
            pA = GpuTransferHelper.CopyToDevice(a);
            pB = GpuTransferHelper.CopyToDevice(b);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pC = GpuTransferHelper.AllocateDevice(outBytes);

            float alpha = 1.0f;
            float beta = 0.0f;

            CublasApi.cublasSgemm(
                _cublasHandle,
                CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
                N, M, K,
                &alpha,
                pB, N,
                pA, K,
                &beta,
                pC, N).ThrowOnCublasError();

            Sync();
            GpuTransferHelper.CopyToHost(output, pC, outBytes);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pA);
            GpuTransferHelper.FreeDevice(pB);
            GpuTransferHelper.FreeDevice(pC);
        }
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

        ulong pA = 0, pB = 0, pC = 0;
        try
        {
            pA = GpuTransferHelper.CopyToDevice(a);
            pB = GpuTransferHelper.CopyToDevice(b);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pC = GpuTransferHelper.AllocateDevice(outBytes);

            float alpha = 1.0f;
            float beta = 0.0f;

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

            Sync();
            GpuTransferHelper.CopyToHost(output, pC, outBytes);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pA);
            GpuTransferHelper.FreeDevice(pB);
            GpuTransferHelper.FreeDevice(pC);
        }
    }

    // ── Convolution ----------------------------------------------------------

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

        bool is1x1 = kH == 1 && kW == 1 && strideH == 1 && strideW == 1 && padH == 0 && padW == 0;

        ulong pInput = 0, pWeight = 0, pBias = 0, pOutput = 0, colBuf = 0;
        try
        {
            pInput = GpuTransferHelper.CopyToDevice(input);
            pWeight = GpuTransferHelper.CopyToDevice(weight);
            if (bias is not null)
            {
                pBias = GpuTransferHelper.CopyToDevice(bias);
            }
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOutput = GpuTransferHelper.AllocateDevice(outBytes);

            if (!is1x1)
            {
                colBuf = CudaMemory.Allocate((nuint)(colRows * colCols * sizeof(float)));
            }

            float alpha = 1.0f;
            float beta = 0.0f;

            for (int b = 0; b < batch; b++)
            {
                int inputBatchOffset = b * inCh;

                ulong colPtr;
                if (is1x1)
                {
                    colPtr = pInput + (ulong)(b * inCh * inH * inW * sizeof(float));
                }
                else
                {
                    _kernels!.LaunchIm2Col(
                        colBuf, pInput,
                        inCh, inH, inW, kH, kW,
                        padH, padW, strideH, strideW,
                        outH, outW, inputBatchOffset,
                        _stream.Handle);
                    colPtr = colBuf;
                }

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

            if (bias is not null)
            {
                int totalElements = batch * outCh * outH * outW;
                _kernels!.LaunchBiasAdd(
                    pOutput, pBias,
                    outCh, outH * outW, totalElements,
                    _stream.Handle);
            }

            Sync();
            GpuTransferHelper.CopyToHost(output, pOutput, outBytes);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pInput);
            GpuTransferHelper.FreeDevice(pWeight);
            GpuTransferHelper.FreeDevice(pBias);
            GpuTransferHelper.FreeDevice(pOutput);
            if (colBuf != 0) CudaMemory.Free(colBuf);
        }
    }

    // ── Normalization --------------------------------------------------------

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

        ulong pOut = 0, pIn = 0, pW = 0, pB = 0;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(weight);
            pB = GpuTransferHelper.CopyToDevice(bias);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            _kernels!.LaunchGroupNorm(
                pOut, pIn, pW, pB,
                batch, channels, spatial, groups, eps,
                _stream.Handle);

            Sync();
            GpuTransferHelper.CopyToHost(output, pOut, outBytes);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pW);
            GpuTransferHelper.FreeDevice(pB);
        }
    }

    public void LayerNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, float eps)
    {
        EnsureKernels();

        int normDim = (int)input.Shape[input.Shape.Rank - 1];
        int totalRows = (int)(input.ElementCount / normDim);

        ulong pOut = 0, pIn = 0, pW = 0, pB = 0;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(weight);
            pB = GpuTransferHelper.CopyToDevice(bias);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            _kernels!.LaunchLayerNorm(
                pOut, pIn, pW, pB,
                normDim, totalRows, eps,
                _stream.Handle);

            Sync();
            GpuTransferHelper.CopyToHost(output, pOut, outBytes);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pW);
            GpuTransferHelper.FreeDevice(pB);
        }
    }

    public void RmsNorm(Tensor output, Tensor input, Tensor weight, float eps)
    {
        throw new NotImplementedException("CUDA RmsNorm not yet implemented");
    }

    // ── Attention ------------------------------------------------------------

    /// <summary>Scaled dot-product attention via cuBLAS batched GEMM: softmax(Q @ K^T * scale) @ V.</summary>
    public unsafe void ScaledDotProductAttention(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale)
    {
        EnsureKernels();

        long B = query.Shape[0];
        long H = query.Shape[1];
        long Sq = query.Shape[2];
        long D = query.Shape[3];
        long Skv = key.Shape[2];

        long totalHeads = B * H;

        ulong pQ = 0, pK = 0, pV = 0, pMask = 0, pOut = 0, scoresBuf = 0;
        try
        {
            pQ = GpuTransferHelper.CopyToDevice(query);
            pK = GpuTransferHelper.CopyToDevice(key);
            pV = GpuTransferHelper.CopyToDevice(value);
            if (mask is not null)
            {
                pMask = GpuTransferHelper.CopyToDevice(mask);
            }
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            nuint scoresBytes = (nuint)(totalHeads * Sq * Skv * sizeof(float));
            scoresBuf = CudaMemory.Allocate(scoresBytes);

            float alpha = scale;
            float beta = 0.0f;

            long strideQ = Sq * D;
            long strideK = Skv * D;
            long strideScores = Sq * Skv;

            // QK^T per head
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

            // Apply mask if present
            if (mask is not null)
            {
                long maskElements = mask.ElementCount;
                long scoreElements = totalHeads * Sq * Skv;

                if (maskElements == Sq * Skv)
                {
                    for (long bh = 0; bh < totalHeads; bh++)
                    {
                        ulong sPtr = scoresBuf + (ulong)(bh * strideScores * sizeof(float));
                        _kernels!.LaunchAdd(sPtr, sPtr, pMask, (int)(Sq * Skv), _stream.Handle);
                    }
                }
                else if (maskElements == scoreElements)
                {
                    _kernels!.LaunchAdd(scoresBuf, scoresBuf, pMask, (int)scoreElements, _stream.Handle);
                }
            }

            // Softmax
            _kernels!.LaunchSoftmax(scoresBuf, (int)Skv, (int)(totalHeads * Sq), _stream.Handle);

            // attn_weights @ V
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

            Sync();
            GpuTransferHelper.CopyToHost(output, pOut, outBytes);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pQ);
            GpuTransferHelper.FreeDevice(pK);
            GpuTransferHelper.FreeDevice(pV);
            GpuTransferHelper.FreeDevice(pMask);
            GpuTransferHelper.FreeDevice(pOut);
            if (scoresBuf != 0) CudaMemory.Free(scoresBuf);
        }
    }

    // ── Activations ----------------------------------------------------------

    public void Gelu(Tensor output, Tensor input)
    {
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            _kernels!.LaunchGelu(pOut, pIn, (int)input.ElementCount, _stream.Handle);

            Sync();
            GpuTransferHelper.CopyToHost(output, pOut, outBytes);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    public void Silu(Tensor output, Tensor input)
    {
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            _kernels!.LaunchSilu(pOut, pIn, (int)input.ElementCount, _stream.Handle);

            Sync();
            GpuTransferHelper.CopyToHost(output, pOut, outBytes);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    // ── Element-wise ---------------------------------------------------------

    public void Add(Tensor output, Tensor a, Tensor b)
    {
        EnsureKernels();

        ulong pOut = 0, pA = 0, pB = 0;
        try
        {
            pA = GpuTransferHelper.CopyToDevice(a);
            pB = GpuTransferHelper.CopyToDevice(b);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            _kernels!.LaunchAdd(pOut, pA, pB, (int)a.ElementCount, _stream.Handle);

            Sync();
            GpuTransferHelper.CopyToHost(output, pOut, outBytes);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pA);
            GpuTransferHelper.FreeDevice(pB);
        }
    }

    public void Mul(Tensor output, Tensor a, Tensor b)
    {
        EnsureKernels();

        ulong pOut = 0, pA = 0, pB = 0;
        try
        {
            pA = GpuTransferHelper.CopyToDevice(a);
            pB = GpuTransferHelper.CopyToDevice(b);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            _kernels!.LaunchMul(pOut, pA, pB, (int)a.ElementCount, _stream.Handle);

            Sync();
            GpuTransferHelper.CopyToHost(output, pOut, outBytes);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pA);
            GpuTransferHelper.FreeDevice(pB);
        }
    }

    public void Scale(Tensor output, Tensor input, float scalar)
    {
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            _kernels!.LaunchScale(pOut, pIn, scalar, (int)input.ElementCount, _stream.Handle);

            Sync();
            GpuTransferHelper.CopyToHost(output, pOut, outBytes);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    public void Clamp(Tensor output, Tensor input, float min, float max)
    {
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            _kernels!.LaunchClamp(pOut, pIn, min, max, (int)input.ElementCount, _stream.Handle);

            Sync();
            GpuTransferHelper.CopyToHost(output, pOut, outBytes);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    private void EnsureKernels()
    {
        if (_kernels == null)
            throw new InvalidOperationException("PTX kernels not loaded. Provide a ptxDir to the CudaBackend constructor.");
    }

    /// <summary>Synchronizes the default compute stream.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Sync()
    {
        CudaDriverApi.cuStreamSynchronize(_stream.Handle).ThrowOnError();
    }

    // ── Shape Operations -----------------------------------------------------

    /// <summary>Concatenates tensors along the specified dimension.</summary>
    public unsafe void Concat(Tensor output, ReadOnlySpan<Tensor> inputs, int dim)
    {
        ulong[] gpuInputs = new ulong[inputs.Length];
        ulong pOut = 0;
        try
        {
            for (int t = 0; t < inputs.Length; t++)
            {
                gpuInputs[t] = GpuTransferHelper.CopyToDevice(inputs[t]);
            }
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (dim == 0)
            {
                ulong offset = 0;
                for (int t = 0; t < inputs.Length; t++)
                {
                    nuint byteSize = (nuint)(inputs[t].ElementCount * sizeof(float));
                    CudaMemory.CopyDeviceToDevice(pOut + offset, gpuInputs[t], byteSize);
                    offset += (ulong)byteSize;
                }
            }
            else
            {
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

                        CudaMemory.CopyDeviceToDevice(pOut + dstOffset, gpuInputs[t] + srcOffset, sliceBytes);
                        dimOffset += sliceSize;
                    }
                }
            }

            Sync();
            GpuTransferHelper.CopyToHost(output, pOut, outBytes);
        }
        finally
        {
            for (int t = 0; t < gpuInputs.Length; t++)
            {
                GpuTransferHelper.FreeDevice(gpuInputs[t]);
            }
            GpuTransferHelper.FreeDevice(pOut);
        }
    }

    public void Split(ReadOnlySpan<Tensor> outputs, Tensor input, int dim)
    {
        throw new NotImplementedException("CUDA Split not yet implemented");
    }

    // ── Sampling -------------------------------------------------------------

    public void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW)
    {
        EnsureKernels();

        int batch = (int)input.Shape[0];
        int channels = (int)input.Shape[1];
        int inH = (int)input.Shape[2];
        int inW = (int)input.Shape[3];
        int outH = inH * scaleH;
        int outW = inW * scaleW;

        ulong pOut = 0, pIn = 0;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            _kernels!.LaunchUpsampleNearest2D(
                pOut, pIn,
                batch, channels, inH, inW, outH, outW, scaleH, scaleW,
                _stream.Handle);

            Sync();
            GpuTransferHelper.CopyToHost(output, pOut, outBytes);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    public void UpsampleBilinear2D(Tensor output, Tensor input, int scaleH, int scaleW)
    {
        throw new NotImplementedException("CUDA UpsampleBilinear2D not yet implemented");
    }

    // ── Data Movement --------------------------------------------------------

    /// <summary>Copies tensor data between host and device or device to device.</summary>
    public unsafe void CopyTo(Tensor destination, Tensor source)
    {
        nuint byteSize = (nuint)(source.ElementCount * source.DType.SizeInBytes);

        bool srcGpu = source.Device.IsCuda;
        bool dstGpu = destination.Device.IsCuda;

        if (srcGpu && dstGpu)
        {
            CudaMemory.CopyDeviceToDevice(
                (ulong)(nint)destination.DataPointer,
                (ulong)(nint)source.DataPointer,
                byteSize);
        }
        else if (!srcGpu && dstGpu)
        {
            CudaMemory.CopyHostToDevice(
                (ulong)(nint)destination.DataPointer,
                source.DataPointer,
                byteSize);
        }
        else if (srcGpu && !dstGpu)
        {
            CudaMemory.CopyDeviceToHost(
                destination.DataPointer,
                (ulong)(nint)source.DataPointer,
                byteSize);
        }
        else
        {
            // Both CPU - direct memory copy
            Buffer.MemoryCopy(source.DataPointer, destination.DataPointer, (long)byteSize, (long)byteSize);
        }
    }

    /// <summary>Fills a tensor with a constant float value. Works on CPU tensors directly.</summary>
    public unsafe void Fill(Tensor tensor, float value)
    {
        // Fill operates directly on CPU tensor memory - no GPU transfer needed
        float* ptr = (float*)tensor.DataPointer;
        for (long i = 0; i < tensor.ElementCount; i++)
        {
            ptr[i] = value;
        }
    }

    // ── Audio ----------------------------------------------------------------

    public void Fft(Tensor output, Tensor input)
    {
        throw new NotSupportedException("CUDA FFT not supported - use CPU backend for audio");
    }

    public void Stft(Tensor output, Tensor input, int fftSize, int hopLength, Tensor window)
    {
        throw new NotSupportedException("CUDA STFT not supported - use CPU backend for audio");
    }

    public void MelFilterbank(Tensor output, Tensor input, Tensor filters)
    {
        throw new NotSupportedException("CUDA MelFilterbank not supported - use CPU backend for audio");
    }

    // ── Disposal -------------------------------------------------------------

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
