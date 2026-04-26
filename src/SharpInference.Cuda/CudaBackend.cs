using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Cuda;

/// <summary>CUDA GPU backend implementing IBackend. Routes operations to cuBLAS SGEMM for matmul and PTX kernels for element-wise/normalization ops. Uses activation caching to keep intermediate results on GPU between ops — lazy sync to CPU on DataPointer access.</summary>
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
        // Must use blocking stream (CU_STREAM_DEFAULT) because GpuTransferHelper uses synchronous
        // cuMemcpyHtoD/DtoH which operate on the NULL stream. A non-blocking stream does NOT
        // synchronize with the NULL stream, causing race conditions where kernels read incomplete
        // data from in-progress H2D transfers. Fix: switch to cuMemcpyHtoDAsync on this stream.
        _stream = new CudaStream(nonBlocking: false);
        Device = DeviceKind.Cuda(deviceOrdinal);

        // Initialize cuBLAS
        CublasApi.cublasCreate(out _cublasHandle).ThrowOnCublasError();
        CublasApi.cublasSetStream(_cublasHandle, _stream.Handle).ThrowOnCublasError();

        // Give GpuTransferHelper the stream handle for FreeAsync and lazy-sync callbacks
        GpuTransferHelper.SetStream(_stream.Handle);

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

    /// <summary>Matrix multiply via cuBLAS GemmEx: output = a @ b. Supports mixed F32/F16/F8 dtypes.</summary>
    public unsafe void MatMul(Tensor output, Tensor a, Tensor b)
    {
        EnsureKernels();

        int M = (int)a.Shape[0];
        int K = (int)a.Shape[1];
        int N = (int)b.Shape[1];

        ulong pA = 0, pB = 0, pC = 0, pBCast = 0, pACast = 0;
        bool cachedOutput = false;
        try
        {
            pA = GpuTransferHelper.CopyToDevice(a);
            pB = GpuTransferHelper.CopyToDevice(b);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pC = GpuTransferHelper.AllocateDevice(outBytes);

            float alpha = 1.0f;
            float beta = 0.0f;

            // Resolve GEMM dtype: FP8 inputs get cast to F16 (Ampere fallback)
            DType gemmDtype = ResolveGemmDtype(a.DType);
            ulong aPtr = pA;
            if (a.DType.IsFp8)
            {
                pACast = CudaMemory.Allocate((nuint)(a.ElementCount * DType.F16.SizeInBytes));
                _kernels!.LaunchCastF8E4M3ToF16(pACast, pA, (int)a.ElementCount, _stream.Handle);
                aPtr = pACast;
            }

            // cuBLAS requires A and B to have the same dtype; cast B if mismatched
            ulong bPtr = pB;
            DType bResolved = ResolveGemmDtype(b.DType);
            if (bResolved != gemmDtype)
            {
                pBCast = CudaMemory.Allocate((nuint)(b.ElementCount * gemmDtype.SizeInBytes));
                CastOnGpu(pBCast, pB, b.DType, gemmDtype, (int)b.ElementCount);
                bPtr = pBCast;
            }
            else if (b.DType.IsFp8)
            {
                pBCast = CudaMemory.Allocate((nuint)(b.ElementCount * DType.F16.SizeInBytes));
                _kernels!.LaunchCastF8E4M3ToF16(pBCast, pB, (int)b.ElementCount, _stream.Handle);
                bPtr = pBCast;
            }

            int gemmType = gemmDtype == DType.F16 ? CublasApi.CUDA_R_16F : CublasApi.CUDA_R_32F;
            int cType = output.DType == DType.F16 ? CublasApi.CUDA_R_16F : CublasApi.CUDA_R_32F;

            CublasApi.cublasGemmEx(
                _cublasHandle,
                CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
                N, M, K,
                &alpha,
                bPtr, gemmType, N,
                pA, gemmType, K,
                &beta,
                pC, cType, N,
                CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();

            GpuTransferHelper.CacheActivation(output, pC, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pA);
            GpuTransferHelper.FreeDevice(pB);
            if (pACast != 0) CudaMemory.FreeAsync(pACast, _stream.Handle);
            if (pBCast != 0) CudaMemory.FreeAsync(pBCast, _stream.Handle);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pC);
        }
    }

    /// <summary>Linear layer via cuBLAS GemmEx with transpose: output = input × weight^T + bias. Supports mixed F32/F16/F8 dtypes.</summary>
    public unsafe void Linear(Tensor output, Tensor input, Tensor weight, Tensor? bias)
    {
        EnsureKernels();

        int N = (int)weight.Shape[0]; // outDim
        int K = (int)weight.Shape[1]; // inDim
        int M = (int)(input.ElementCount / K); // batch*seqLen

        ulong pInput = 0, pWeight = 0, pBias = 0, pOutput = 0, pInputCast = 0, pWeightCast = 0, pBiasCast = 0;
        bool cachedOutput = false;
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

            float alpha = 1.0f;
            float beta = 0.0f;

            // Resolve GEMM dtype: FP8 inputs get cast to F16 (Ampere fallback)
            DType gemmDtype = ResolveGemmDtype(input.DType);
            ulong inputPtr = pInput;
            if (input.DType.IsFp8)
            {
                pInputCast = CudaMemory.Allocate((nuint)(input.ElementCount * DType.F16.SizeInBytes));
                _kernels!.LaunchCastF8E4M3ToF16(pInputCast, pInput, (int)input.ElementCount, _stream.Handle);
                inputPtr = pInputCast;
            }

            // cuBLAS requires A and B to have the same dtype; cast weight if mismatched
            ulong weightPtr = pWeight;
            DType weightResolved = ResolveGemmDtype(weight.DType);
            if (weightResolved != gemmDtype || weight.DType.IsFp8)
            {
                pWeightCast = CudaMemory.Allocate((nuint)(weight.ElementCount * gemmDtype.SizeInBytes));
                CastOnGpu(pWeightCast, pWeight, weight.DType, gemmDtype, (int)weight.ElementCount);
                weightPtr = pWeightCast;
            }

            int gemmType = gemmDtype == DType.F16 ? CublasApi.CUDA_R_16F : CublasApi.CUDA_R_32F;
            int outputType = output.DType == DType.F16 ? CublasApi.CUDA_R_16F : CublasApi.CUDA_R_32F;

            // cuBLAS col-major: C_cm = op(A) × op(B) where op(A)=weight^T [N,K], op(B)=input [K,M]
            // Row-major interpretation: output[M,N] = input[M,K] × weight^T[K,N]
            CublasApi.cublasGemmEx(
                _cublasHandle,
                CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                N, M, K,
                &alpha,
                weightPtr, gemmType, K,
                inputPtr, gemmType, K,
                &beta,
                pOutput, outputType, N,
                CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();

            // Add bias on GPU if present (cast bias to match output dtype if needed)
            if (bias is not null)
            {
                int totalElements = M * N;
                ulong biasPtr = pBias;

                if (output.DType != bias!.DType)
                {
                    pBiasCast = CudaMemory.Allocate((nuint)(bias.ElementCount * output.DType.SizeInBytes));
                    CastOnGpu(pBiasCast, pBias, bias.DType, output.DType, (int)bias.ElementCount);
                    biasPtr = pBiasCast;
                }

                if (output.DType == DType.F16)
                    _kernels!.LaunchBiasAddF16(pOutput, biasPtr, N, 1, totalElements, _stream.Handle);
                else
                    _kernels!.LaunchBiasAdd(pOutput, biasPtr, N, 1, totalElements, _stream.Handle);
            }

            GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pInput);
            GpuTransferHelper.FreeDevice(pWeight);
            GpuTransferHelper.FreeDevice(pBias);
            if (pInputCast != 0) CudaMemory.FreeAsync(pInputCast, _stream.Handle);
            if (pWeightCast != 0) CudaMemory.FreeAsync(pWeightCast, _stream.Handle);
            if (pBiasCast != 0) CudaMemory.FreeAsync(pBiasCast, _stream.Handle);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOutput);
        }
    }

    /// <summary>Batched matrix multiply via cuBLAS strided batched GEMM. Supports mixed F32/F16/F8 dtypes.</summary>
    public unsafe void BatchedMatMul(Tensor output, Tensor a, Tensor b)
    {
        EnsureKernels();

        long batchSize = a.Shape[0];
        int M = (int)a.Shape[1];
        int K = (int)a.Shape[2];

        bool bIs2D = b.Shape.Rank == 2;
        int N = bIs2D ? (int)b.Shape[1] : (int)b.Shape[2];

        long strideA = M * K;
        long strideB = bIs2D ? 0 : K * N;
        long strideC = M * N;

        ulong pA = 0, pB = 0, pC = 0, pACast = 0, pBCast = 0;
        bool cachedOutput = false;
        try
        {
            pA = GpuTransferHelper.CopyToDevice(a);
            pB = GpuTransferHelper.CopyToDevice(b);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pC = GpuTransferHelper.AllocateDevice(outBytes);

            float alpha = 1.0f;
            float beta = 0.0f;

            // Resolve GEMM dtype: FP8 inputs get cast to F16 (Ampere fallback)
            DType gemmDtype = ResolveGemmDtype(a.DType);
            ulong aPtr = pA;
            if (a.DType.IsFp8)
            {
                pACast = CudaMemory.Allocate((nuint)(a.ElementCount * DType.F16.SizeInBytes));
                _kernels!.LaunchCastF8E4M3ToF16(pACast, pA, (int)a.ElementCount, _stream.Handle);
                aPtr = pACast;
            }

            // cuBLAS requires A and B to have the same dtype; cast B if mismatched
            ulong bPtr = pB;
            DType bResolved = ResolveGemmDtype(b.DType);
            if (bResolved != gemmDtype || b.DType.IsFp8)
            {
                pBCast = CudaMemory.Allocate((nuint)(b.ElementCount * gemmDtype.SizeInBytes));
                CastOnGpu(pBCast, pB, b.DType, gemmDtype, (int)b.ElementCount);
                bPtr = pBCast;
            }

            int gemmType = gemmDtype == DType.F16 ? CublasApi.CUDA_R_16F : CublasApi.CUDA_R_32F;
            int cType = output.DType == DType.F16 ? CublasApi.CUDA_R_16F : CublasApi.CUDA_R_32F;

            CublasApi.cublasGemmStridedBatchedEx(
                _cublasHandle,
                CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
                N, M, K,
                &alpha,
                bPtr, gemmType, N, strideB,
                aPtr, gemmType, K, strideA,
                &beta,
                pC, cType, N, strideC,
                (int)batchSize,
                CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();

            GpuTransferHelper.CacheActivation(output, pC, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pA);
            GpuTransferHelper.FreeDevice(pB);
            if (pACast != 0) CudaMemory.FreeAsync(pACast, _stream.Handle);
            if (pBCast != 0) CudaMemory.FreeAsync(pBCast, _stream.Handle);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pC);
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
        // FP8 inputs get treated as F16 after cast
        DType effectiveInputDtype = ResolveGemmDtype(input.DType);
        bool inputIsF16 = effectiveInputDtype == DType.F16;
        bool outputIsF16 = output.DType == DType.F16;
        int elemSize = effectiveInputDtype.SizeInBytes;
        int outElemSize = output.DType.SizeInBytes;

        ulong pInput = 0, pWeight = 0, pBias = 0, pOutput = 0, colBuf = 0, pInputCast = 0, pWeightCast = 0, pBiasCast = 0;
        bool cachedOutput = false;
        try
        {
            pInput = GpuTransferHelper.CopyToDevice(input);
            pWeight = GpuTransferHelper.CopyToDevice(weight);

            // Cast FP8 input to F16 for GEMM
            ulong inputPtr = pInput;
            if (input.DType.IsFp8)
            {
                pInputCast = CudaMemory.Allocate((nuint)(input.ElementCount * DType.F16.SizeInBytes));
                _kernels!.LaunchCastF8E4M3ToF16(pInputCast, pInput, (int)input.ElementCount, _stream.Handle);
                inputPtr = pInputCast;
            }
            if (bias is not null)
            {
                pBias = GpuTransferHelper.CopyToDevice(bias);
            }
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOutput = GpuTransferHelper.AllocateDevice(outBytes);

            if (!is1x1)
            {
                colBuf = CudaMemory.Allocate((nuint)((long)colRows * colCols * elemSize));
            }

            float alpha = 1.0f;
            float beta = 0.0f;

            // cuBLAS requires A and B to have the same dtype; cast weight if mismatched
            DType gemmDtype = effectiveInputDtype;
            ulong weightPtr = pWeight;
            DType weightResolved = ResolveGemmDtype(weight.DType);
            if (weightResolved != gemmDtype || weight.DType.IsFp8)
            {
                pWeightCast = CudaMemory.Allocate((nuint)(weight.ElementCount * gemmDtype.SizeInBytes));
                CastOnGpu(pWeightCast, pWeight, weight.DType, gemmDtype, (int)weight.ElementCount);
                weightPtr = pWeightCast;
            }

            int gemmType = inputIsF16 ? CublasApi.CUDA_R_16F : CublasApi.CUDA_R_32F;
            int gemmOutType = outputIsF16 ? CublasApi.CUDA_R_16F : CublasApi.CUDA_R_32F;

            for (int b = 0; b < batch; b++)
            {
                int inputBatchOffset = b * inCh;

                ulong colPtr;
                if (is1x1)
                {
                    colPtr = inputPtr + (ulong)((long)b * inCh * inH * inW * elemSize);
                }
                else
                {
                    if (inputIsF16)
                        _kernels!.LaunchIm2ColF16(
                            colBuf, inputPtr,
                            inCh, inH, inW, kH, kW,
                            padH, padW, strideH, strideW,
                            outH, outW, inputBatchOffset,
                            _stream.Handle);
                    else
                        _kernels!.LaunchIm2Col(
                            colBuf, inputPtr,
                            inCh, inH, inW, kH, kW,
                            padH, padW, strideH, strideW,
                            outH, outW, inputBatchOffset,
                            _stream.Handle);
                    colPtr = colBuf;
                }

                ulong outBatchPtr = pOutput + (ulong)((long)b * outCh * outH * outW * outElemSize);

                CublasApi.cublasGemmEx(
                    _cublasHandle,
                    CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
                    colCols, outCh, colRows,
                    &alpha,
                    colPtr, gemmType, colCols,
                    weightPtr, gemmType, colRows,
                    &beta,
                    outBatchPtr, gemmOutType, colCols,
                    CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
            }

            // Add bias (cast if dtype mismatch)
            if (bias is not null)
            {
                int totalElements = batch * outCh * outH * outW;
                ulong biasPtr = pBias;

                if (output.DType != bias!.DType)
                {
                    pBiasCast = CudaMemory.Allocate((nuint)(bias.ElementCount * output.DType.SizeInBytes));
                    CastOnGpu(pBiasCast, pBias, bias.DType, output.DType, (int)bias.ElementCount);
                    biasPtr = pBiasCast;
                }

                if (outputIsF16)
                    _kernels!.LaunchBiasAddF16(
                        pOutput, biasPtr,
                        outCh, outH * outW, totalElements,
                        _stream.Handle);
                else
                    _kernels!.LaunchBiasAdd(
                        pOutput, biasPtr,
                        outCh, outH * outW, totalElements,
                        _stream.Handle);
            }

            GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pInput);
            GpuTransferHelper.FreeDevice(pWeight);
            GpuTransferHelper.FreeDevice(pBias);
            if (pInputCast != 0) CudaMemory.FreeAsync(pInputCast, _stream.Handle);
            if (pWeightCast != 0) CudaMemory.FreeAsync(pWeightCast, _stream.Handle);
            if (pBiasCast != 0) CudaMemory.FreeAsync(pBiasCast, _stream.Handle);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOutput);
            if (colBuf != 0) CudaMemory.FreeAsync(colBuf, _stream.Handle);
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

        ulong pOut = 0, pIn = 0, pW = 0, pB = 0, pWCast = 0, pBCast = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(weight);
            pB = GpuTransferHelper.CopyToDevice(bias);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.F16)
            {
                // Cast weight/bias to F16 if stored as F32 (common for norm params in FP16 models)
                ulong wPtr = pW;
                ulong bPtr = pB;
                if (weight.DType == DType.F32)
                {
                    pWCast = CudaMemory.Allocate((nuint)(weight.ElementCount * 2));
                    _kernels!.LaunchCastF32ToF16(pWCast, pW, (int)weight.ElementCount, _stream.Handle);
                    wPtr = pWCast;
                }
                if (bias.DType == DType.F32)
                {
                    pBCast = CudaMemory.Allocate((nuint)(bias.ElementCount * 2));
                    _kernels!.LaunchCastF32ToF16(pBCast, pB, (int)bias.ElementCount, _stream.Handle);
                    bPtr = pBCast;
                }
                _kernels!.LaunchGroupNormF16(
                    pOut, pIn, wPtr, bPtr,
                    batch, channels, spatial, groups, eps,
                    _stream.Handle);
            }
            else
            {
                _kernels!.LaunchGroupNorm(
                    pOut, pIn, pW, pB,
                    batch, channels, spatial, groups, eps,
                    _stream.Handle);
            }

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pW);
            GpuTransferHelper.FreeDevice(pB);
            if (pWCast != 0) CudaMemory.FreeAsync(pWCast, _stream.Handle);
            if (pBCast != 0) CudaMemory.FreeAsync(pBCast, _stream.Handle);
        }
    }

    public void LayerNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, float eps)
    {
        EnsureKernels();

        int normDim = (int)input.Shape[input.Shape.Rank - 1];
        int totalRows = (int)(input.ElementCount / normDim);

        ulong pOut = 0, pIn = 0, pW = 0, pB = 0, pWCast = 0, pBCast = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(weight);
            pB = GpuTransferHelper.CopyToDevice(bias);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.F16)
            {
                // Cast weight/bias to F16 if stored as F32 (common for norm params in FP16 models)
                ulong wPtr = pW;
                ulong bPtr = pB;
                if (weight.DType == DType.F32)
                {
                    pWCast = CudaMemory.Allocate((nuint)(weight.ElementCount * 2));
                    _kernels!.LaunchCastF32ToF16(pWCast, pW, (int)weight.ElementCount, _stream.Handle);
                    wPtr = pWCast;
                }
                if (bias.DType == DType.F32)
                {
                    pBCast = CudaMemory.Allocate((nuint)(bias.ElementCount * 2));
                    _kernels!.LaunchCastF32ToF16(pBCast, pB, (int)bias.ElementCount, _stream.Handle);
                    bPtr = pBCast;
                }
                _kernels!.LaunchLayerNormF16(
                    pOut, pIn, wPtr, bPtr,
                    normDim, totalRows, eps,
                    _stream.Handle);
            }
            else
            {
                _kernels!.LaunchLayerNorm(
                    pOut, pIn, pW, pB,
                    normDim, totalRows, eps,
                    _stream.Handle);
            }

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pW);
            GpuTransferHelper.FreeDevice(pB);
            if (pWCast != 0) CudaMemory.FreeAsync(pWCast, _stream.Handle);
            if (pBCast != 0) CudaMemory.FreeAsync(pBCast, _stream.Handle);
        }
    }

    public unsafe void RmsNorm(Tensor output, Tensor input, Tensor weight, float eps)
    {
        // CPU fallback — T5 encoding runs once per generation, not a bottleneck.
        // DataPointer access triggers D2H copy for GPU-cached tensors.
        int rank = input.Shape.Rank;
        long lastDim = input.Shape[rank - 1];
        long outerSize = input.ElementCount / lastDim;

        float* pIn = (float*)input.DataPointer;
        float* pOut = (float*)output.DataPointer;
        float* pWeight = (float*)weight.DataPointer;

        for (long outer = 0; outer < outerSize; outer++)
        {
            long baseIdx = outer * lastDim;

            float sumSq = 0f;
            for (long i = 0; i < lastDim; i++)
            {
                float val = pIn[baseIdx + i];
                sumSq += val * val;
            }
            float invRms = 1.0f / MathF.Sqrt(sumSq / lastDim + eps);

            for (long i = 0; i < lastDim; i++)
            {
                pOut[baseIdx + i] = pIn[baseIdx + i] * invRms * pWeight[i];
            }
        }
    }

    /// <summary>Fused GroupNorm + SiLU via single PTX kernel. Eliminates intermediate allocation.</summary>
    public void GroupNormSilu(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps)
    {
        EnsureKernels();

        int batch = (int)input.Shape[0];
        int channels = (int)input.Shape[1];
        int spatial = 1;
        for (int d = 2; d < input.Shape.Rank; d++)
        {
            spatial *= (int)input.Shape[d];
        }

        ulong pOut = 0, pIn = 0, pW = 0, pB = 0, pWCast = 0, pBCast = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(weight);
            pB = GpuTransferHelper.CopyToDevice(bias);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.F16)
            {
                // Cast weight/bias to F16 if stored as F32 (common for norm params in FP16 models)
                ulong wPtr = pW;
                ulong bPtr = pB;
                if (weight.DType == DType.F32)
                {
                    pWCast = CudaMemory.Allocate((nuint)(weight.ElementCount * 2));
                    _kernels!.LaunchCastF32ToF16(pWCast, pW, (int)weight.ElementCount, _stream.Handle);
                    wPtr = pWCast;
                }
                if (bias.DType == DType.F32)
                {
                    pBCast = CudaMemory.Allocate((nuint)(bias.ElementCount * 2));
                    _kernels!.LaunchCastF32ToF16(pBCast, pB, (int)bias.ElementCount, _stream.Handle);
                    bPtr = pBCast;
                }
                _kernels!.LaunchGroupNormSiluF16(
                    pOut, pIn, wPtr, bPtr,
                    batch, channels, spatial, groups, eps,
                    _stream.Handle);
            }
            else
            {
                _kernels!.LaunchGroupNormSilu(
                    pOut, pIn, pW, pB,
                    batch, channels, spatial, groups, eps,
                    _stream.Handle);
            }

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pW);
            GpuTransferHelper.FreeDevice(pB);
            if (pWCast != 0) CudaMemory.FreeAsync(pWCast, _stream.Handle);
            if (pBCast != 0) CudaMemory.FreeAsync(pBCast, _stream.Handle);
        }
    }

    /// <summary>GPU cast FP32 → FP16 via PTX kernel.</summary>
    public void CastToF16(Tensor output, Tensor input)
    {
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            _kernels!.LaunchCastF32ToF16(pOut, pIn, (int)input.ElementCount, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <summary>GPU cast FP16 → FP32 via PTX kernel.</summary>
    public void CastToF32(Tensor output, Tensor input)
    {
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            _kernels!.LaunchCastF16ToF32(pOut, pIn, (int)input.ElementCount, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
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
        bool cachedOutput = false;
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

            bool isF16 = query.DType == DType.F16;
            int elemSize = query.DType.SizeInBytes;

            nuint scoresBytes = (nuint)(totalHeads * Sq * Skv * elemSize);
            scoresBuf = CudaMemory.Allocate(scoresBytes);

            float alpha = scale;
            float beta = 0.0f;

            long strideQ = Sq * D;
            long strideK = Skv * D;
            long strideScores = Sq * Skv;

            // QK^T per head
            for (long bh = 0; bh < totalHeads; bh++)
            {
                ulong qPtr = pQ + (ulong)(bh * strideQ * elemSize);
                ulong kPtr = pK + (ulong)(bh * strideK * elemSize);
                ulong sPtr = scoresBuf + (ulong)(bh * strideScores * elemSize);

                if (isF16)
                {
                    CublasApi.cublasGemmEx(
                        _cublasHandle,
                        CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                        (int)Skv, (int)Sq, (int)D,
                        &alpha,
                        kPtr, CublasApi.CUDA_R_16F, (int)D,
                        qPtr, CublasApi.CUDA_R_16F, (int)D,
                        &beta,
                        sPtr, CublasApi.CUDA_R_16F, (int)Skv,
                        CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
                }
                else
                {
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
                        ulong sPtr = scoresBuf + (ulong)(bh * strideScores * elemSize);
                        if (isF16)
                            _kernels!.LaunchAddF16(sPtr, sPtr, pMask, (int)(Sq * Skv), _stream.Handle);
                        else
                            _kernels!.LaunchAdd(sPtr, sPtr, pMask, (int)(Sq * Skv), _stream.Handle);
                    }
                }
                else if (maskElements == scoreElements)
                {
                    if (isF16)
                        _kernels!.LaunchAddF16(scoresBuf, scoresBuf, pMask, (int)scoreElements, _stream.Handle);
                    else
                        _kernels!.LaunchAdd(scoresBuf, scoresBuf, pMask, (int)scoreElements, _stream.Handle);
                }
            }

            // Softmax
            if (isF16)
                _kernels!.LaunchSoftmaxF16(scoresBuf, (int)Skv, (int)(totalHeads * Sq), _stream.Handle);
            else
                _kernels!.LaunchSoftmax(scoresBuf, (int)Skv, (int)(totalHeads * Sq), _stream.Handle);

            // attn_weights @ V
            long strideV = Skv * D;
            long strideOut = Sq * D;
            float one = 1.0f;
            float zero = 0.0f;

            for (long bh = 0; bh < totalHeads; bh++)
            {
                ulong sPtr = scoresBuf + (ulong)(bh * strideScores * elemSize);
                ulong vPtr = pV + (ulong)(bh * strideV * elemSize);
                ulong oPtr = pOut + (ulong)(bh * strideOut * elemSize);

                if (isF16)
                {
                    CublasApi.cublasGemmEx(
                        _cublasHandle,
                        CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
                        (int)D, (int)Sq, (int)Skv,
                        &one,
                        vPtr, CublasApi.CUDA_R_16F, (int)D,
                        sPtr, CublasApi.CUDA_R_16F, (int)Skv,
                        &zero,
                        oPtr, CublasApi.CUDA_R_16F, (int)D,
                        CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
                }
                else
                {
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

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pQ);
            GpuTransferHelper.FreeDevice(pK);
            GpuTransferHelper.FreeDevice(pV);
            GpuTransferHelper.FreeDevice(pMask);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            if (scoresBuf != 0) CudaMemory.FreeAsync(scoresBuf, _stream.Handle);
        }
    }

    // ── Activations ----------------------------------------------------------

    public void Gelu(Tensor output, Tensor input)
    {
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.F16)
                _kernels!.LaunchGeluF16(pOut, pIn, (int)input.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchGelu(pOut, pIn, (int)input.ElementCount, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    public void Silu(Tensor output, Tensor input)
    {
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.F16)
                _kernels!.LaunchSiluF16(pOut, pIn, (int)input.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchSilu(pOut, pIn, (int)input.ElementCount, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    // ── Element-wise ---------------------------------------------------------

    public void Add(Tensor output, Tensor a, Tensor b)
    {
        EnsureKernels();

        ulong pOut = 0, pA = 0, pB = 0;
        bool cachedOutput = false;
        try
        {
            pA = GpuTransferHelper.CopyToDevice(a);
            pB = GpuTransferHelper.CopyToDevice(b);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (a.DType == DType.F16)
                _kernels!.LaunchAddF16(pOut, pA, pB, (int)a.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchAdd(pOut, pA, pB, (int)a.ElementCount, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pA);
            GpuTransferHelper.FreeDevice(pB);
        }
    }

    public void Mul(Tensor output, Tensor a, Tensor b)
    {
        EnsureKernels();

        ulong pOut = 0, pA = 0, pB = 0;
        bool cachedOutput = false;
        try
        {
            pA = GpuTransferHelper.CopyToDevice(a);
            pB = GpuTransferHelper.CopyToDevice(b);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (a.DType == DType.F16)
                _kernels!.LaunchMulF16(pOut, pA, pB, (int)a.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchMul(pOut, pA, pB, (int)a.ElementCount, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pA);
            GpuTransferHelper.FreeDevice(pB);
        }
    }

    public void Scale(Tensor output, Tensor input, float scalar)
    {
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.F16)
                _kernels!.LaunchScaleF16(pOut, pIn, scalar, (int)input.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchScale(pOut, pIn, scalar, (int)input.ElementCount, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    public void Clamp(Tensor output, Tensor input, float min, float max)
    {
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.F16)
                _kernels!.LaunchClampF16(pOut, pIn, min, max, (int)input.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchClamp(pOut, pIn, min, max, (int)input.ElementCount, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    // ── FP8 Helpers ─────────────────────────────────────────────────────────

    /// <summary>Resolves FP8 dtypes to their compute dtype (F16 on Ampere, native on Ada+). Non-FP8 dtypes pass through unchanged.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private DType ResolveGemmDtype(DType dtype)
    {
        if (dtype.IsFp8) return DType.F16; // Ampere fallback: cast F8→F16 for GEMM
        return dtype;
    }

    /// <summary>Casts GPU data between dtypes using PTX kernels. Handles F8↔F16, F16↔F32 conversions.</summary>
    private void CastOnGpu(ulong output, ulong input, DType srcDtype, DType dstDtype, int count)
    {
        if (srcDtype == dstDtype) return;

        if (srcDtype.IsFp8 && dstDtype == DType.F16)
            _kernels!.LaunchCastF8E4M3ToF16(output, input, count, _stream.Handle);
        else if (srcDtype == DType.F16 && dstDtype.IsFp8)
            _kernels!.LaunchCastF16ToF8E4M3(output, input, count, _stream.Handle);
        else if (srcDtype.IsFp8 && dstDtype == DType.F32)
        {
            // F8 → F16 → F32 (two-step via temp buffer)
            ulong temp = CudaMemory.Allocate((nuint)(count * DType.F16.SizeInBytes));
            _kernels!.LaunchCastF8E4M3ToF16(temp, input, count, _stream.Handle);
            _kernels!.LaunchCastF16ToF32(output, temp, count, _stream.Handle);
            CudaMemory.FreeAsync(temp, _stream.Handle);
        }
        else if (srcDtype == DType.F32 && dstDtype == DType.F16)
            _kernels!.LaunchCastF32ToF16(output, input, count, _stream.Handle);
        else if (srcDtype == DType.F16 && dstDtype == DType.F32)
            _kernels!.LaunchCastF16ToF32(output, input, count, _stream.Handle);
        else if (srcDtype == DType.F32 && dstDtype.IsFp8)
        {
            // F32 → F16 → F8 (two-step via temp buffer)
            ulong temp = CudaMemory.Allocate((nuint)(count * DType.F16.SizeInBytes));
            _kernels!.LaunchCastF32ToF16(temp, input, count, _stream.Handle);
            _kernels!.LaunchCastF16ToF8E4M3(output, temp, count, _stream.Handle);
            CudaMemory.FreeAsync(temp, _stream.Handle);
        }
        else
            throw new NotSupportedException($"GPU cast from {srcDtype} to {dstDtype} not supported.");
    }

    /// <summary>Implements CastF8E4M3ToF16 using the PTX cast kernel on GPU.</summary>
    public void CastF8E4M3ToF16(Tensor output, Tensor input)
    {
        EnsureKernels();
        ulong pIn = 0, pOut = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchCastF8E4M3ToF16(pOut, pIn, (int)input.ElementCount, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pIn);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
        }
    }

    /// <summary>Implements CastF16ToF8E4M3 using the PTX cast kernel on GPU.</summary>
    public void CastF16ToF8E4M3(Tensor output, Tensor input)
    {
        EnsureKernels();
        ulong pIn = 0, pOut = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchCastF16ToF8E4M3(pOut, pIn, (int)input.ElementCount, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pIn);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
        }
    }

    private void EnsureKernels()
    {
        if (_kernels == null)
            throw new InvalidOperationException("PTX kernels not loaded. Provide a ptxDir to the CudaBackend constructor.");
    }

    /// <summary>Synchronizes the default compute stream. Only needed at pipeline boundaries or before explicit D2H.
    /// Per-op sync removed — CUDA guarantees sequential execution on a single blocking stream.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Sync()
    {
        CudaDriverApi.cuStreamSynchronize(_stream.Handle).ThrowOnError();
    }

    // ── Transpose / Permute ---------------------------------------------------

    /// <summary>Batched 2D transpose: [B, D1, D2] -> [B, D2, D1] via PTX kernel.</summary>
    public void Transpose2D(Tensor output, Tensor input, int d1, int d2)
    {
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.F16)
                _kernels!.LaunchTranspose2DF16(pOut, pIn, d1, d2, (int)output.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchTranspose2D(pOut, pIn, d1, d2, (int)output.ElementCount, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <summary>4D permute(0,2,1,3): [B, S, H, D] -> [B, H, S, D] via PTX kernel.</summary>
    public void Permute0213(Tensor output, Tensor input, int s, int h, int d)
    {
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.F16)
                _kernels!.LaunchPermute0213F16(pOut, pIn, s, h, d, (int)output.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchPermute0213(pOut, pIn, s, h, d, (int)output.ElementCount, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <summary>GEGLU activation: output[i] = input[i] * GELU(input[i + outputElements]) via PTX kernel.</summary>
    public void GeGlu(Tensor output, Tensor input)
    {
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            int innerDim = (int)output.Shape[output.Shape.Rank - 1];
            if (input.DType == DType.F16)
                _kernels!.LaunchGeGluF16(pOut, pIn, innerDim, (int)output.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchGeGlu(pOut, pIn, innerDim, (int)output.ElementCount, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <summary>Broadcast add: hidden[b,c,s] += bias[b,c] in-place via PTX kernel.</summary>
    public void BroadcastAdd(Tensor hidden, Tensor bias, int channels, int spatial)
    {
        EnsureKernels();

        ulong pHidden = 0, pBias = 0;
        try
        {
            pHidden = GpuTransferHelper.CopyToDevice(hidden);
            pBias = GpuTransferHelper.CopyToDevice(bias);

            if (hidden.DType == DType.F16)
                _kernels!.LaunchBroadcastAddF16(pHidden, pBias, channels, spatial, (int)hidden.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchBroadcastAdd(pHidden, pBias, channels, spatial, (int)hidden.ElementCount, _stream.Handle);

            // BroadcastAdd modifies hidden in-place. Clear old GPU callbacks before re-caching
            // to prevent CacheActivation's DataPointer access from firing the old sync callback
            // (which would FreeAsync the GPU pointer we're about to re-cache).
            hidden._gpuSyncCallback = null;
            hidden._gpuDisposeCallback = null;
            nuint hiddenBytes = GpuTransferHelper.ByteSize(hidden);
            GpuTransferHelper.CacheActivation(hidden, pHidden, hiddenBytes);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pBias);
        }
    }

    // ── Shape Operations -----------------------------------------------------

    /// <summary>Concatenates tensors along the specified dimension.</summary>
    public unsafe void Concat(Tensor output, ReadOnlySpan<Tensor> inputs, int dim)
    {
        ulong[] gpuInputs = new ulong[inputs.Length];
        ulong pOut = 0;
        bool cachedOutput = false;
        try
        {
            for (int t = 0; t < inputs.Length; t++)
            {
                gpuInputs[t] = GpuTransferHelper.CopyToDevice(inputs[t]);
            }
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            int elemSize = output.DType.SizeInBytes;

            if (dim == 0)
            {
                ulong offset = 0;
                for (int t = 0; t < inputs.Length; t++)
                {
                    nuint byteSize = (nuint)(inputs[t].ElementCount * elemSize);
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
                        nuint sliceBytes = (nuint)(sliceSize * elemSize);

                        long inDimStride = inputDimSize * innerSize;
                        ulong srcOffset = (ulong)((outer * inDimStride) * elemSize);
                        ulong dstOffset = (ulong)((outer * outDimStride + dimOffset) * elemSize);

                        CudaMemory.CopyDeviceToDevice(pOut + dstOffset, gpuInputs[t] + srcOffset, sliceBytes);
                        dimOffset += sliceSize;
                    }
                }
            }

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            for (int t = 0; t < gpuInputs.Length; t++)
            {
                GpuTransferHelper.FreeDevice(gpuInputs[t]);
            }
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
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
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.F16)
                _kernels!.LaunchUpsampleNearest2DF16(
                    pOut, pIn,
                    batch, channels, inH, inW, outH, outW, scaleH, scaleW,
                    _stream.Handle);
            else
                _kernels!.LaunchUpsampleNearest2D(
                    pOut, pIn,
                    batch, channels, inH, inW, outH, outW, scaleH, scaleW,
                    _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
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

    // ── GPU Cache Management -------------------------------------------------

    /// <summary>Preloads weight tensors to GPU memory. Subsequent ops using these tensors skip H2D transfer.</summary>
    public void PreloadWeights(IEnumerable<Tensor> weights)
    {
        foreach (Tensor weight in weights)
        {
            GpuTransferHelper.PreloadWeight(weight);
        }
    }

    /// <summary>Frees specific weight tensors from GPU to reclaim VRAM (e.g., UNet weights before VAE decode).</summary>
    public void FreeWeights(IEnumerable<Tensor> weights)
    {
        GpuTransferHelper.FreeWeights(weights);
    }

    /// <summary>Frees all preloaded weight memory from GPU and clears the cache.</summary>
    public void FreePreloadedWeights()
    {
        GpuTransferHelper.FreeAllCached();
    }

    /// <summary>Evicts all cached GPU weight buffers. Call between pipeline stages to free VRAM.</summary>
    public void EvictGpuCache()
    {
        GpuTransferHelper.EvictAll();
    }

    /// <summary>Returns GPU cache stats: (cachedBytes, hits, misses).</summary>
    public (long cachedBytes, long hits, long misses) GetGpuCacheStats()
    {
        return GpuTransferHelper.GetStats();
    }

    // ── Disposal -------------------------------------------------------------

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;

            GpuTransferHelper.EvictAll();
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
