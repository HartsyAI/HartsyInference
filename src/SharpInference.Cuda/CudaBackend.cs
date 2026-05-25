using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Cuda.Profiling;

namespace SharpInference.Cuda;

/// <summary>CUDA GPU backend implementing IBackend. Routes operations to cuBLAS SGEMM for matmul and PTX kernels for element-wise/normalization ops. Uses activation caching to keep intermediate results on GPU between ops — lazy sync to CPU on DataPointer access.</summary>
public sealed class CudaBackend : IBackend
{
    private readonly CudaContext _context;
    private readonly CudaStream _stream;
    /// <summary>Side stream used by <see cref="_streamingCache"/> for asynchronous
    /// weight uploads that overlap with compute on <see cref="_stream"/>. Created
    /// non-blocking so it doesn't serialize with the compute stream — synchronization
    /// between the two is explicit via <c>cuEventRecord</c> / <c>cuStreamWaitEvent</c>
    /// inside the streaming cache.</summary>
    private readonly CudaStream _uploadStream;
    private readonly CudaStreamingWeightCache _streamingCache;
    private readonly CudaKernels? _kernels;
    private nint _cublasHandle;
    private Fp8GemmExecutor? _fp8Executor;
    private bool _disposed;

    /// <summary>The device this backend targets.</summary>
    public DeviceKind Device { get; }

    /// <summary>Capabilities of this CUDA backend.</summary>
    public BackendCapabilities Capabilities { get; }

    /// <summary>The CUDA context used by this backend.</summary>
    public CudaContext Context => _context;

    /// <summary>The default compute stream.</summary>
    public CudaStream Stream => _stream;

    /// <summary>The upload stream for asynchronous weight transfers. Exposed for
    /// diagnostics and tests; production callers should use <see cref="StreamingCache"/>.</summary>
    public CudaStream UploadStream => _uploadStream;

    /// <inheritdoc/>
    public IStreamingWeightCache? StreamingCache => _streamingCache;

    /// <summary>The cuBLAS handle for GEMM operations.</summary>
    public nint CublasHandle => _cublasHandle;

    /// <summary>Opt-in flag for the native cuBLASLt FP8 GEMM path on Ada+ (SM 8.9+) GPUs. Defaults to <c>false</c> — on Ampere and below the path is unsupported and the existing cast-to-F16 fallback is correct. The native path is gated on this flag because it has not been end-to-end validated on Ada hardware in CI; flip on after benchmarking against the F16 fallback.</summary>
    public bool EnableNativeFp8Gemm { get; set; }

    /// <summary>Lazily-initialized FP8 GEMM executor. Exposed for diagnostic and benchmarking callers; production GEMM dispatch goes through <see cref="MatMul"/> / <see cref="Linear"/>.</summary>
    public Fp8GemmExecutor Fp8Executor
    {
        get
        {
            _fp8Executor ??= new Fp8GemmExecutor(_context.ComputeCapabilityMajor, _context.ComputeCapabilityMinor);
            return _fp8Executor;
        }
    }

    /// <summary>Creates a CUDA backend for the specified device ordinal. If ptxDir is provided, loads PTX kernels from that directory.</summary>
    public CudaBackend(int deviceOrdinal = 0, string? ptxDir = null)
    {
        _context = new CudaContext(deviceOrdinal);
        // Must use blocking stream (CU_STREAM_DEFAULT) because GpuTransferHelper uses synchronous
        // cuMemcpyHtoD/DtoH which operate on the NULL stream. A non-blocking stream does NOT
        // synchronize with the NULL stream, causing race conditions where kernels read incomplete
        // data from in-progress H2D transfers. Fix: switch to cuMemcpyHtoDAsync on this stream.
        _stream = new CudaStream(nonBlocking: false);
        // Upload stream is non-blocking so its in-flight work doesn't gate the compute
        // stream's NULL-stream "wait for everything" semantics — without that, prefetched
        // uploads would force compute to wait, defeating overlap. The streaming cache uses
        // explicit cuEventRecord/cuStreamWaitEvent for the parts that *do* need to sync.
        _uploadStream = new CudaStream(nonBlocking: true);
        _streamingCache = new CudaStreamingWeightCache(_context, _stream.Handle, _uploadStream.Handle);
        Device = DeviceKind.Cuda(deviceOrdinal);

        // Initialize cuBLAS
        CublasApi.cublasCreate(out _cublasHandle).ThrowOnCublasError();
        CublasApi.cublasSetStream(_cublasHandle, _stream.Handle).ThrowOnCublasError();

        // Give GpuTransferHelper the stream handle for FreeAsync and lazy-sync callbacks
        GpuTransferHelper.SetStream(_stream.Handle);
        // ...and the context, so its lazy callbacks (which fire on whatever thread
        // later reads/disposes a tensor — possibly the GC finalizer thread) can bind
        // the primary context before issuing CUDA Driver API calls.
        GpuTransferHelper.SetContext(_context);
        // ...and the streaming cache, so the OOM retry path can drain its upload
        // stream and trim the device mempool when sync allocs are starved by memory
        // locked up in the stream-ordered allocator pool.
        GpuTransferHelper.SetStreamingCache(_streamingCache);

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
        using NvtxRange _nvtx = NvtxRange.Push("MatMul");
        _context.EnsureCurrent();
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

            // Joint dtype resolution — see ResolveGemmDtype(a, b) docs. Fp8 forces F16.
            DType gemmDtype = ResolveGemmDtype(a.DType, b.DType);
            ulong aPtr = CastIfNeeded(pA, a.DType, gemmDtype, (int)a.ElementCount, out pACast);
            ulong bPtr = CastIfNeeded(pB, b.DType, gemmDtype, (int)b.ElementCount, out pBCast);

            int gemmType = CublasDataType(gemmDtype);
            int cType = output.DType == DType.F16 ? CublasApi.CUDA_R_16F : CublasApi.CUDA_R_32F;

            CublasApi.cublasGemmEx(
                _cublasHandle,
                CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
                N, M, K,
                &alpha,
                bPtr, gemmType, N,
                aPtr, gemmType, K,
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
        using NvtxRange _nvtx = NvtxRange.Push("Linear");
        _context.EnsureCurrent();
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

            // For ComfyUI fp8_scaled checkpoints, every FP8 weight has a per-tensor scalar scale.
            // We store it on the Tensor itself; folding it into cuBLAS' alpha applies the scaling
            // for free during the GEMM (no extra kernel launch). Default Fp8ScaleFactor is 1.0.
            float alpha = weight.Fp8ScaleFactor;
            float beta = 0.0f;

            // Native FP8 GEMM path (Ada/Hopper, opt-in). Both operands must be FP8 and the output
            // F16 to dispatch via cublasLtMatmul. The Ampere fallback below stays the default
            // because the native path has not been end-to-end validated on Ada in CI.
            if (EnableNativeFp8Gemm
                && input.DType.IsFp8 && weight.DType.IsFp8
                && output.DType == DType.F16
                && Fp8Executor.IsSupported)
            {
                Fp8Executor.Run(weight: pWeight, input: pInput, outPtr: pOutput, m: M, n: N, k: K, weightScale: alpha, stream: _stream.Handle);
                if (bias is not null)
                {
                    int totalElementsFp8 = M * N;
                    ulong biasPtr = pBias;
                    if (output.DType != bias!.DType)
                    {
                        pBiasCast = CudaMemory.Allocate((nuint)(bias.ElementCount * output.DType.SizeInBytes));
                        CastOnGpu(pBiasCast, pBias, bias.DType, output.DType, (int)bias.ElementCount);
                        biasPtr = pBiasCast;
                    }
                    _kernels!.LaunchBiasAddF16(pOutput, biasPtr, N, 1, totalElementsFp8, _stream.Handle);
                }
                GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
                cachedOutput = true;
                return;
            }

            // Joint resolution: when fp8 is in play we run the whole GEMM in F16, casting the
            // F32 activation down too. Old behaviour resolved per-operand and ended up at F32,
            // forcing an oversized weight cast (151 MB for proj_mlp) plus a 75 MB intermediate
            // inside CastOnGpu's F8→F32 path.
            DType gemmDtype = ResolveGemmDtype(input.DType, weight.DType);
            ulong inputPtr = CastIfNeeded(pInput, input.DType, gemmDtype, (int)input.ElementCount, out pInputCast);
            ulong weightPtr = CastIfNeeded(pWeight, weight.DType, gemmDtype, (int)weight.ElementCount, out pWeightCast);

            int gemmType = CublasDataType(gemmDtype);
            int outputType = CublasDataType(output.DType);

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
                else if (output.DType == DType.BF16)
                    _kernels!.LaunchBiasAddBf16(pOutput, biasPtr, N, 1, totalElements, _stream.Handle);
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
        using NvtxRange _nvtx = NvtxRange.Push("BatchedMatMul");
        _context.EnsureCurrent();
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

            // Joint dtype resolution — see ResolveGemmDtype(a, b) docs.
            DType gemmDtype = ResolveGemmDtype(a.DType, b.DType);
            ulong aPtr = CastIfNeeded(pA, a.DType, gemmDtype, (int)a.ElementCount, out pACast);
            ulong bPtr = CastIfNeeded(pB, b.DType, gemmDtype, (int)b.ElementCount, out pBCast);

            int gemmType = CublasDataType(gemmDtype);
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
        using NvtxRange _nvtx = NvtxRange.Push("Conv2D");
        _context.EnsureCurrent();
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
        // Joint dtype resolution — fp8 on either side forces a 16-bit GEMM. The im2col
        // buffer matches the GEMM dtype so element size has to be derived from gemmDtype,
        // not the original input dtype.
        DType gemmDtype = ResolveGemmDtype(input.DType, weight.DType);
        int elemSize = gemmDtype.SizeInBytes;
        int outElemSize = output.DType.SizeInBytes;

        ulong pInput = 0, pWeight = 0, pBias = 0, pOutput = 0, colBuf = 0, pInputCast = 0, pWeightCast = 0, pBiasCast = 0;
        bool cachedOutput = false;
        try
        {
            pInput = GpuTransferHelper.CopyToDevice(input);
            pWeight = GpuTransferHelper.CopyToDevice(weight);

            ulong inputPtr = CastIfNeeded(pInput, input.DType, gemmDtype, (int)input.ElementCount, out pInputCast);
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

            ulong weightPtr = CastIfNeeded(pWeight, weight.DType, gemmDtype, (int)weight.ElementCount, out pWeightCast);

            int gemmType = CublasDataType(gemmDtype);
            int gemmOutType = CublasDataType(output.DType);

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
                    if (gemmDtype == DType.F16)
                        _kernels!.LaunchIm2ColF16(
                            colBuf, inputPtr,
                            inCh, inH, inW, kH, kW,
                            padH, padW, strideH, strideW,
                            outH, outW, inputBatchOffset,
                            _stream.Handle);
                    else if (gemmDtype == DType.BF16)
                        _kernels!.LaunchIm2ColBf16(
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

                if (output.DType == DType.F16)
                    _kernels!.LaunchBiasAddF16(
                        pOutput, biasPtr,
                        outCh, outH * outW, totalElements,
                        _stream.Handle);
                else if (output.DType == DType.BF16)
                    _kernels!.LaunchBiasAddBf16(
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
        using NvtxRange _nvtx = NvtxRange.Push("GroupNorm");
        _context.EnsureCurrent();
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
            else if (input.DType == DType.BF16)
            {
                ulong wPtr = pW;
                ulong bPtr = pB;
                if (weight.DType == DType.F32)
                {
                    pWCast = CudaMemory.Allocate((nuint)(weight.ElementCount * 2));
                    _kernels!.LaunchCastF32ToBf16(pWCast, pW, (int)weight.ElementCount, _stream.Handle);
                    wPtr = pWCast;
                }
                if (bias.DType == DType.F32)
                {
                    pBCast = CudaMemory.Allocate((nuint)(bias.ElementCount * 2));
                    _kernels!.LaunchCastF32ToBf16(pBCast, pB, (int)bias.ElementCount, _stream.Handle);
                    bPtr = pBCast;
                }
                _kernels!.LaunchGroupNormBf16(
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
        using NvtxRange _nvtx = NvtxRange.Push("LayerNorm");
        _context.EnsureCurrent();
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
            else if (input.DType == DType.BF16)
            {
                ulong wPtr = pW;
                ulong bPtr = pB;
                if (weight.DType == DType.F32)
                {
                    pWCast = CudaMemory.Allocate((nuint)(weight.ElementCount * 2));
                    _kernels!.LaunchCastF32ToBf16(pWCast, pW, (int)weight.ElementCount, _stream.Handle);
                    wPtr = pWCast;
                }
                if (bias.DType == DType.F32)
                {
                    pBCast = CudaMemory.Allocate((nuint)(bias.ElementCount * 2));
                    _kernels!.LaunchCastF32ToBf16(pBCast, pB, (int)bias.ElementCount, _stream.Handle);
                    bPtr = pBCast;
                }
                _kernels!.LaunchLayerNormBf16(
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
        using NvtxRange _nvtx = NvtxRange.Push("RmsNorm");
        _context.EnsureCurrent(); // DataPointer access below triggers lazy D2H — needs context bound.
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
        using NvtxRange _nvtx = NvtxRange.Push("GroupNormSilu");
        _context.EnsureCurrent();
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
            else if (input.DType == DType.BF16)
            {
                // BF16 path: chosen for SDXL VAE so resnet activations (which exceed
                // F16's 65504 range) stay finite. Weights/biases must match BF16 — cast
                // from F32 if needed. See PHASE_3_DEVIATIONS.md #36 for the F16-overflow
                // pattern; same family of bug, different op site.
                ulong wPtr = pW;
                ulong bPtr = pB;
                if (weight.DType == DType.F32)
                {
                    pWCast = CudaMemory.Allocate((nuint)(weight.ElementCount * 2));
                    _kernels!.LaunchCastF32ToBf16(pWCast, pW, (int)weight.ElementCount, _stream.Handle);
                    wPtr = pWCast;
                }
                if (bias.DType == DType.F32)
                {
                    pBCast = CudaMemory.Allocate((nuint)(bias.ElementCount * 2));
                    _kernels!.LaunchCastF32ToBf16(pBCast, pB, (int)bias.ElementCount, _stream.Handle);
                    bPtr = pBCast;
                }
                _kernels!.LaunchGroupNormSiluBf16(
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
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.BF16)
                _kernels!.LaunchCastBf16ToF32(pOut, pIn, (int)input.ElementCount, _stream.Handle);
            else
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

    /// <summary>GPU cast FP32 â BF16 via PTX kernel. Routes through F32 when input
    /// dtype is anything else (F16, etc.) by chaining the F16âF32 cast first.</summary>
    public void CastToBf16(Tensor output, Tensor input)
    {
        _context.EnsureCurrent();
        EnsureKernels();

        ulong pOut = 0, pIn = 0, pIntermediate = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            ulong srcPtr = pIn;
            if (input.DType == DType.F16)
            {
                pIntermediate = CudaMemory.Allocate((nuint)(input.ElementCount * 4));
                _kernels!.LaunchCastF16ToF32(pIntermediate, pIn, (int)input.ElementCount, _stream.Handle);
                srcPtr = pIntermediate;
            }
            else if (input.DType != DType.F32)
            {
                throw new NotSupportedException($"CastToBf16: source dtype {input.DType} not supported.");
            }

            _kernels!.LaunchCastF32ToBf16(pOut, srcPtr, (int)input.ElementCount, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            if (pIntermediate != 0) CudaMemory.FreeAsync(pIntermediate, _stream.Handle);
        }
    }

    // ── Attention ------------------------------------------------------------

    /// <summary>Scaled dot-product attention via cuBLAS batched GEMM: softmax(Q @ K^T * scale) @ V.</summary>
    public unsafe void ScaledDotProductAttention(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale)
    {
        using NvtxRange _nvtx = NvtxRange.Push("SDPA");
        _context.EnsureCurrent();
        EnsureKernels();

        long B = query.Shape[0];
        long H = query.Shape[1];
        long Sq = query.Shape[2];
        long D = query.Shape[3];
        long Skv = key.Shape[2];

        long totalHeads = B * H;

        ulong pQ = 0, pK = 0, pV = 0, pMask = 0, pOut = 0, scoresBuf = 0;
        ulong pQCast = 0, pKCast = 0, pVCast = 0, pOutCast = 0;
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

            // BF16 path: SDPA's softmax kernel only has F16/F32 variants. For the
            // single VaeAttention call in the SDXL VAE, cast Q/K/V to F32 internally
            // and write the output back as BF16. The cost is one extra ~24 MB of temp
            // F32 (Q+K+V combined for VAE-typical 4096-token attention) — negligible
            // vs the precision cost of trying to squeeze SDXL VAE through F16. A
            // dedicated BF16 SDPA path can be a future optimization once it's hot.
            ulong opQ = pQ, opK = pK, opV = pV, opOut = pOut;
            DType opDtype = query.DType;
            if (query.DType == DType.BF16)
            {
                pQCast = CudaMemory.Allocate((nuint)(query.ElementCount * 4));
                _kernels!.LaunchCastBf16ToF32(pQCast, pQ, (int)query.ElementCount, _stream.Handle);
                pKCast = CudaMemory.Allocate((nuint)(key.ElementCount * 4));
                _kernels!.LaunchCastBf16ToF32(pKCast, pK, (int)key.ElementCount, _stream.Handle);
                pVCast = CudaMemory.Allocate((nuint)(value.ElementCount * 4));
                _kernels!.LaunchCastBf16ToF32(pVCast, pV, (int)value.ElementCount, _stream.Handle);
                pOutCast = CudaMemory.Allocate((nuint)(output.ElementCount * 4));
                opQ = pQCast;
                opK = pKCast;
                opV = pVCast;
                opOut = pOutCast;
                opDtype = DType.F32;
            }

            bool isF16 = opDtype == DType.F16;
            int elemSize = opDtype.SizeInBytes;

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
                ulong qPtr = opQ + (ulong)(bh * strideQ * elemSize);
                ulong kPtr = opK + (ulong)(bh * strideK * elemSize);
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
                ulong vPtr = opV + (ulong)(bh * strideV * elemSize);
                ulong oPtr = opOut + (ulong)(bh * strideOut * elemSize);

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

            // If we did the BF16 internal-cast detour, the output is F32 in pOutCast — cast
            // it back to BF16 in pOut before caching.
            if (output.DType == DType.BF16 && pOutCast != 0)
            {
                _kernels!.LaunchCastF32ToBf16(pOut, pOutCast, (int)output.ElementCount, _stream.Handle);
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
            if (pQCast != 0) CudaMemory.FreeAsync(pQCast, _stream.Handle);
            if (pKCast != 0) CudaMemory.FreeAsync(pKCast, _stream.Handle);
            if (pVCast != 0) CudaMemory.FreeAsync(pVCast, _stream.Handle);
            if (pOutCast != 0) CudaMemory.FreeAsync(pOutCast, _stream.Handle);
        }
    }

    // ── Activations ----------------------------------------------------------

    public void Gelu(Tensor output, Tensor input)
    {
        _context.EnsureCurrent();
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

    public void Sigmoid(Tensor output, Tensor input)
    {
        // Sigmoid / Tanh / Snake PTX kernels and Conv1d / ConvTranspose1d PTX kernels
        // are not yet implemented. Audio models that use any of these (every codec in
        // PHASE_5_AUDIO §4) currently must run those layers on CpuBackend; the rest of
        // the pipeline (attention, matmul, cast) can stay on GPU. PTX work tracked under
        // Phase 5 §3 "PTX kernels" in PHASE_5_AUDIO.md (conv_transpose1d.ptx +
        // snake_activation.ptx are explicitly listed).
        throw new NotSupportedException("CUDA Sigmoid not yet implemented — use CpuBackend for LSTM activations.");
    }

    public void Tanh(Tensor output, Tensor input)
    {
        throw new NotSupportedException("CUDA Tanh not yet implemented — use CpuBackend for LSTM activations.");
    }

    public void Elu(Tensor output, Tensor input, float alpha)
    {
        throw new NotSupportedException("CUDA Elu not yet implemented — use CpuBackend for SEANet codec models.");
    }

    public void Snake(Tensor output, Tensor input, Tensor alpha, Tensor? beta)
    {
        throw new NotSupportedException("CUDA Snake not yet implemented — use CpuBackend for snake-using vocoders.");
    }

    public void Conv1d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int stride, int padLeft, int padRight, int dilation, int groups)
    {
        throw new NotSupportedException("CUDA Conv1d not yet implemented — use CpuBackend for codec models.");
    }

    public void ConvTranspose1d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int stride, int padLeft, int padRight, int dilation)
    {
        throw new NotSupportedException("CUDA ConvTranspose1d not yet implemented — use CpuBackend for codec models.");
    }

    public void Silu(Tensor output, Tensor input)
    {
        _context.EnsureCurrent();
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
            else if (input.DType == DType.BF16)
                _kernels!.LaunchSiluBf16(pOut, pIn, (int)input.ElementCount, _stream.Handle);
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
        _context.EnsureCurrent();
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
            else if (a.DType == DType.BF16)
                _kernels!.LaunchAddBf16(pOut, pA, pB, (int)a.ElementCount, _stream.Handle);
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
        _context.EnsureCurrent();
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
            else if (a.DType == DType.BF16)
                _kernels!.LaunchMulBf16(pOut, pA, pB, (int)a.ElementCount, _stream.Handle);
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
        _context.EnsureCurrent();
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
            else if (input.DType == DType.BF16)
                _kernels!.LaunchScaleBf16(pOut, pIn, scalar, (int)input.ElementCount, _stream.Handle);
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
        _context.EnsureCurrent();
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

    /// <summary>Resolves the dtype a single operand will end up at after fp8 → F16 fallback. Kept for callers (e.g. Conv2D's im2col elemSize) that need the per-operand answer without the joint-dtype rule; new code should prefer the two-operand overload.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private DType ResolveGemmDtype(DType dtype)
    {
        if (dtype.IsFp8) return DType.F16; // Ampere fallback: cast F8→F16 for GEMM
        return dtype;
    }

    /// <summary>Maps a SharpInference dtype to its cuBLAS data-type constant for use with
    /// <c>cublasGemmEx</c> / <c>cublasGemmStridedBatchedEx</c>. Handles F16, BF16, and F32;
    /// throws on anything else (FP8 should be cast to F16/BF16 via <see cref="CastIfNeeded"/>
    /// before reaching cuBLAS).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int CublasDataType(DType dtype)
    {
        if (dtype == DType.F16) return CublasApi.CUDA_R_16F;
        if (dtype == DType.BF16) return CublasApi.CUDA_R_16BF;
        if (dtype == DType.F32) return CublasApi.CUDA_R_32F;
        throw new NotSupportedException($"cuBLAS GEMM does not support dtype {dtype}.");
    }

    /// <summary>Resolves the GEMM compute dtype for an op with two operands. Prefer F16 over F32 whenever either operand is F16 (or fp8, which casts to F16). cublasGemmEx COMPUTE_32F does not support F32×F32→F16, so when an F16 activation feeds an F16 output and the weight happens to be F32, the F32 weight must cast down to F16 — F16 also gets Tensor Core acceleration on Ampere+.</summary>
    private DType ResolveGemmDtype(DType a, DType b)
    {
        // FP8 forces a 16-bit GEMM (Ampere has fast Tensor Cores in F16/BF16, not F32). Pick:
        //  - BF16 when the other operand is F32 — BF16 has F32's full dynamic range, so the
        //    F32→16-bit activation cast cannot produce ±Inf even when SwiGLU's `gated`
        //    intermediate momentarily exceeds 65504 in F32. F16 *would* overflow there
        //    (Z-Image L0 ffnOut had INF=9024 at step 1 of CUDA bring-up before this fix).
        //  - F16 otherwise — keeps the existing F16 fast path that Flux/SDXL FP8 paths rely on
        //    when their activations are already F16 (and therefore in-range).
        if (a.IsFp8 || b.IsFp8)
        {
            return (a == DType.F32 || b == DType.F32) ? DType.BF16 : DType.F16;
        }
        // GGUF quants always dequantize to F16 (or BF16 if the other operand is F32). The
        // dequant kernels emit F16 directly; routing through F32 would force an extra F16→F32
        // cast pass for no benefit. Same precedence rule as FP8 above.
        if (a.IsQuantized || b.IsQuantized)
        {
            return (a == DType.F32 || b == DType.F32) ? DType.BF16 : DType.F16;
        }
        if (a == DType.F16 || b == DType.F16) return DType.F16;
        if (a == DType.BF16 || b == DType.BF16) return DType.BF16;
        return a == DType.F32 || b == DType.F32 ? DType.F32 : a;
    }

    /// <summary>Ensures a GPU buffer holding a tensor of <paramref name="srcDtype"/> is
    /// available in <paramref name="dstDtype"/>. Returns the existing pointer if no cast
    /// is needed, or allocates + casts and writes the new dptr to <paramref name="castOut"/>
    /// (which the caller is responsible for freeing with <c>cuMemFreeAsync</c>). Hides the
    /// F8 special case so the four GEMM call sites all look the same.</summary>
    private unsafe ulong CastIfNeeded(ulong srcPtr, DType srcDtype, DType dstDtype, int elementCount, out ulong castOut)
    {
        if (srcDtype == dstDtype)
        {
            castOut = 0;
            return srcPtr;
        }
        castOut = CudaMemory.Allocate((nuint)(elementCount * dstDtype.SizeInBytes));
        CastOnGpu(castOut, srcPtr, srcDtype, dstDtype, elementCount);
        return castOut;
    }

    /// <summary>Casts GPU data between dtypes using PTX kernels. Handles F8↔F16, F16↔F32 conversions, and GGUF quantized → F16/F32 dequant (Q8_0 / Q4_K / Q5_K / Q6_K).</summary>
    private void CastOnGpu(ulong output, ulong input, DType srcDtype, DType dstDtype, int count)
    {
        if (srcDtype == dstDtype) return;

        // ── GGUF dequant paths. F16 is the kernel's native output. F32 and BF16 stage through F16. ──
        if (srcDtype.IsQuantized && dstDtype == DType.F16)
        {
            LaunchGgufDequantToF16(output, input, srcDtype, count);
            return;
        }
        if (srcDtype.IsQuantized && dstDtype == DType.F32)
        {
            ulong tempF16 = CudaMemory.Allocate((nuint)(count * DType.F16.SizeInBytes));
            try
            {
                LaunchGgufDequantToF16(tempF16, input, srcDtype, count);
                _kernels!.LaunchCastF16ToF32(output, tempF16, count, _stream.Handle);
            }
            finally
            {
                CudaMemory.FreeAsync(tempF16, _stream.Handle);
            }
            return;
        }
        if (srcDtype.IsQuantized && dstDtype == DType.BF16)
        {
            // quant → F16 → F32 → BF16. F32 staging needed because BF16 conversion goes through F32 in our kernel set.
            ulong tempF16 = CudaMemory.Allocate((nuint)(count * DType.F16.SizeInBytes));
            ulong tempF32 = CudaMemory.Allocate((nuint)(count * DType.F32.SizeInBytes));
            try
            {
                LaunchGgufDequantToF16(tempF16, input, srcDtype, count);
                _kernels!.LaunchCastF16ToF32(tempF32, tempF16, count, _stream.Handle);
                _kernels!.LaunchCastF32ToBf16(output, tempF32, count, _stream.Handle);
            }
            finally
            {
                CudaMemory.FreeAsync(tempF16, _stream.Handle);
                CudaMemory.FreeAsync(tempF32, _stream.Handle);
            }
            return;
        }

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
        else if (srcDtype.IsFp8 && dstDtype == DType.BF16)
        {
            // F8 → F32 → BF16 (the values FP8 represents are within F16 range, so we could go
            // F8→F16 first, but the F16→BF16 path also goes via F32; folding them avoids a
            // redundant intermediate). FP8 max ≈ 448, well within BF16's range.
            ulong temp = CudaMemory.Allocate((nuint)(count * DType.F32.SizeInBytes));
            // F8 → F32 (re-uses the two-step F8→F16→F32 ladder via recursion).
            CastOnGpu(temp, input, srcDtype, DType.F32, count);
            _kernels!.LaunchCastF32ToBf16(output, temp, count, _stream.Handle);
            CudaMemory.FreeAsync(temp, _stream.Handle);
        }
        else if (srcDtype == DType.BF16 && dstDtype.IsFp8)
        {
            // BF16 → F32 → F16 → F8. BF16 represents values up to 3.4e38; FP8's max is 448.
            // Going through F32 then F16 catches saturation at the F16 stage (which clips to ±Inf,
            // then the F16→F8 stage maps Inf to FP8's NaN encoding — so over-range values are
            // marked rather than wrapping silently).
            ulong temp32 = CudaMemory.Allocate((nuint)(count * DType.F32.SizeInBytes));
            ulong temp16 = CudaMemory.Allocate((nuint)(count * DType.F16.SizeInBytes));
            _kernels!.LaunchCastBf16ToF32(temp32, input, count, _stream.Handle);
            _kernels!.LaunchCastF32ToF16(temp16, temp32, count, _stream.Handle);
            _kernels!.LaunchCastF16ToF8E4M3(output, temp16, count, _stream.Handle);
            CudaMemory.FreeAsync(temp32, _stream.Handle);
            CudaMemory.FreeAsync(temp16, _stream.Handle);
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
        else if (srcDtype == DType.BF16 && dstDtype == DType.F32)
            _kernels!.LaunchCastBf16ToF32(output, input, count, _stream.Handle);
        else if (srcDtype == DType.F32 && dstDtype == DType.BF16)
            _kernels!.LaunchCastF32ToBf16(output, input, count, _stream.Handle);
        else if (srcDtype == DType.BF16 && dstDtype == DType.F16)
        {
            // BF16 → F32 → F16 (lossy via temp F32 buffer)
            ulong temp = CudaMemory.Allocate((nuint)(count * DType.F32.SizeInBytes));
            _kernels!.LaunchCastBf16ToF32(temp, input, count, _stream.Handle);
            _kernels!.LaunchCastF32ToF16(output, temp, count, _stream.Handle);
            CudaMemory.FreeAsync(temp, _stream.Handle);
        }
        else if (srcDtype == DType.F16 && dstDtype == DType.BF16)
        {
            // F16 → F32 → BF16 (round-trip via F32)
            ulong temp = CudaMemory.Allocate((nuint)(count * DType.F32.SizeInBytes));
            _kernels!.LaunchCastF16ToF32(temp, input, count, _stream.Handle);
            _kernels!.LaunchCastF32ToBf16(output, temp, count, _stream.Handle);
            CudaMemory.FreeAsync(temp, _stream.Handle);
        }
        else
            throw new NotSupportedException($"GPU cast from {srcDtype} to {dstDtype} not supported.");
    }

    /// <summary>Dispatches the right per-DType GGUF dequant kernel. Element count must respect the block size of the source dtype (32 for Q8_0, 256 for Q*_K).</summary>
    private void LaunchGgufDequantToF16(ulong output, ulong input, DType srcDtype, int count)
    {
        if (srcDtype == DType.Q8_0)
            _kernels!.LaunchDequantQ8_0ToF16(output, input, count, _stream.Handle);
        else if (srcDtype == DType.Q4_K)
            _kernels!.LaunchDequantQ4_KToF16(output, input, count, _stream.Handle);
        else if (srcDtype == DType.Q5_K)
            _kernels!.LaunchDequantQ5_KToF16(output, input, count, _stream.Handle);
        else if (srcDtype == DType.Q6_K)
            _kernels!.LaunchDequantQ6_KToF16(output, input, count, _stream.Handle);
        else
            throw new NotSupportedException($"GPU dequant for {srcDtype} not yet implemented. Supported: Q8_0, Q4_K, Q5_K, Q6_K. Use CPU dequant via GgufDequantizer for other GGUF types.");
    }

    /// <summary>Implements CastF8E4M3ToF16 using the PTX cast kernel on GPU.</summary>
    public void CastF8E4M3ToF16(Tensor output, Tensor input)
    {
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
        CudaDriverApi.cuStreamSynchronize(_stream.Handle).ThrowOnError();
    }

    // ── Transpose / Permute ---------------------------------------------------

    /// <summary>Batched 2D transpose: [B, D1, D2] -> [B, D2, D1] via PTX kernel.</summary>
    public void Transpose2D(Tensor output, Tensor input, int d1, int d2)
    {
        _context.EnsureCurrent();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            // BF16 piggy-backs on the F16 kernel — both are pure 16-bit byte shuffles
            // (no math, no precision concern), so the same kernel produces correct output.
            if (input.DType == DType.F16 || input.DType == DType.BF16)
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
        _context.EnsureCurrent();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            // BF16 piggy-backs on the F16 kernel (pure 16-bit byte shuffle, see Transpose2D).
            if (input.DType == DType.F16 || input.DType == DType.BF16)
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
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
        EnsureKernels();

        ulong pHidden = 0, pBias = 0;
        try
        {
            pHidden = GpuTransferHelper.CopyToDevice(hidden);
            pBias = GpuTransferHelper.CopyToDevice(bias);

            if (hidden.DType == DType.F16)
                _kernels!.LaunchBroadcastAddF16(pHidden, pBias, channels, spatial, (int)hidden.ElementCount, _stream.Handle);
            else if (hidden.DType == DType.BF16)
                _kernels!.LaunchBroadcastAddBf16(pHidden, pBias, channels, spatial, (int)hidden.ElementCount, _stream.Handle);
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
        _context.EnsureCurrent();
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

    /// <summary>Splits <paramref name="input"/> into <paramref name="outputs"/> along <paramref name="dim"/>. Delegates to the CPU kernel — Split is pure memcpy and rare in our pipelines (one call per VAE encode), so the GPU round-trip is acceptable. TODO: GPU-native Split via cuMemcpyDtoDAsync.</summary>
    public void Split(ReadOnlySpan<Tensor> outputs, Tensor input, int dim)
    {
        SharpInference.Cpu.Kernels.ElementWiseKernels.Split(outputs, input, dim);
    }

    // ── Sampling -------------------------------------------------------------

    public void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW)
    {
        _context.EnsureCurrent();
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
            else if (input.DType == DType.BF16)
                _kernels!.LaunchUpsampleNearest2DBf16(
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
        _context.EnsureCurrent();
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
        // CPU-side fill — DataPointer access syncs the GPU copy out (if cached) and
        // disposes its dptr, so the next op will re-upload from the just-written CPU
        // buffer. Dtype-aware so VAE F16 codepaths can use this for shift/scale broadcasts.
        if (tensor.DType == DType.F16)
        {
            Half* ptr = (Half*)tensor.DataPointer;
            Half h = (Half)value;
            for (long i = 0; i < tensor.ElementCount; i++) ptr[i] = h;
        }
        else if (tensor.DType == DType.BF16)
        {
            // BF16 = upper 16 bits of F32. Truncate via right-shift (RTNE not needed
            // for typical fill values; if `value` lands exactly between two BF16 grid
            // points the trunc bias is acceptable for init scalars).
            ushort* ptr = (ushort*)tensor.DataPointer;
            uint bits = *(uint*)&value;
            ushort bf = (ushort)(bits >> 16);
            for (long i = 0; i < tensor.ElementCount; i++) ptr[i] = bf;
        }
        else if (tensor.DType == DType.F32)
        {
            float* ptr = (float*)tensor.DataPointer;
            for (long i = 0; i < tensor.ElementCount; i++) ptr[i] = value;
        }
        else
        {
            throw new NotSupportedException($"Fill not supported for dtype {tensor.DType}");
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
        _context.EnsureCurrent();
        foreach (Tensor weight in weights)
        {
            GpuTransferHelper.PreloadWeight(weight);
        }
    }

    /// <summary>Frees specific weight tensors from GPU to reclaim VRAM (e.g., UNet weights before VAE decode).</summary>
    public void FreeWeights(IEnumerable<Tensor> weights)
    {
        _context.EnsureCurrent();
        GpuTransferHelper.FreeWeights(weights);
    }

    /// <summary>Frees all preloaded weight memory from GPU and clears the cache.</summary>
    public void FreePreloadedWeights()
    {
        _context.EnsureCurrent();
        GpuTransferHelper.FreeAllCached();
    }

    /// <summary>Evicts all cached GPU weight buffers. Call between pipeline stages to free VRAM.</summary>
    public void EvictGpuCache()
    {
        _context.EnsureCurrent();
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

            // Bind context on the disposing thread so cuMemFree / cublasDestroy /
            // cuStreamDestroy don't hit CUDA_ERROR_INVALID_CONTEXT. Disposal can run
            // on the finalizer thread or a different worker than constructed us.
            _context.EnsureCurrent();

            GpuTransferHelper.EvictAll();
            _kernels?.Dispose();

            if (_fp8Executor is not null)
            {
                _fp8Executor.Dispose();
                _fp8Executor = null;
            }

            if (_cublasHandle != 0)
            {
                CublasApi.cublasDestroy(_cublasHandle);
                _cublasHandle = 0;
            }

            // Order: upload stream first (no other code holds events on it after
            // EvictAll above), then compute stream, then context.
            _uploadStream.Dispose();
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
