namespace SharpInference.Cuda;

/// <summary>Loads PTX modules from disk and provides typed kernel launch methods. Function handles stored as nint fields for zero-alloc dispatch.</summary>
public sealed class CudaKernels : IDisposable
{
    private readonly CudaModule _elementwiseModule;
    private readonly CudaModule _groupnormModule;
    private readonly CudaModule _layernormModule;
    private readonly CudaModule _spatialModule;
    private readonly CudaModule _softmaxModule;

    // ── Elementwise kernel function handles ─────────────────────────────
    private readonly nint _addF32;
    private readonly nint _mulF32;
    private readonly nint _scaleF32;
    private readonly nint _siluF32;
    private readonly nint _geluF32;
    private readonly nint _clampF32;

    // ── Normalization kernel function handles ───────────────────────────
    private readonly nint _groupnormF32;
    private readonly nint _layernormF32;

    // ── Spatial kernel function handles ─────────────────────────────────
    private readonly nint _upsampleNearest2dF32;
    private readonly nint _im2colF32;
    private readonly nint _col2biasAddF32;

    // ── Softmax kernel function handle ────────────────────────────────
    private readonly nint _softmaxF32;

    private const uint BlockSize = 256;

    /// <summary>Loads all PTX kernels from the specified directory.</summary>
    public CudaKernels(string ptxDir)
    {
        if (!Directory.Exists(ptxDir))
            throw new DirectoryNotFoundException($"PTX directory not found: {ptxDir}");

        // Elementwise kernels
        string elementwisePath = Path.Combine(ptxDir, "elementwise_f32.ptx");
        _elementwiseModule = CudaModule.LoadFromFile(elementwisePath);
        _addF32 = _elementwiseModule.GetFunction("elementwise_add_f32");
        _mulF32 = _elementwiseModule.GetFunction("elementwise_mul_f32");
        _scaleF32 = _elementwiseModule.GetFunction("elementwise_scale_f32");
        _siluF32 = _elementwiseModule.GetFunction("elementwise_silu_f32");
        _geluF32 = _elementwiseModule.GetFunction("elementwise_gelu_f32");
        _clampF32 = _elementwiseModule.GetFunction("elementwise_clamp_f32");

        // GroupNorm kernel
        string groupnormPath = Path.Combine(ptxDir, "groupnorm_f32.ptx");
        _groupnormModule = CudaModule.LoadFromFile(groupnormPath);
        _groupnormF32 = _groupnormModule.GetFunction("groupnorm_f32");

        // LayerNorm kernel
        string layernormPath = Path.Combine(ptxDir, "layernorm_f32.ptx");
        _layernormModule = CudaModule.LoadFromFile(layernormPath);
        _layernormF32 = _layernormModule.GetFunction("layernorm_f32");

        // Spatial kernels (upsample, im2col, bias add)
        string spatialPath = Path.Combine(ptxDir, "spatial_f32.ptx");
        _spatialModule = CudaModule.LoadFromFile(spatialPath);
        _upsampleNearest2dF32 = _spatialModule.GetFunction("upsample_nearest2d_f32");
        _im2colF32 = _spatialModule.GetFunction("im2col_f32");
        _col2biasAddF32 = _spatialModule.GetFunction("col2bias_add_f32");

        // Softmax kernel
        string softmaxPath = Path.Combine(ptxDir, "softmax_f32.ptx");
        _softmaxModule = CudaModule.LoadFromFile(softmaxPath);
        _softmaxF32 = _softmaxModule.GetFunction("softmax_f32");
    }

    // ── Elementwise Launches ────────────────────────────────────────────

    /// <summary>Launches elementwise add: output[i] = a[i] + b[i]</summary>
    public unsafe void LaunchAdd(ulong output, ulong a, ulong b, int count, nint stream)
    {
        ulong outArg = output, aArg = a, bArg = b;
        uint countArg = (uint)count;

        void** args = stackalloc void*[4];
        args[0] = &outArg;
        args[1] = &aArg;
        args[2] = &bArg;
        args[3] = &countArg;

        uint gridDim = ((uint)count + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            _addF32, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches elementwise multiply: output[i] = a[i] * b[i]</summary>
    public unsafe void LaunchMul(ulong output, ulong a, ulong b, int count, nint stream)
    {
        ulong outArg = output, aArg = a, bArg = b;
        uint countArg = (uint)count;

        void** args = stackalloc void*[4];
        args[0] = &outArg;
        args[1] = &aArg;
        args[2] = &bArg;
        args[3] = &countArg;

        uint gridDim = ((uint)count + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            _mulF32, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches elementwise scale: output[i] = input[i] * scalar</summary>
    public unsafe void LaunchScale(ulong output, ulong input, float scalar, int count, nint stream)
    {
        ulong outArg = output, inArg = input;
        float scalarArg = scalar;
        uint countArg = (uint)count;

        void** args = stackalloc void*[4];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &scalarArg;
        args[3] = &countArg;

        uint gridDim = ((uint)count + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            _scaleF32, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches SiLU activation: output[i] = input[i] * sigmoid(input[i])</summary>
    public unsafe void LaunchSilu(ulong output, ulong input, int count, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint countArg = (uint)count;

        void** args = stackalloc void*[3];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &countArg;

        uint gridDim = ((uint)count + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            _siluF32, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches GELU activation: output[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))</summary>
    public unsafe void LaunchGelu(ulong output, ulong input, int count, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint countArg = (uint)count;

        void** args = stackalloc void*[3];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &countArg;

        uint gridDim = ((uint)count + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            _geluF32, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches elementwise clamp: output[i] = clamp(input[i], min, max)</summary>
    public unsafe void LaunchClamp(ulong output, ulong input, float min, float max, int count, nint stream)
    {
        ulong outArg = output, inArg = input;
        float minArg = min, maxArg = max;
        uint countArg = (uint)count;

        void** args = stackalloc void*[5];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &minArg;
        args[3] = &maxArg;
        args[4] = &countArg;

        uint gridDim = ((uint)count + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            _clampF32, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    // ── Normalization Launches ──────────────────────────────────────────

    /// <summary>Launches GroupNorm: each block handles one (batch, group) pair. Shared memory = blockDim * sizeof(float).</summary>
    public unsafe void LaunchGroupNorm(ulong output, ulong input, ulong weight, ulong bias,
        int batch, int channels, int spatial, int groups, float eps, nint stream)
    {
        ulong outArg = output, inArg = input, wArg = weight, bArg = bias;
        uint batchArg = (uint)batch, chArg = (uint)channels, spatialArg = (uint)spatial, groupsArg = (uint)groups;
        float epsArg = eps;

        void** args = stackalloc void*[9];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &wArg;
        args[3] = &bArg;
        args[4] = &batchArg;
        args[5] = &chArg;
        args[6] = &spatialArg;
        args[7] = &groupsArg;
        args[8] = &epsArg;

        uint gridDim = (uint)(batch * groups);
        uint sharedMem = BlockSize * sizeof(float);
        CudaDriverApi.cuLaunchKernel(
            _groupnormF32, gridDim, 1, 1, BlockSize, 1, 1,
            sharedMem, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches LayerNorm: each block handles one row. Shared memory = blockDim * sizeof(float).</summary>
    public unsafe void LaunchLayerNorm(ulong output, ulong input, ulong weight, ulong bias,
        int normDim, int totalRows, float eps, nint stream)
    {
        ulong outArg = output, inArg = input, wArg = weight, bArg = bias;
        uint normDimArg = (uint)normDim, rowsArg = (uint)totalRows;
        float epsArg = eps;

        void** args = stackalloc void*[7];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &wArg;
        args[3] = &bArg;
        args[4] = &normDimArg;
        args[5] = &rowsArg;
        args[6] = &epsArg;

        uint gridDim = (uint)totalRows;
        uint sharedMem = BlockSize * sizeof(float);
        CudaDriverApi.cuLaunchKernel(
            _layernormF32, gridDim, 1, 1, BlockSize, 1, 1,
            sharedMem, stream, (nint)args, 0).ThrowOnError();
    }

    // ── Spatial Launches ────────────────────────────────────────────────

    /// <summary>Launches UpsampleNearest2D.</summary>
    public unsafe void LaunchUpsampleNearest2D(ulong output, ulong input,
        int batch, int channels, int inH, int inW, int outH, int outW, int scaleH, int scaleW, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint batchArg = (uint)batch, chArg = (uint)channels;
        uint inHArg = (uint)inH, inWArg = (uint)inW;
        uint outHArg = (uint)outH, outWArg = (uint)outW;
        uint scaleHArg = (uint)scaleH, scaleWArg = (uint)scaleW;

        void** args = stackalloc void*[10];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &batchArg;
        args[3] = &chArg;
        args[4] = &inHArg;
        args[5] = &inWArg;
        args[6] = &outHArg;
        args[7] = &outWArg;
        args[8] = &scaleHArg;
        args[9] = &scaleWArg;

        int totalElements = batch * channels * outH * outW;
        uint gridDim = ((uint)totalElements + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            _upsampleNearest2dF32, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches Im2Col for one batch element.</summary>
    public unsafe void LaunchIm2Col(ulong col, ulong input,
        int channels, int inH, int inW, int kH, int kW,
        int padH, int padW, int strideH, int strideW,
        int outH, int outW, int batchOffset, nint stream)
    {
        ulong colArg = col, inArg = input;
        uint chArg = (uint)channels, inHArg = (uint)inH, inWArg = (uint)inW;
        uint kHArg = (uint)kH, kWArg = (uint)kW;
        uint padHArg = (uint)padH, padWArg = (uint)padW;
        uint strHArg = (uint)strideH, strWArg = (uint)strideW;
        uint outHArg = (uint)outH, outWArg = (uint)outW;
        uint batchOffsetArg = (uint)batchOffset;

        void** args = stackalloc void*[14];
        args[0] = &colArg;
        args[1] = &inArg;
        args[2] = &chArg;
        args[3] = &inHArg;
        args[4] = &inWArg;
        args[5] = &kHArg;
        args[6] = &kWArg;
        args[7] = &padHArg;
        args[8] = &padWArg;
        args[9] = &strHArg;
        args[10] = &strWArg;
        args[11] = &outHArg;
        args[12] = &outWArg;
        args[13] = &batchOffsetArg;

        int totalElements = channels * kH * kW * outH * outW;
        uint gridDim = ((uint)totalElements + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            _im2colF32, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches bias addition: output[i] += bias[channel_of(i)]</summary>
    public unsafe void LaunchBiasAdd(ulong output, ulong bias, int outChannels, int spatial, int totalElements, nint stream)
    {
        ulong outArg = output, biasArg = bias;
        uint outChArg = (uint)outChannels, spatialArg = (uint)spatial, totalArg = (uint)totalElements;

        void** args = stackalloc void*[5];
        args[0] = &outArg;
        args[1] = &biasArg;
        args[2] = &outChArg;
        args[3] = &spatialArg;
        args[4] = &totalArg;

        uint gridDim = ((uint)totalElements + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            _col2biasAddF32, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    // ── Softmax Launch ─────────────────────────────────────────────────

    /// <summary>Launches in-place per-row softmax: data[row, col] = softmax(data[row, :]). One block per row, shared memory = blockDim * sizeof(float).</summary>
    public unsafe void LaunchSoftmax(ulong data, int rowLen, int totalRows, nint stream)
    {
        ulong dataArg = data;
        uint rowLenArg = (uint)rowLen;
        uint totalRowsArg = (uint)totalRows;

        void** args = stackalloc void*[3];
        args[0] = &dataArg;
        args[1] = &rowLenArg;
        args[2] = &totalRowsArg;

        uint gridDim = (uint)totalRows;
        uint sharedMem = BlockSize * sizeof(float);
        CudaDriverApi.cuLaunchKernel(
            _softmaxF32, gridDim, 1, 1, BlockSize, 1, 1,
            sharedMem, stream, (nint)args, 0).ThrowOnError();
    }

    public void Dispose()
    {
        _elementwiseModule.Dispose();
        _groupnormModule.Dispose();
        _layernormModule.Dispose();
        _spatialModule.Dispose();
        _softmaxModule.Dispose();
        GC.SuppressFinalize(this);
    }

    ~CudaKernels()
    {
        _elementwiseModule.Dispose();
        _groupnormModule.Dispose();
        _layernormModule.Dispose();
        _spatialModule.Dispose();
        _softmaxModule.Dispose();
    }
}
