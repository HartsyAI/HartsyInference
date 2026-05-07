namespace SharpInference.Cuda;

/// <summary>Loads PTX modules from disk and provides typed kernel launch methods. Function handles stored as nint fields for zero-alloc dispatch.</summary>
public sealed class CudaKernels : IDisposable
{
    // ── F32 Modules ──────────────────────────────────────────────────────
    private readonly CudaModule _elementwiseModule;
    private readonly CudaModule _groupnormModule;
    private readonly CudaModule _layernormModule;
    private readonly CudaModule _spatialModule;
    private readonly CudaModule _softmaxModule;
    private readonly CudaModule _transposeModule;
    private readonly CudaModule _gegluModule;
    private readonly CudaModule _broadcastAddModule;

    // ── F16 Modules ──────────────────────────────────────────────────────
    private readonly CudaModule _elementwiseF16Module;
    private readonly CudaModule _groupnormF16Module;
    private readonly CudaModule _layernormF16Module;
    private readonly CudaModule _spatialF16Module;
    private readonly CudaModule _softmaxF16Module;
    private readonly CudaModule _transposeF16Module;
    private readonly CudaModule _gegluF16Module;
    private readonly CudaModule _broadcastAddF16Module;

    // ── Fused + Cast Modules ─────────────────────────────────────────────
    private readonly CudaModule _groupnormSiluModule;
    private readonly CudaModule _groupnormSiluF16Module;
    private readonly CudaModule _castModule;

    // ── Elementwise F32 function handles ─────────────────────────────────
    private readonly nint _addF32;
    private readonly nint _mulF32;
    private readonly nint _scaleF32;
    private readonly nint _siluF32;
    private readonly nint _geluF32;
    private readonly nint _clampF32;

    // ── Elementwise F16 function handles ─────────────────────────────────
    private readonly nint _addF16;
    private readonly nint _mulF16;
    private readonly nint _scaleF16;
    private readonly nint _siluF16;
    private readonly nint _geluF16;
    private readonly nint _clampF16;

    // ── Normalization function handles ───────────────────────────────────
    private readonly nint _groupnormF32;
    private readonly nint _groupnormF16;
    private readonly nint _layernormF32;
    private readonly nint _layernormF16;

    // ── Spatial function handles ─────────────────────────────────────────
    private readonly nint _upsampleNearest2dF32;
    private readonly nint _im2colF32;
    private readonly nint _col2biasAddF32;
    private readonly nint _upsampleNearest2dF16;
    private readonly nint _im2colF16;
    private readonly nint _col2biasAddF16;

    // ── Softmax function handles ─────────────────────────────────────────
    private readonly nint _softmaxF32;
    private readonly nint _softmaxF16;

    // ── Transpose/Permute function handles ───────────────────────────────
    private readonly nint _transpose2dF32;
    private readonly nint _permute0213F32;
    private readonly nint _transpose2dF16;
    private readonly nint _permute0213F16;

    // ── GeGlu function handles ───────────────────────────────────────────
    private readonly nint _gegluF32;
    private readonly nint _gegluF16;

    // ── BroadcastAdd function handles ────────────────────────────────────
    private readonly nint _broadcastAddF32;
    private readonly nint _broadcastAddF16;

    // ── Fused GroupNorm+SiLU function handles ────────────────────────────
    private readonly nint _groupnormSiluF32;
    private readonly nint _groupnormSiluF16;

    // ── Cast function handles ────────────────────────────────────────────
    private readonly nint _castF32ToF16;
    private readonly nint _castF16ToF32;

    // ── FP8 Cast Modules + Handles ────────────────────────────────────────
    private readonly CudaModule _castF8Module;
    private readonly nint _castF8E4M3ToF16;

    private readonly CudaModule _castBf16Module;
    private readonly nint _castBf16ToF32;
    private readonly nint _castF32ToBf16;
    private readonly nint _castF16ToF8E4M3;

    // ── GGUF Dequant Modules + Handles ───────────────────────────────────
    private readonly CudaModule _dequantQ8_0Module;
    private readonly nint _dequantQ8_0ToF16;
    private readonly CudaModule _dequantQ4_KModule;
    private readonly nint _dequantQ4_KToF16;
    private readonly CudaModule _dequantQ5_KModule;
    private readonly nint _dequantQ5_KToF16;
    private readonly CudaModule _dequantQ6_KModule;
    private readonly nint _dequantQ6_KToF16;

    private const uint BlockSize = 256;

    /// <summary>Loads all PTX kernels from the specified directory.</summary>
    public CudaKernels(string ptxDir)
    {
        if (!Directory.Exists(ptxDir))
            throw new DirectoryNotFoundException($"PTX directory not found: {ptxDir}");

        // ── F32 modules ──────────────────────────────────────────────────
        _elementwiseModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "elementwise_f32.ptx"));
        _addF32 = _elementwiseModule.GetFunction("elementwise_add_f32");
        _mulF32 = _elementwiseModule.GetFunction("elementwise_mul_f32");
        _scaleF32 = _elementwiseModule.GetFunction("elementwise_scale_f32");
        _siluF32 = _elementwiseModule.GetFunction("elementwise_silu_f32");
        _geluF32 = _elementwiseModule.GetFunction("elementwise_gelu_f32");
        _clampF32 = _elementwiseModule.GetFunction("elementwise_clamp_f32");

        _groupnormModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "groupnorm_f32.ptx"));
        _groupnormF32 = _groupnormModule.GetFunction("groupnorm_f32");

        _layernormModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "layernorm_f32.ptx"));
        _layernormF32 = _layernormModule.GetFunction("layernorm_f32");

        _spatialModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "spatial_f32.ptx"));
        _upsampleNearest2dF32 = _spatialModule.GetFunction("upsample_nearest2d_f32");
        _im2colF32 = _spatialModule.GetFunction("im2col_f32");
        _col2biasAddF32 = _spatialModule.GetFunction("col2bias_add_f32");

        _softmaxModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "softmax_f32.ptx"));
        _softmaxF32 = _softmaxModule.GetFunction("softmax_f32");

        _transposeModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "transpose_f32.ptx"));
        _transpose2dF32 = _transposeModule.GetFunction("transpose_2d_f32");
        _permute0213F32 = _transposeModule.GetFunction("permute_0213_f32");

        _gegluModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "geglu_f32.ptx"));
        _gegluF32 = _gegluModule.GetFunction("geglu_f32");

        _broadcastAddModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "broadcast_add_f32.ptx"));
        _broadcastAddF32 = _broadcastAddModule.GetFunction("broadcast_add_f32");

        // ── F16 modules ──────────────────────────────────────────────────
        _elementwiseF16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "elementwise_f16.ptx"));
        _addF16 = _elementwiseF16Module.GetFunction("elementwise_add_f16");
        _mulF16 = _elementwiseF16Module.GetFunction("elementwise_mul_f16");
        _scaleF16 = _elementwiseF16Module.GetFunction("elementwise_scale_f16");
        _siluF16 = _elementwiseF16Module.GetFunction("elementwise_silu_f16");
        _geluF16 = _elementwiseF16Module.GetFunction("elementwise_gelu_f16");
        _clampF16 = _elementwiseF16Module.GetFunction("elementwise_clamp_f16");

        _groupnormF16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "groupnorm_f16.ptx"));
        _groupnormF16 = _groupnormF16Module.GetFunction("groupnorm_f16");

        _layernormF16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "layernorm_f16.ptx"));
        _layernormF16 = _layernormF16Module.GetFunction("layernorm_f16");

        _spatialF16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "spatial_f16.ptx"));
        _upsampleNearest2dF16 = _spatialF16Module.GetFunction("upsample_nearest2d_f16");
        _im2colF16 = _spatialF16Module.GetFunction("im2col_f16");
        _col2biasAddF16 = _spatialF16Module.GetFunction("col2bias_add_f16");

        _softmaxF16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "softmax_f16.ptx"));
        _softmaxF16 = _softmaxF16Module.GetFunction("softmax_f16");

        _transposeF16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "transpose_f16.ptx"));
        _transpose2dF16 = _transposeF16Module.GetFunction("transpose_2d_f16");
        _permute0213F16 = _transposeF16Module.GetFunction("permute_0213_f16");

        _gegluF16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "geglu_f16.ptx"));
        _gegluF16 = _gegluF16Module.GetFunction("geglu_f16");

        _broadcastAddF16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "broadcast_add_f16.ptx"));
        _broadcastAddF16 = _broadcastAddF16Module.GetFunction("broadcast_add_f16");

        // ── Fused GroupNorm+SiLU ─────────────────────────────────────────
        _groupnormSiluModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "groupnorm_silu_f32.ptx"));
        _groupnormSiluF32 = _groupnormSiluModule.GetFunction("groupnorm_silu_f32");

        _groupnormSiluF16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "groupnorm_silu_f16.ptx"));
        _groupnormSiluF16 = _groupnormSiluF16Module.GetFunction("groupnorm_silu_f16");

        // ── Cast ─────────────────────────────────────────────────────────
        _castModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "cast_f32_f16.ptx"));
        _castF32ToF16 = _castModule.GetFunction("cast_f32_to_f16");
        _castF16ToF32 = _castModule.GetFunction("cast_f16_to_f32");

        // ── FP8 Cast ─────────────────────────────────────────────────────
        _castF8Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "cast_f8e4m3_f16.ptx"));
        _castF8E4M3ToF16 = _castF8Module.GetFunction("cast_f8e4m3_to_f16");
        _castF16ToF8E4M3 = _castF8Module.GetFunction("cast_f16_to_f8e4m3");

        // ── BF16 <-> F32 Cast ───────────────────────────────────────────
        _castBf16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "cast_bf16_f32.ptx"));
        _castBf16ToF32 = _castBf16Module.GetFunction("cast_bf16_to_f32");
        _castF32ToBf16 = _castBf16Module.GetFunction("cast_f32_to_bf16");

        // ── GGUF Dequant ─────────────────────────────────────────────────
        _dequantQ8_0Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "dequant_q8_0_to_f16.ptx"));
        _dequantQ8_0ToF16 = _dequantQ8_0Module.GetFunction("dequant_q8_0_to_f16");
        _dequantQ4_KModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "dequant_q4_k_to_f16.ptx"));
        _dequantQ4_KToF16 = _dequantQ4_KModule.GetFunction("dequant_q4_k_to_f16");
        _dequantQ5_KModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "dequant_q5_k_to_f16.ptx"));
        _dequantQ5_KToF16 = _dequantQ5_KModule.GetFunction("dequant_q5_k_to_f16");
        _dequantQ6_KModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "dequant_q6_k_to_f16.ptx"));
        _dequantQ6_KToF16 = _dequantQ6_KModule.GetFunction("dequant_q6_k_to_f16");
    }

    // ── Private Launch Helpers ───────────────────────────────────────────

    private unsafe void LaunchBinaryImpl(nint func, ulong output, ulong a, ulong b, int count, nint stream)
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
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchUnaryImpl(nint func, ulong output, ulong input, int count, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint countArg = (uint)count;

        void** args = stackalloc void*[3];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &countArg;

        uint gridDim = ((uint)count + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchScaleImpl(nint func, ulong output, ulong input, float scalar, int count, nint stream)
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
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchClampImpl(nint func, ulong output, ulong input, float min, float max, int count, nint stream)
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
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchGroupNormImpl(nint func, ulong output, ulong input, ulong weight, ulong bias,
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
            func, gridDim, 1, 1, BlockSize, 1, 1,
            sharedMem, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchLayerNormImpl(nint func, ulong output, ulong input, ulong weight, ulong bias,
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
            func, gridDim, 1, 1, BlockSize, 1, 1,
            sharedMem, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchSoftmaxImpl(nint func, ulong data, int rowLen, int totalRows, nint stream)
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
            func, gridDim, 1, 1, BlockSize, 1, 1,
            sharedMem, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchUpsampleImpl(nint func, ulong output, ulong input,
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

        long totalElements = (long)batch * channels * outH * outW;
        uint gridDim = (uint)((totalElements + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchIm2ColImpl(nint func, ulong col, ulong input,
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

        long totalElements = (long)channels * kH * kW * outH * outW;
        uint gridDim = (uint)((totalElements + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchBiasAddImpl(nint func, ulong output, ulong bias, int outChannels, int spatial, int totalElements, nint stream)
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
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchTranspose2DImpl(nint func, ulong output, ulong input, int d1, int d2, int totalElements, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint d1Arg = (uint)d1, d2Arg = (uint)d2, totalArg = (uint)totalElements;

        void** args = stackalloc void*[5];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &d1Arg;
        args[3] = &d2Arg;
        args[4] = &totalArg;

        uint gridDim = ((uint)totalElements + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchPermute0213Impl(nint func, ulong output, ulong input, int s, int h, int d, int totalElements, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint sArg = (uint)s, hArg = (uint)h, dArg = (uint)d, totalArg = (uint)totalElements;

        void** args = stackalloc void*[6];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &sArg;
        args[3] = &hArg;
        args[4] = &dArg;
        args[5] = &totalArg;

        uint gridDim = ((uint)totalElements + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchGeGluImpl(nint func, ulong output, ulong input, int innerDim, int outputElements, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint innerDimArg = (uint)innerDim;
        uint outElemArg = (uint)outputElements;

        void** args = stackalloc void*[4];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &innerDimArg;
        args[3] = &outElemArg;

        uint gridDim = ((uint)outputElements + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchBroadcastAddImpl(nint func, ulong hidden, ulong bias, int channels, int spatial, int totalElements, nint stream)
    {
        ulong hiddenArg = hidden, biasArg = bias;
        uint chArg = (uint)channels, spatialArg = (uint)spatial, totalArg = (uint)totalElements;

        void** args = stackalloc void*[5];
        args[0] = &hiddenArg;
        args[1] = &biasArg;
        args[2] = &chArg;
        args[3] = &spatialArg;
        args[4] = &totalArg;

        uint gridDim = ((uint)totalElements + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    // ── Elementwise Launches ────────────────────────────────────────────

    /// <summary>Launches elementwise add: output[i] = a[i] + b[i] (F32)</summary>
    public void LaunchAdd(ulong output, ulong a, ulong b, int count, nint stream)
        => LaunchBinaryImpl(_addF32, output, a, b, count, stream);

    /// <summary>Launches elementwise add: output[i] = a[i] + b[i] (F16)</summary>
    public void LaunchAddF16(ulong output, ulong a, ulong b, int count, nint stream)
        => LaunchBinaryImpl(_addF16, output, a, b, count, stream);

    /// <summary>Launches elementwise multiply: output[i] = a[i] * b[i] (F32)</summary>
    public void LaunchMul(ulong output, ulong a, ulong b, int count, nint stream)
        => LaunchBinaryImpl(_mulF32, output, a, b, count, stream);

    /// <summary>Launches elementwise multiply: output[i] = a[i] * b[i] (F16)</summary>
    public void LaunchMulF16(ulong output, ulong a, ulong b, int count, nint stream)
        => LaunchBinaryImpl(_mulF16, output, a, b, count, stream);

    /// <summary>Launches elementwise scale: output[i] = input[i] * scalar (F32)</summary>
    public void LaunchScale(ulong output, ulong input, float scalar, int count, nint stream)
        => LaunchScaleImpl(_scaleF32, output, input, scalar, count, stream);

    /// <summary>Launches elementwise scale: output[i] = input[i] * scalar (F16)</summary>
    public void LaunchScaleF16(ulong output, ulong input, float scalar, int count, nint stream)
        => LaunchScaleImpl(_scaleF16, output, input, scalar, count, stream);

    /// <summary>Launches SiLU activation: output[i] = input[i] * sigmoid(input[i]) (F32)</summary>
    public void LaunchSilu(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_siluF32, output, input, count, stream);

    /// <summary>Launches SiLU activation: output[i] = input[i] * sigmoid(input[i]) (F16)</summary>
    public void LaunchSiluF16(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_siluF16, output, input, count, stream);

    /// <summary>Launches GELU activation (F32)</summary>
    public void LaunchGelu(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_geluF32, output, input, count, stream);

    /// <summary>Launches GELU activation (F16)</summary>
    public void LaunchGeluF16(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_geluF16, output, input, count, stream);

    /// <summary>Launches elementwise clamp: output[i] = clamp(input[i], min, max) (F32)</summary>
    public void LaunchClamp(ulong output, ulong input, float min, float max, int count, nint stream)
        => LaunchClampImpl(_clampF32, output, input, min, max, count, stream);

    /// <summary>Launches elementwise clamp: output[i] = clamp(input[i], min, max) (F16)</summary>
    public void LaunchClampF16(ulong output, ulong input, float min, float max, int count, nint stream)
        => LaunchClampImpl(_clampF16, output, input, min, max, count, stream);

    // ── Normalization Launches ──────────────────────────────────────────

    /// <summary>Launches GroupNorm (F32). Each block handles one (batch, group) pair.</summary>
    public void LaunchGroupNorm(ulong output, ulong input, ulong weight, ulong bias,
        int batch, int channels, int spatial, int groups, float eps, nint stream)
        => LaunchGroupNormImpl(_groupnormF32, output, input, weight, bias, batch, channels, spatial, groups, eps, stream);

    /// <summary>Launches GroupNorm with F16 I/O, FP32 accumulation.</summary>
    public void LaunchGroupNormF16(ulong output, ulong input, ulong weight, ulong bias,
        int batch, int channels, int spatial, int groups, float eps, nint stream)
        => LaunchGroupNormImpl(_groupnormF16, output, input, weight, bias, batch, channels, spatial, groups, eps, stream);

    /// <summary>Launches LayerNorm (F32). Each block handles one row.</summary>
    public void LaunchLayerNorm(ulong output, ulong input, ulong weight, ulong bias,
        int normDim, int totalRows, float eps, nint stream)
        => LaunchLayerNormImpl(_layernormF32, output, input, weight, bias, normDim, totalRows, eps, stream);

    /// <summary>Launches LayerNorm with F16 I/O, FP32 accumulation.</summary>
    public void LaunchLayerNormF16(ulong output, ulong input, ulong weight, ulong bias,
        int normDim, int totalRows, float eps, nint stream)
        => LaunchLayerNormImpl(_layernormF16, output, input, weight, bias, normDim, totalRows, eps, stream);

    // ── Fused GroupNorm+SiLU Launches ───────────────────────────────────

    /// <summary>Launches fused GroupNorm+SiLU: normalize, affine, then SiLU in one kernel (F32).</summary>
    public void LaunchGroupNormSilu(ulong output, ulong input, ulong weight, ulong bias,
        int batch, int channels, int spatial, int groups, float eps, nint stream)
        => LaunchGroupNormImpl(_groupnormSiluF32, output, input, weight, bias, batch, channels, spatial, groups, eps, stream);

    /// <summary>Launches fused GroupNorm+SiLU with F16 I/O, FP32 accumulation.</summary>
    public void LaunchGroupNormSiluF16(ulong output, ulong input, ulong weight, ulong bias,
        int batch, int channels, int spatial, int groups, float eps, nint stream)
        => LaunchGroupNormImpl(_groupnormSiluF16, output, input, weight, bias, batch, channels, spatial, groups, eps, stream);

    // ── Spatial Launches ────────────────────────────────────────────────

    /// <summary>Launches UpsampleNearest2D (F32).</summary>
    public void LaunchUpsampleNearest2D(ulong output, ulong input,
        int batch, int channels, int inH, int inW, int outH, int outW, int scaleH, int scaleW, nint stream)
        => LaunchUpsampleImpl(_upsampleNearest2dF32, output, input, batch, channels, inH, inW, outH, outW, scaleH, scaleW, stream);

    /// <summary>Launches UpsampleNearest2D (F16).</summary>
    public void LaunchUpsampleNearest2DF16(ulong output, ulong input,
        int batch, int channels, int inH, int inW, int outH, int outW, int scaleH, int scaleW, nint stream)
        => LaunchUpsampleImpl(_upsampleNearest2dF16, output, input, batch, channels, inH, inW, outH, outW, scaleH, scaleW, stream);

    /// <summary>Launches Im2Col for one batch element (F32).</summary>
    public void LaunchIm2Col(ulong col, ulong input,
        int channels, int inH, int inW, int kH, int kW,
        int padH, int padW, int strideH, int strideW,
        int outH, int outW, int batchOffset, nint stream)
        => LaunchIm2ColImpl(_im2colF32, col, input, channels, inH, inW, kH, kW, padH, padW, strideH, strideW, outH, outW, batchOffset, stream);

    /// <summary>Launches Im2Col for one batch element (F16).</summary>
    public void LaunchIm2ColF16(ulong col, ulong input,
        int channels, int inH, int inW, int kH, int kW,
        int padH, int padW, int strideH, int strideW,
        int outH, int outW, int batchOffset, nint stream)
        => LaunchIm2ColImpl(_im2colF16, col, input, channels, inH, inW, kH, kW, padH, padW, strideH, strideW, outH, outW, batchOffset, stream);

    /// <summary>Launches bias addition: output[i] += bias[channel_of(i)] (F32)</summary>
    public void LaunchBiasAdd(ulong output, ulong bias, int outChannels, int spatial, int totalElements, nint stream)
        => LaunchBiasAddImpl(_col2biasAddF32, output, bias, outChannels, spatial, totalElements, stream);

    /// <summary>Launches bias addition: output[i] += bias[channel_of(i)] (F16)</summary>
    public void LaunchBiasAddF16(ulong output, ulong bias, int outChannels, int spatial, int totalElements, nint stream)
        => LaunchBiasAddImpl(_col2biasAddF16, output, bias, outChannels, spatial, totalElements, stream);

    // ── Softmax Launches ────────────────────────────────────────────────

    /// <summary>Launches in-place per-row softmax (F32). One block per row.</summary>
    public void LaunchSoftmax(ulong data, int rowLen, int totalRows, nint stream)
        => LaunchSoftmaxImpl(_softmaxF32, data, rowLen, totalRows, stream);

    /// <summary>Launches in-place per-row softmax with F16 I/O, FP32 accumulation.</summary>
    public void LaunchSoftmaxF16(ulong data, int rowLen, int totalRows, nint stream)
        => LaunchSoftmaxImpl(_softmaxF16, data, rowLen, totalRows, stream);

    // ── Transpose/Permute Launches ──────────────────────────────────────

    /// <summary>Launches batched 2D transpose: [B, D1, D2] -> [B, D2, D1] (F32).</summary>
    public void LaunchTranspose2D(ulong output, ulong input, int d1, int d2, int totalElements, nint stream)
        => LaunchTranspose2DImpl(_transpose2dF32, output, input, d1, d2, totalElements, stream);

    /// <summary>Launches batched 2D transpose: [B, D1, D2] -> [B, D2, D1] (F16).</summary>
    public void LaunchTranspose2DF16(ulong output, ulong input, int d1, int d2, int totalElements, nint stream)
        => LaunchTranspose2DImpl(_transpose2dF16, output, input, d1, d2, totalElements, stream);

    /// <summary>Launches 4D permute(0,2,1,3): [B, S, H, D] -> [B, H, S, D] (F32).</summary>
    public void LaunchPermute0213(ulong output, ulong input, int s, int h, int d, int totalElements, nint stream)
        => LaunchPermute0213Impl(_permute0213F32, output, input, s, h, d, totalElements, stream);

    /// <summary>Launches 4D permute(0,2,1,3): [B, S, H, D] -> [B, H, S, D] (F16).</summary>
    public void LaunchPermute0213F16(ulong output, ulong input, int s, int h, int d, int totalElements, nint stream)
        => LaunchPermute0213Impl(_permute0213F16, output, input, s, h, d, totalElements, stream);

    // ── GeGlu Launches ──────────────────────────────────────────────────

    /// <summary>Launches GEGLU: splits input along last dim, applies GELU gate (F32).</summary>
    public void LaunchGeGlu(ulong output, ulong input, int innerDim, int outputElements, nint stream)
        => LaunchGeGluImpl(_gegluF32, output, input, innerDim, outputElements, stream);

    /// <summary>Launches GEGLU with F16 I/O, F32 internal compute.</summary>
    public void LaunchGeGluF16(ulong output, ulong input, int innerDim, int outputElements, nint stream)
        => LaunchGeGluImpl(_gegluF16, output, input, innerDim, outputElements, stream);

    // ── BroadcastAdd Launches ───────────────────────────────────────────

    /// <summary>Launches broadcast add: hidden[b,c,s] += bias[b,c] in-place (F32).</summary>
    public void LaunchBroadcastAdd(ulong hidden, ulong bias, int channels, int spatial, int totalElements, nint stream)
        => LaunchBroadcastAddImpl(_broadcastAddF32, hidden, bias, channels, spatial, totalElements, stream);

    /// <summary>Launches broadcast add: hidden[b,c,s] += bias[b,c] in-place (F16).</summary>
    public void LaunchBroadcastAddF16(ulong hidden, ulong bias, int channels, int spatial, int totalElements, nint stream)
        => LaunchBroadcastAddImpl(_broadcastAddF16, hidden, bias, channels, spatial, totalElements, stream);

    // ── Cast Launches ───────────────────────────────────────────────────

    /// <summary>Launches FP32 to FP16 cast.</summary>
    public void LaunchCastF32ToF16(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_castF32ToF16, output, input, count, stream);

    /// <summary>Launches FP16 to FP32 cast.</summary>
    public void LaunchCastF16ToF32(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_castF16ToF32, output, input, count, stream);

    /// <summary>Launches FP8 E4M3 to FP16 cast. Input is 1 byte/element, output is 2 bytes/element.</summary>
    public void LaunchCastF8E4M3ToF16(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_castF8E4M3ToF16, output, input, count, stream);

    /// <summary>Launches FP16 to FP8 E4M3 cast with saturation. Input is 2 bytes/element, output is 1 byte/element.</summary>
    public void LaunchCastF16ToF8E4M3(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_castF16ToF8E4M3, output, input, count, stream);

    /// <summary>Launches BF16 to F32 cast (lossless — BF16 is the upper 16 bits of F32). Input is 2 bytes/element, output is 4 bytes/element.</summary>
    public void LaunchCastBf16ToF32(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_castBf16ToF32, output, input, count, stream);

    /// <summary>Launches F32 to BF16 cast with round-to-nearest-even. Input is 4 bytes/element, output is 2 bytes/element.</summary>
    public void LaunchCastF32ToBf16(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_castF32ToBf16, output, input, count, stream);

    // ── GGUF Dequant Launches ────────────────────────────────────────────

    /// <summary>Launches Q8_0 → F16 dequant. <paramref name="elementCount"/> is the total element count (must be a multiple of 32). Internally launches one CUDA block per Q8_0 quant block (32 elements), 32 threads per CUDA block.</summary>
    public unsafe void LaunchDequantQ8_0ToF16(ulong output, ulong input, int elementCount, nint stream)
    {
        if (elementCount % 32 != 0)
            throw new ArgumentException($"Q8_0 element count must be a multiple of 32, got {elementCount}.");
        int superBlockCount = elementCount / 32;
        LaunchDequantImpl(_dequantQ8_0ToF16, output, input, superBlockCount, threadsPerBlock: 32, stream);
    }

    /// <summary>Launches Q4_K → F16 dequant. Element count must be a multiple of 256 (super-block size).</summary>
    public unsafe void LaunchDequantQ4_KToF16(ulong output, ulong input, int elementCount, nint stream)
    {
        if (elementCount % 256 != 0)
            throw new ArgumentException($"Q4_K element count must be a multiple of 256, got {elementCount}.");
        int superBlockCount = elementCount / 256;
        LaunchDequantImpl(_dequantQ4_KToF16, output, input, superBlockCount, threadsPerBlock: 256, stream);
    }

    /// <summary>Launches Q5_K → F16 dequant. Element count must be a multiple of 256.</summary>
    public unsafe void LaunchDequantQ5_KToF16(ulong output, ulong input, int elementCount, nint stream)
    {
        if (elementCount % 256 != 0)
            throw new ArgumentException($"Q5_K element count must be a multiple of 256, got {elementCount}.");
        int superBlockCount = elementCount / 256;
        LaunchDequantImpl(_dequantQ5_KToF16, output, input, superBlockCount, threadsPerBlock: 256, stream);
    }

    /// <summary>Launches Q6_K → F16 dequant. Element count must be a multiple of 256. The Q6_K kernel uses 64 threads per CUDA block — each thread emits 4 elements at strides {0, +32, +64, +96} (2 halves × 32 l-values = 64 threads cover all 256 elements).</summary>
    public unsafe void LaunchDequantQ6_KToF16(ulong output, ulong input, int elementCount, nint stream)
    {
        if (elementCount % 256 != 0)
            throw new ArgumentException($"Q6_K element count must be a multiple of 256, got {elementCount}.");
        int superBlockCount = elementCount / 256;
        LaunchDequantImpl(_dequantQ6_KToF16, output, input, superBlockCount, threadsPerBlock: 64, stream);
    }

    private unsafe void LaunchDequantImpl(nint func, ulong output, ulong input, int superBlockCount, int threadsPerBlock, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint countArg = (uint)superBlockCount;
        void** args = stackalloc void*[3];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &countArg;
        CudaDriverApi.cuLaunchKernel(
            func,
            (uint)superBlockCount, 1, 1,
            (uint)threadsPerBlock, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    // ── Dispose ─────────────────────────────────────────────────────────

    private void DisposeModules()
    {
        _elementwiseModule.Dispose();
        _groupnormModule.Dispose();
        _layernormModule.Dispose();
        _spatialModule.Dispose();
        _softmaxModule.Dispose();
        _transposeModule.Dispose();
        _gegluModule.Dispose();
        _broadcastAddModule.Dispose();
        _elementwiseF16Module.Dispose();
        _groupnormF16Module.Dispose();
        _layernormF16Module.Dispose();
        _spatialF16Module.Dispose();
        _softmaxF16Module.Dispose();
        _transposeF16Module.Dispose();
        _gegluF16Module.Dispose();
        _broadcastAddF16Module.Dispose();
        _groupnormSiluModule.Dispose();
        _groupnormSiluF16Module.Dispose();
        _castModule.Dispose();
        _castF8Module.Dispose();
        _castBf16Module.Dispose();
        _dequantQ8_0Module.Dispose();
        _dequantQ4_KModule.Dispose();
        _dequantQ5_KModule.Dispose();
        _dequantQ6_KModule.Dispose();
    }

    public void Dispose()
    {
        DisposeModules();
        GC.SuppressFinalize(this);
    }

    ~CudaKernels()
    {
        DisposeModules();
    }
}
