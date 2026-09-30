namespace HartsyInference.Cuda;

// Recipe dequant to BF16 (Kernels/dequant/dequant_recipe_to_bf16.cu): DeepSeek-V4.1 MXFP4/FP8-E8M0, ModelOpt NVFP4 and MLX affine.
public sealed partial class CudaKernels
{
    private CudaModule? _recipeDequantModule;
    private nint _mxfp4E8m0ToBf16;
    private nint _fp8BlockE8m0ToBf16;
    private nint _nvfp4ModelOptToBf16;
    private nint _affineToBf16;

    /// <summary>True when dequant_recipe_to_bf16.ptx loaded with every recipe kernel.</summary>
    public bool HasRecipeDequantKernels =>
        _mxfp4E8m0ToBf16 != 0 && _fp8BlockE8m0ToBf16 != 0 && _nvfp4ModelOptToBf16 != 0 && _affineToBf16 != 0;

    // Optional module: absence leaves recipe dequant unsupported instead of failing construction.
    private void LoadRecipeDequantKernels()
    {
        string path = Ptx("dequant_recipe_to_bf16");
        if (!File.Exists(path)) return;
        _recipeDequantModule = LoadOwnedModule(path);
        _mxfp4E8m0ToBf16 = _recipeDequantModule.GetFunction("dequant_mxfp4_e8m0_to_bf16");
        _fp8BlockE8m0ToBf16 = _recipeDequantModule.GetFunction("dequant_fp8_block_e8m0_to_bf16");
        _nvfp4ModelOptToBf16 = _recipeDequantModule.GetFunction("dequant_nvfp4_modelopt_to_bf16");
        _affineToBf16 = _recipeDequantModule.GetFunction("dequant_affine_to_bf16");
    }

    /// <summary>E2M1 nibbles times F8E8M0 scales to BF16 <c>[rows, 2 * packedCols]</c>; one thread per packed byte.</summary>
    /// <param name="scaleStride">Scale row length in bytes (the scale tensor's second dimension).</param>
    public unsafe void LaunchMxfp4E8m0Dequant(ulong output, ulong packed, ulong scale, int rows, int packedCols,
        long scaleStride, int scaleColOffset, int blockRows, int blockCols, nint stream)
    {
        if (_mxfp4E8m0ToBf16 == 0) throw new InvalidOperationException("dequant_recipe_to_bf16.ptx not present in the Ptx folder.");
        LaunchRecipe(_mxfp4E8m0ToBf16, output, packed, scale, rows, packedCols, scaleStride, scaleColOffset, blockRows, blockCols, stream);
    }

    /// <summary>E4M3 bytes times F8E8M0 block scales to BF16 <c>[rows, cols]</c>; one thread per element.</summary>
    public unsafe void LaunchFp8BlockE8m0Dequant(ulong output, ulong packed, ulong scale, int rows, int cols,
        long scaleStride, int scaleColOffset, int blockRows, int blockCols, nint stream)
    {
        if (_fp8BlockE8m0ToBf16 == 0) throw new InvalidOperationException("dequant_recipe_to_bf16.ptx not present in the Ptx folder.");
        LaunchRecipe(_fp8BlockE8m0ToBf16, output, packed, scale, rows, cols, scaleStride, scaleColOffset, blockRows, blockCols, stream);
    }

    /// <summary>ModelOpt NVFP4 (E2M1 pairs, E4M3 scales per block, scalar F32 <paramref name="globalScale"/>) to BF16 <c>[rows, 2 * packedCols]</c>.</summary>
    public unsafe void LaunchNvfp4ModelOptDequant(ulong output, ulong packed, ulong scale, ulong globalScale, int rows, int packedCols,
        long scaleStride, int scaleColOffset, int blockRows, int blockCols, nint stream)
    {
        if (_nvfp4ModelOptToBf16 == 0) throw new InvalidOperationException("dequant_recipe_to_bf16.ptx not present in the Ptx folder.");
        ulong pArg = packed, sArg = scale, oArg = output, gArg = globalScale;
        uint rowsArg = (uint)rows, widthArg = (uint)packedCols, offsetArg = (uint)scaleColOffset;
        uint blockRowsArg = (uint)blockRows, blockColsArg = (uint)blockCols;
        ulong strideArg = (ulong)scaleStride;
        void** args = stackalloc void*[10];
        args[0] = &pArg; args[1] = &sArg; args[2] = &oArg; args[3] = &rowsArg; args[4] = &widthArg;
        args[5] = &strideArg; args[6] = &offsetArg; args[7] = &blockRowsArg; args[8] = &blockColsArg; args[9] = &gArg;
        LaunchGrid(_nvfp4ModelOptToBf16, packedCols, rows, args, stream);
    }

    /// <summary>MLX affine <c>q * scale + bias</c> (4- or 8-bit q, F32 scales and biases per <paramref name="group"/> columns) to BF16 <c>[rows, cols]</c>.</summary>
    /// <param name="scaleStride">Scale and bias row length in elements.</param>
    public unsafe void LaunchAffineDequant(ulong output, ulong packed, ulong scale, ulong bias, int rows, int cols,
        long scaleStride, int scaleColOffset, int blockRows, int group, int bits, nint stream)
    {
        if (_affineToBf16 == 0) throw new InvalidOperationException("dequant_recipe_to_bf16.ptx not present in the Ptx folder.");
        ulong pArg = packed, sArg = scale, bArg = bias, oArg = output;
        uint rowsArg = (uint)rows, colsArg = (uint)cols, offsetArg = (uint)scaleColOffset;
        uint blockRowsArg = (uint)blockRows, groupArg = (uint)group, bitsArg = (uint)bits;
        ulong strideArg = (ulong)scaleStride;
        void** args = stackalloc void*[11];
        args[0] = &pArg; args[1] = &sArg; args[2] = &bArg; args[3] = &oArg; args[4] = &rowsArg; args[5] = &colsArg;
        args[6] = &strideArg; args[7] = &offsetArg; args[8] = &blockRowsArg; args[9] = &groupArg; args[10] = &bitsArg;
        LaunchGrid(_affineToBf16, cols, rows, args, stream);
    }

    private static unsafe void LaunchGrid(nint function, int width, int rows, void** args, nint stream)
    {
        const uint BlockSize = 256;
        uint gridX = (uint)(((long)width + BlockSize - 1) / BlockSize);
        uint gridY = (uint)Math.Min(rows, 65535);
        CudaDriverApi.cuLaunchKernel(function, gridX, gridY, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    private static unsafe void LaunchRecipe(nint function, ulong output, ulong packed, ulong scale, int rows, int width,
        long scaleStride, int scaleColOffset, int blockRows, int blockCols, nint stream)
    {
        ulong pArg = packed, sArg = scale, oArg = output;
        uint rowsArg = (uint)rows, widthArg = (uint)width, offsetArg = (uint)scaleColOffset;
        uint blockRowsArg = (uint)blockRows, blockColsArg = (uint)blockCols;
        ulong strideArg = (ulong)scaleStride;
        void** args = stackalloc void*[9];
        args[0] = &pArg; args[1] = &sArg; args[2] = &oArg; args[3] = &rowsArg; args[4] = &widthArg;
        args[5] = &strideArg; args[6] = &offsetArg; args[7] = &blockRowsArg; args[8] = &blockColsArg;
        LaunchGrid(function, width, rows, args, stream);
    }
}
