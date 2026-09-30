namespace HartsyInference.Cuda;

// Recipe dequant to BF16 (Kernels/dequant/dequant_recipe_to_bf16.cu): DeepSeek-V4.1 MXFP4/FP8-E8M0, ModelOpt NVFP4 and MLX affine.
public sealed partial class CudaKernels
{
    private CudaModule? _recipeDequantModule;
    private nint _mxfp4E8m0ToBf16;
    private nint _fp8BlockE8m0ToBf16;
    private nint _nvfp4ModelOptToBf16;
    private nint _affineToBf16;
    private nint _exl3ToBf16;
    private const int Exl3SharedBytes = 128 * 129 * sizeof(float);

    /// <summary>True when dequant_recipe_to_bf16.ptx loaded with every recipe kernel.</summary>
    public bool HasRecipeDequantKernels =>
        _mxfp4E8m0ToBf16 != 0 && _fp8BlockE8m0ToBf16 != 0 && _nvfp4ModelOptToBf16 != 0 && _affineToBf16 != 0;

    /// <summary>True when the EXL3 decode kernel loaded and the device allows its 66,048 bytes of dynamic shared memory; the other recipe kernels do not depend on it.</summary>
    public bool HasExl3DequantKernel => _exl3ToBf16 != 0;

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
        _exl3ToBf16 = _recipeDequantModule.GetFunction("dequant_exl3_2bit_to_bf16");
        // 66,048 bytes of dynamic shared memory: past the 48 KB default, so the opt-in attribute (CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8) is required.
        int attributeResult = CudaDriverApi.cuFuncSetAttribute(_exl3ToBf16, 8, Exl3SharedBytes);
        // A device capped at 48 KB of shared memory loses only the EXL3 path; the other recipe kernels stay usable.
        if (attributeResult != 0) _exl3ToBf16 = 0;
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
        if (blockCols <= 0) throw new ArgumentOutOfRangeException(nameof(blockCols), blockCols, "NVFP4 dequant needs a positive block width.");
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
        if (bits != 4 && bits != 8) throw new ArgumentOutOfRangeException(nameof(bits), bits, "Affine dequant reads 4- or 8-bit fields.");
        if (bits == 4 && cols % 2 != 0) throw new ArgumentException($"4-bit affine dequant needs an even column count; got {cols}.", nameof(cols));
        ulong pArg = packed, sArg = scale, bArg = bias, oArg = output;
        uint rowsArg = (uint)rows, colsArg = (uint)cols, offsetArg = (uint)scaleColOffset;
        uint blockRowsArg = (uint)blockRows, groupArg = (uint)group, bitsArg = (uint)bits;
        ulong strideArg = (ulong)scaleStride;
        void** args = stackalloc void*[11];
        args[0] = &pArg; args[1] = &sArg; args[2] = &bArg; args[3] = &oArg; args[4] = &rowsArg; args[5] = &colsArg;
        args[6] = &strideArg; args[7] = &offsetArg; args[8] = &blockRowsArg; args[9] = &groupArg; args[10] = &bitsArg;
        LaunchGrid(_affineToBf16, cols, rows, args, stream);
    }

    /// <summary>EXL3 2-bit MCG trellis (with <c>suh</c>/<c>svh</c> F16 sign vectors) to BF16 <c>[rowBlocks * 128, cols]</c>; one thread block per 128x128 Hadamard block.</summary>
    /// <param name="trellis">The whole <c>[in/16, out/16, 32]</c> int16 trellis.</param>
    /// <param name="outTiles">Output width in 16-wide tiles (the trellis tile-row pitch).</param>
    /// <param name="firstOutBlock">First output 128-row block to decode.</param>
    /// <param name="rowBlocks">Output 128-row blocks to decode.</param>
    /// <param name="cols">Input width, a multiple of 128.</param>
    public unsafe void LaunchExl3Dequant(ulong output, ulong trellis, ulong suh, ulong svh, int outTiles, int firstOutBlock, int rowBlocks, int cols, nint stream)
    {
        if (_exl3ToBf16 == 0) throw new NotSupportedException("The EXL3 dequant kernel is not loaded: dequant_recipe_to_bf16.ptx is missing it, or the device cannot opt in to 66,048 bytes of dynamic shared memory.");
        if (cols <= 0 || cols % 128 != 0) throw new ArgumentOutOfRangeException(nameof(cols), cols, "EXL3 dequant needs a positive multiple of 128 input columns.");
        ulong tArg = trellis, uArg = suh, vArg = svh, oArg = output;
        uint outTilesArg = (uint)outTiles, firstArg = (uint)firstOutBlock, colsArg = (uint)cols;
        void** args = stackalloc void*[7];
        args[0] = &tArg; args[1] = &uArg; args[2] = &vArg; args[3] = &oArg; args[4] = &outTilesArg; args[5] = &firstArg; args[6] = &colsArg;
        CudaDriverApi.cuLaunchKernel(_exl3ToBf16, (uint)rowBlocks, (uint)(cols / 128), 1, 256, 1, 1, Exl3SharedBytes, stream, (nint)args, 0).ThrowOnError();
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
