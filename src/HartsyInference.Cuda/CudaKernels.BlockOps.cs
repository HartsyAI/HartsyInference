using HartsyInference.Core.Backends.Latent;

namespace HartsyInference.Cuda;

// Shared block ops: the Walsh-Hadamard stages (Kernels/rotary), the FP8x128 activation round trip (Kernels/latent) and the
// block-scaled FP8 linear (Kernels/dequant). Each module is optional, so absence leaves its primitive unsupported.
public sealed partial class CudaKernels
{
    private CudaModule? _hadamardModule;
    private nint _hadamardStageF32;
    private nint _hadamardScaleF32;
    private CudaModule? _fp8BlockLinearModule;
    private nint _fp8BlockLinearF32;
    private nint _actQuantDequantFp8x128F32;

    /// <summary>True when hadamard.ptx loaded, so <c>Hadamard</c> runs on this device.</summary>
    public bool HasHadamardKernels => _hadamardStageF32 != 0 && _hadamardScaleF32 != 0;

    /// <summary>True when fp8_block_linear.ptx loaded, so <c>Fp8BlockLinear</c> runs on this device.</summary>
    public bool HasFp8BlockLinearKernels => _fp8BlockLinearF32 != 0;

    /// <summary>True when the FP8x128 entry of latent_quant.ptx loaded, so the FP8x128 activation round trip runs on this device.</summary>
    public bool HasActQuantFp8x128Kernel => _actQuantDequantFp8x128F32 != 0;

    // Runs after LoadAttentionKernels, which loads the latent_quant module this entry lives in.
    private void LoadBlockOpKernels()
    {
        string hadamardPath = Ptx("hadamard");
        if (CanLoad(hadamardPath))
        {
            _hadamardModule = LoadOwnedModule(hadamardPath);
            _hadamardStageF32 = _hadamardModule.TryGetFunction("hadamard_stage_f32");
            _hadamardScaleF32 = _hadamardModule.TryGetFunction("hadamard_scale_f32");
        }
        string fp8LinearPath = Ptx("fp8_block_linear");
        if (CanLoad(fp8LinearPath))
        {
            _fp8BlockLinearModule = LoadOwnedModule(fp8LinearPath);
            _fp8BlockLinearF32 = _fp8BlockLinearModule.TryGetFunction("fp8_block_linear_f32");
        }
        if (_latentQuantModule is not null && _actQuantDequantInplaceF32 != 0)
            _actQuantDequantFp8x128F32 = _latentQuantModule.TryGetFunction("act_quant_dequant_fp8x128_inplace_f32");
    }

    /// <summary>One butterfly stage over every row pair (src may equal dst); scale multiplies both outputs.</summary>
    public unsafe void LaunchHadamardStage(ulong src, ulong dst, long pairs, int width, int h, float scale, nint stream)
    {
        if (_hadamardStageF32 == 0) throw new InvalidOperationException("hadamard.ptx not present in the Ptx folder.");
        ulong sA = src, dA = dst;
        long pA = pairs;
        int wA = width, hA = h;
        float scaleA = scale;
        void** a = stackalloc void*[6];
        a[0] = &sA; a[1] = &dA; a[2] = &pA; a[3] = &wA; a[4] = &hA; a[5] = &scaleA;
        CudaDriverApi.cuLaunchKernel(_hadamardStageF32, Blocks(pairs), 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0)
            .ThrowOnError();
    }

    /// <summary>Multiplies <paramref name="count"/> floats by <paramref name="scale"/> (the width-1 Hadamard).</summary>
    public unsafe void LaunchHadamardScale(ulong src, ulong dst, long count, float scale, nint stream)
    {
        if (_hadamardScaleF32 == 0) throw new InvalidOperationException("hadamard.ptx not present in the Ptx folder.");
        ulong sA = src, dA = dst;
        long nA = count;
        float scaleA = scale;
        void** a = stackalloc void*[4];
        a[0] = &sA; a[1] = &dA; a[2] = &nA; a[3] = &scaleA;
        CudaDriverApi.cuLaunchKernel(_hadamardScaleF32, Blocks(count), 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0)
            .ThrowOnError();
    }

    /// <summary><c>y[m, n] = x[m, :] · e4m3(w[n, :]) · scale + bias[n]</c>; one 128-thread block per output column and row.</summary>
    /// <param name="bias">Device pointer to N floats, or 0 for no bias.</param>
    public unsafe void LaunchFp8BlockLinear(ulong x, ulong w, ulong scale, ulong bias, ulong y, int m, int n, int k,
        int blockRows, int blockCols, nint stream)
    {
        if (_fp8BlockLinearF32 == 0) throw new InvalidOperationException("fp8_block_linear.ptx not present in the Ptx folder.");
        ulong xA = x, wA = w, sA = scale, bA = bias, yA = y;
        int mA = m, nA = n, kA = k, brA = blockRows, bcA = blockCols, gcA = (k + blockCols - 1) / blockCols;
        void** a = stackalloc void*[11];
        a[0] = &xA; a[1] = &wA; a[2] = &sA; a[3] = &bA; a[4] = &yA; a[5] = &mA; a[6] = &nA; a[7] = &kA;
        a[8] = &brA; a[9] = &bcA; a[10] = &gcA;
        uint gridY = (uint)Math.Min(m, 65535);
        CudaDriverApi.cuLaunchKernel(_fp8BlockLinearF32, (uint)n, gridY, 1, 128, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Quantize-dequantize <paramref name="groups"/> FP8x128 groups in place; one thread per group.</summary>
    public unsafe void LaunchActQuantDequantFp8x128(ulong x, long groups, nint stream)
    {
        if (_actQuantDequantFp8x128F32 == 0) throw new InvalidOperationException("latent_quant.ptx has no FP8x128 entry in the Ptx folder.");
        ulong xA = x;
        long gA = groups;
        void** a = stackalloc void*[2];
        a[0] = &xA; a[1] = &gA;
        CudaDriverApi.cuLaunchKernel(_actQuantDequantFp8x128F32, Blocks(groups), 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0)
            .ThrowOnError();
    }
}
