using HartsyInference.Core.Backends.Latent;

namespace HartsyInference.Cuda;

// Shared block ops: the Walsh-Hadamard stages (Kernels/rotary), the FP8x128 activation round trip and the sparse index
// builders (Kernels/latent), and the block-scaled FP8 and grouped linears (Kernels/dequant). Each module is optional, so
// absence leaves its primitive unsupported.
public sealed partial class CudaKernels
{
    private CudaModule? _hadamardModule;
    private nint _hadamardStageF32;
    private nint _hadamardScaleF32;
    private CudaModule? _fp8BlockLinearModule;
    private nint _fp8BlockLinearF32;
    private nint _blockDiagLinearFp8F32;
    private nint _blockDiagLinearF32;
    private nint _actQuantDequantFp8x128F32;
    private CudaModule? _sparseIndexModule;
    private nint _sparseTopKF32;
    private nint _mergeWindowTopKI32;

    /// <summary>Threads per block of <c>sparse_topk_f32</c>; must equal <c>TOPK_THREADS</c> in sparse_index.cu.</summary>
    private const uint SparseTopKThreads = 512;

    /// <summary>True when hadamard.ptx loaded, so <c>Hadamard</c> runs on this device.</summary>
    public bool HasHadamardKernels => _hadamardStageF32 != 0 && _hadamardScaleF32 != 0;

    /// <summary>True when fp8_block_linear.ptx loaded, so <c>Fp8BlockLinear</c> runs on this device.</summary>
    public bool HasFp8BlockLinearKernels => _fp8BlockLinearF32 != 0;

    /// <summary>True when fp8_block_linear.ptx carries the block-diagonal entries, so <c>BlockDiagonalLinear</c> runs on this device.</summary>
    public bool HasBlockDiagonalLinearKernels => _blockDiagLinearFp8F32 != 0 && _blockDiagLinearF32 != 0;

    /// <summary>True when sparse_index.ptx loaded, so <c>SparseTopK</c> and <c>MergeWindowTopK</c> run on this device.</summary>
    public bool HasSparseIndexKernels => _sparseTopKF32 != 0 && _mergeWindowTopKI32 != 0;

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
            _blockDiagLinearFp8F32 = _fp8BlockLinearModule.TryGetFunction("block_diag_linear_fp8_f32");
            _blockDiagLinearF32 = _fp8BlockLinearModule.TryGetFunction("block_diag_linear_f32");
        }
        string sparseIndexPath = Ptx("sparse_index");
        if (CanLoad(sparseIndexPath))
        {
            _sparseIndexModule = LoadOwnedModule(sparseIndexPath);
            _sparseTopKF32 = _sparseIndexModule.TryGetFunction("sparse_topk_f32");
            _mergeWindowTopKI32 = _sparseIndexModule.TryGetFunction("merge_window_topk_i32");
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

    /// <summary><c>y[m, n] = x[m, (n / groupRows)·K + :] · w[n, :]</c> over an F32 weight (<paramref name="scale"/> 0) or an FP8 e4m3
    /// weight with its block scales; one 128-thread block per output column and row.</summary>
    public unsafe void LaunchBlockDiagonalLinear(ulong x, ulong w, ulong scale, ulong y, int m, int n, int k, int inWidth, int groupRows,
        int blockRows, int blockCols, nint stream)
    {
        if (!HasBlockDiagonalLinearKernels)
            throw new InvalidOperationException("fp8_block_linear.ptx in the Ptx folder has no block_diag_linear entries.");
        ulong xA = x, wA = w, sA = scale, yA = y;
        int mA = m, nA = n, kA = k, iwA = inWidth, grA = groupRows, brA = blockRows, bcA = blockCols, gcA = (k + blockCols - 1) / blockCols;
        uint gridY = (uint)Math.Min(m, 65535);
        if (scale == 0)
        {
            void** f = stackalloc void*[8];
            f[0] = &xA; f[1] = &wA; f[2] = &yA; f[3] = &mA; f[4] = &nA; f[5] = &kA; f[6] = &iwA; f[7] = &grA;
            CudaDriverApi.cuLaunchKernel(_blockDiagLinearF32, (uint)n, gridY, 1, 128, 1, 1, 0, stream, (nint)f, 0).ThrowOnError();
            return;
        }
        void** a = stackalloc void*[12];
        a[0] = &xA; a[1] = &wA; a[2] = &sA; a[3] = &yA; a[4] = &mA; a[5] = &nA; a[6] = &kA; a[7] = &brA; a[8] = &bcA; a[9] = &gcA;
        a[10] = &iwA; a[11] = &grA;
        CudaDriverApi.cuLaunchKernel(_blockDiagLinearFp8F32, (uint)n, gridY, 1, 128, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Exact per-row top-k of <paramref name="rows"/> F32 score rows into ascending I32 columns; one block per row.</summary>
    public unsafe void LaunchSparseTopK(ulong scores, ulong limits, ulong output, int rows, int cols, int k, int offset, nint stream)
    {
        if (_sparseTopKF32 == 0) throw new InvalidOperationException("sparse_index.ptx not present in the Ptx folder.");
        ulong sA = scores, lA = limits, oA = output;
        int cA = cols, kA = k, offA = offset;
        void** a = stackalloc void*[6];
        a[0] = &sA; a[1] = &lA; a[2] = &oA; a[3] = &cA; a[4] = &kA; a[5] = &offA;
        CudaDriverApi.cuLaunchKernel(_sparseTopKF32, (uint)rows, 1, 1, SparseTopKThreads, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Writes <c>tokens x width</c> merged window/top-k rows; one thread per entry.</summary>
    public unsafe void LaunchMergeWindowTopK(ulong topk, ulong positions, ulong rows, long total, int topkCols, int window, int width,
        nint stream)
    {
        if (_mergeWindowTopKI32 == 0) throw new InvalidOperationException("sparse_index.ptx not present in the Ptx folder.");
        ulong tA = topk, pA = positions, rA = rows;
        long totA = total;
        int cA = topkCols, wA = window, widthA = width;
        void** a = stackalloc void*[7];
        a[0] = &tA; a[1] = &pA; a[2] = &rA; a[3] = &totA; a[4] = &cA; a[5] = &wA; a[6] = &widthA;
        CudaDriverApi.cuLaunchKernel(_mergeWindowTopKI32, Blocks(total), 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
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
