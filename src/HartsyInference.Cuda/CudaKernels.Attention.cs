using HartsyInference.Core.Backends;

namespace HartsyInference.Cuda;

// Single-latent sparse attention, indexer, hyper-connection, latent quantization, window and rope kernels (Kernels/latent).
public sealed partial class CudaKernels
{
    /// <summary>Threads per block of the sparse latent attention kernel.</summary>
    private const int LatentAttentionThreads = 128;

    /// <summary>Most gathered rows per token the attention kernel can hold: one probability each in 48 KB of shared memory.</summary>
    public const int LatentAttentionMaxK = 12288;

    /// <summary>Widest latent row the indexer kernel keeps per warp lane (16 registers of 32 lanes).</summary>
    public const int IndexerMaxDim = 512;

    private CudaModule? _latentAttentionModule;
    private nint _sparseLatentAttentionF32;
    private nint _indexerScoresF32;
    private CudaModule? _latentQuantModule;
    private nint _latentRowWinnerInit;
    private nint _latentRowWinnerMark;
    private nint _latentQuantizeRowsF32;
    private nint _actQuantDequantInplaceF32;
    private CudaModule? _hcMixModule;
    private nint _hcSplitSinkhornF32;
    private nint _hcPreMixF32;
    private nint _hcPostMixF32;
    private CudaModule? _latentPositionsModule;
    private nint _ropeInterleavedOffsetF32;
    private nint _windowIndicesI32;

    /// <summary>True when all four latent PTX modules loaded, so every attention primitive can run on this device.</summary>
    public bool HasAttentionKernels => _sparseLatentAttentionF32 != 0 && _indexerScoresF32 != 0 &&
        _latentRowWinnerInit != 0 && _latentRowWinnerMark != 0 && _latentQuantizeRowsF32 != 0 &&
        _actQuantDequantInplaceF32 != 0 && _hcSplitSinkhornF32 != 0 && _hcPreMixF32 != 0 && _hcPostMixF32 != 0 &&
        _ropeInterleavedOffsetF32 != 0 && _windowIndicesI32 != 0;

    // Optional modules: absence leaves the primitives unsupported on this backend instead of failing construction.
    private void LoadAttentionKernels()
    {
        string attentionPath = Ptx("latent_attention");
        if (File.Exists(attentionPath))
        {
            _latentAttentionModule = LoadOwnedModule(attentionPath);
            _sparseLatentAttentionF32 = _latentAttentionModule.GetFunction("sparse_latent_attention_f32");
            _indexerScoresF32 = _latentAttentionModule.GetFunction("indexer_scores_f32");
        }
        string quantPath = Ptx("latent_quant");
        if (File.Exists(quantPath))
        {
            _latentQuantModule = LoadOwnedModule(quantPath);
            _latentRowWinnerInit = _latentQuantModule.GetFunction("latent_row_winner_init");
            _latentRowWinnerMark = _latentQuantModule.GetFunction("latent_row_winner_mark");
            _latentQuantizeRowsF32 = _latentQuantModule.GetFunction("latent_quantize_rows_f32");
            _actQuantDequantInplaceF32 = _latentQuantModule.GetFunction("act_quant_dequant_inplace_f32");
        }
        string hcPath = Ptx("hc_mix");
        if (File.Exists(hcPath))
        {
            _hcMixModule = LoadOwnedModule(hcPath);
            _hcSplitSinkhornF32 = _hcMixModule.GetFunction("hc_split_sinkhorn_f32");
            _hcPreMixF32 = _hcMixModule.GetFunction("hc_pre_mix_f32");
            _hcPostMixF32 = _hcMixModule.GetFunction("hc_post_mix_f32");
        }
        string positionsPath = Ptx("latent_positions");
        if (File.Exists(positionsPath))
        {
            _latentPositionsModule = LoadOwnedModule(positionsPath);
            _ropeInterleavedOffsetF32 = _latentPositionsModule.GetFunction("rope_interleaved_offset_f32");
            _windowIndicesI32 = _latentPositionsModule.GetFunction("window_indices_i32");
        }
    }

    private static uint Blocks(long count) => (uint)((count + BlockSize - 1) / BlockSize);

    /// <summary>One 128-thread block per (token, head). Source pointers are 0 when that source has no rows.</summary>
    public unsafe void LaunchSparseLatentAttention(ulong output, ulong query, ulong winCodes, ulong winScales,
        ulong mainCodes, ulong mainScales, ulong indices, ulong sink, int tokens, int heads, int dim, int k,
        int windowSlots, int mainRows, LatentEncoding winEnc, LatentEncoding mainEnc, float scale, nint stream)
    {
        if (_sparseLatentAttentionF32 == 0) throw new InvalidOperationException("latent_attention.ptx not present in the Ptx folder.");
        ulong oA = output, qA = query, wcA = winCodes, wsA = winScales, mcA = mainCodes, msA = mainScales, iA = indices, sA = sink;
        int hA = heads, dA = dim, kA = k, wsl = windowSlots, mr = mainRows, we = (int)winEnc, me = (int)mainEnc;
        float scaleA = scale;
        void** a = stackalloc void*[16];
        a[0] = &oA; a[1] = &qA; a[2] = &wcA; a[3] = &wsA; a[4] = &mcA; a[5] = &msA; a[6] = &iA; a[7] = &sA;
        a[8] = &hA; a[9] = &dA; a[10] = &kA; a[11] = &wsl; a[12] = &mr; a[13] = &we; a[14] = &me; a[15] = &scaleA;
        CudaDriverApi.cuLaunchKernel(_sparseLatentAttentionF32, (uint)tokens, (uint)heads, 1, LatentAttentionThreads, 1, 1,
            (uint)(4 * k), stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Eight keys per 256-thread block (one warp each) along x, one query token per y.</summary>
    public unsafe void LaunchIndexerScores(ulong scores, ulong query, ulong keyCodes, ulong keyScales, ulong headWeights,
        ulong compressLens, ulong candidates, int tokens, int heads, int dim, int keys, LatentEncoding keyEnc, float scale,
        nint stream)
    {
        if (_indexerScoresF32 == 0) throw new InvalidOperationException("latent_attention.ptx not present in the Ptx folder.");
        ulong sA = scores, qA = query, kcA = keyCodes, ksA = keyScales, hwA = headWeights, clA = compressLens, cA = candidates;
        int hA = heads, dA = dim, nA = keys, eA = (int)keyEnc;
        float scaleA = scale;
        void** a = stackalloc void*[12];
        a[0] = &sA; a[1] = &qA; a[2] = &kcA; a[3] = &ksA; a[4] = &hwA; a[5] = &clA; a[6] = &cA; a[7] = &hA;
        a[8] = &dA; a[9] = &nA; a[10] = &eA; a[11] = &scaleA;
        CudaDriverApi.cuLaunchKernel(_indexerScoresF32, (uint)((keys + 7) / 8), (uint)tokens, 1, 256, 1, 1, 0, stream, (nint)a, 0)
            .ThrowOnError();
    }

    /// <summary>Marks the last source row for every destination row, then quantizes only the winners, so duplicated
    /// destinations resolve deterministically to the later row. <paramref name="winner"/> holds destRows ints.</summary>
    public unsafe void LaunchQuantizeLatentRows(ulong codes, ulong scales, ulong rows, ulong physicalRows, ulong winner,
        int count, int dim, int destRows, LatentEncoding enc, nint stream)
    {
        if (_latentQuantizeRowsF32 == 0) throw new InvalidOperationException("latent_quant.ptx not present in the Ptx folder.");
        ulong wA = winner, pA = physicalRows, cA = codes, sA = scales, rA = rows;
        int drA = destRows, nA = count, dA = dim, eA = (int)enc;
        void** init = stackalloc void*[2];
        init[0] = &wA; init[1] = &drA;
        CudaDriverApi.cuLaunchKernel(_latentRowWinnerInit, Blocks(destRows), 1, 1, BlockSize, 1, 1, 0, stream, (nint)init, 0)
            .ThrowOnError();
        void** mark = stackalloc void*[4];
        mark[0] = &wA; mark[1] = &pA; mark[2] = &nA; mark[3] = &drA;
        CudaDriverApi.cuLaunchKernel(_latentRowWinnerMark, Blocks(count), 1, 1, BlockSize, 1, 1, 0, stream, (nint)mark, 0)
            .ThrowOnError();
        int group = LatentEncodings.GroupSize(enc);
        long units = (long)count * (group == 0 ? dim : dim / group);
        void** quant = stackalloc void*[9];
        quant[0] = &cA; quant[1] = &sA; quant[2] = &rA; quant[3] = &pA; quant[4] = &wA; quant[5] = &nA; quant[6] = &dA;
        quant[7] = &eA; quant[8] = &drA;
        CudaDriverApi.cuLaunchKernel(_latentQuantizeRowsF32, Blocks(units), 1, 1, BlockSize, 1, 1, 0, stream, (nint)quant, 0)
            .ThrowOnError();
    }

    /// <summary>Quantize-dequantize <paramref name="groups"/> consecutive groups in place; one thread per group.</summary>
    public unsafe void LaunchActQuantDequantInPlace(ulong x, long groups, LatentEncoding enc, nint stream)
    {
        if (_actQuantDequantInplaceF32 == 0) throw new InvalidOperationException("latent_quant.ptx not present in the Ptx folder.");
        ulong xA = x;
        long gA = groups;
        int eA = (int)enc;
        void** a = stackalloc void*[3];
        a[0] = &xA; a[1] = &gA; a[2] = &eA;
        CudaDriverApi.cuLaunchKernel(_actQuantDequantInplaceF32, Blocks(groups), 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0)
            .ThrowOnError();
    }

    /// <summary>Hyper-connection split and Sinkhorn normalization, one thread per token.</summary>
    public unsafe void LaunchHcSplitSinkhorn(ulong pre, ulong post, ulong comb, ulong mixes, ulong scale, ulong bias,
        int tokens, int hc, int iters, float eps, nint stream)
    {
        if (_hcSplitSinkhornF32 == 0) throw new InvalidOperationException("hc_mix.ptx not present in the Ptx folder.");
        ulong prA = pre, poA = post, cA = comb, mA = mixes, sA = scale, bA = bias;
        int tA = tokens, hA = hc, iA = iters;
        float eA = eps;
        void** a = stackalloc void*[10];
        a[0] = &prA; a[1] = &poA; a[2] = &cA; a[3] = &mA; a[4] = &sA; a[5] = &bA; a[6] = &tA; a[7] = &hA; a[8] = &iA;
        a[9] = &eA;
        CudaDriverApi.cuLaunchKernel(_hcSplitSinkhornF32, Blocks(tokens), 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0)
            .ThrowOnError();
    }

    /// <summary>Weighted sum over the hc streams, one thread per (token, column).</summary>
    public unsafe void LaunchHcPreMix(ulong output, ulong x, ulong pre, long tokens, int hc, int dim, nint stream)
    {
        if (_hcPreMixF32 == 0) throw new InvalidOperationException("hc_mix.ptx not present in the Ptx folder.");
        ulong oA = output, xA = x, pA = pre;
        long tA = tokens;
        int hA = hc, dA = dim;
        void** a = stackalloc void*[6];
        a[0] = &oA; a[1] = &xA; a[2] = &pA; a[3] = &tA; a[4] = &hA; a[5] = &dA;
        CudaDriverApi.cuLaunchKernel(_hcPreMixF32, Blocks(tokens * dim), 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Post gate plus comb-mixed residual streams, one thread per output element.</summary>
    public unsafe void LaunchHcPostMix(ulong output, ulong x, ulong residual, ulong post, ulong comb, long tokens, int hc,
        int dim, nint stream)
    {
        if (_hcPostMixF32 == 0) throw new InvalidOperationException("hc_mix.ptx not present in the Ptx folder.");
        ulong oA = output, xA = x, rA = residual, pA = post, cA = comb;
        long tA = tokens;
        int hA = hc, dA = dim;
        void** a = stackalloc void*[8];
        a[0] = &oA; a[1] = &xA; a[2] = &rA; a[3] = &pA; a[4] = &cA; a[5] = &tA; a[6] = &hA; a[7] = &dA;
        CudaDriverApi.cuLaunchKernel(_hcPostMixF32, Blocks(tokens * hc * dim), 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0)
            .ThrowOnError();
    }

    /// <summary>In-place interleaved rotary of the [dimOffset, dimOffset + 2*half) slice; one thread per pair.</summary>
    public unsafe void LaunchRopeInterleavedOffset(ulong x, ulong cos, ulong sin, long positions, int heads, int dim,
        int half, int dimOffset, nint stream)
    {
        if (_ropeInterleavedOffsetF32 == 0) throw new InvalidOperationException("latent_positions.ptx not present in the Ptx folder.");
        ulong xA = x, cA = cos, sA = sin;
        long pA = positions;
        int hA = heads, dA = dim, halfA = half, oA = dimOffset;
        void** a = stackalloc void*[8];
        a[0] = &xA; a[1] = &cA; a[2] = &sA; a[3] = &pA; a[4] = &hA; a[5] = &dA; a[6] = &halfA; a[7] = &oA;
        CudaDriverApi.cuLaunchKernel(_ropeInterleavedOffsetF32, Blocks(positions * heads * half), 1, 1, BlockSize, 1, 1, 0,
            stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Fills the [rows, cols] sliding-window ring slots; one thread per entry.</summary>
    public unsafe void LaunchWindowIndices(ulong indices, int windowSize, int startPos, int rows, int cols, nint stream)
    {
        if (_windowIndicesI32 == 0) throw new InvalidOperationException("latent_positions.ptx not present in the Ptx folder.");
        ulong iA = indices;
        int wA = windowSize, sA = startPos, rA = rows, cA = cols;
        void** a = stackalloc void*[5];
        a[0] = &iA; a[1] = &wA; a[2] = &sA; a[3] = &rA; a[4] = &cA;
        CudaDriverApi.cuLaunchKernel(_windowIndicesI32, Blocks((long)rows * cols), 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0)
            .ThrowOnError();
    }
}
