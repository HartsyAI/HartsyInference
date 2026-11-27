using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

// 3D block-sparse attention (Kernels/attention/block_sparse_attention.cu): block means, block selection, attention.
public sealed partial class CudaKernels
{
    /// <summary>Query rows and key rows of one attention tile (BSA_QT, BSA_KT), and the threads of one selection block.</summary>
    private const int BsaQueryTile = 64, BsaKeyTile = 64, BsaSelectThreads = 256;

    /// <summary>The threads of one attention block: a group of 4 query rows per warp, per half warp at head dim 16.</summary>
    private static uint BsaAttendThreads(int headDim) => headDim == 16 ? 256u : 512u;

    /// <summary>The static shared memory bsa_select takes beside its keys: the query block's mean (BSA_MAX_DIM floats) and one
    /// count per warp.</summary>
    private const int BsaSelectStaticSharedBytes = (128 + BsaSelectThreads / 32) * sizeof(float);

    /// <summary>The head dims the attention entries are compiled for.</summary>
    private static readonly int[] BsaHeadDims = [16, 32, 64, 128];

    private CudaModule? _blockSparseModule;
    private nint _bsaMeansF32, _bsaMeansBf16, _bsaSelect;
    // By operand type (F32, BF16) and head dim; zero where the card cannot give the entry its shared memory.
    private readonly nint[] _bsaAttend = new nint[8];

    /// <summary>True when block_sparse_attention.ptx loaded and at least one head dim's attention entries can run. A build without
    /// the file leaves this false, and block-sparse attention is then refused.</summary>
    public bool HasBlockSparseAttention => _bsaMeansF32 != 0 && _bsaMeansBf16 != 0 && _bsaSelect != 0 && Array.Exists(_bsaAttend, static f => f != 0);

    /// <summary>The most key blocks one call may have: the selection keeps one 4-byte key per block in what its static arrays
    /// leave of a block's default shared memory.</summary>
    internal int MaxBlockSparseKeyBlocks => (DefaultDynamicSharedBytes - BsaSelectStaticSharedBytes) / sizeof(uint);

    /// <summary>Whether both operand types have an attention entry of this head dim on this card.</summary>
    internal bool SupportsBlockSparseHeadDim(int headDim) =>
        Array.IndexOf(BsaHeadDims, headDim) is int at and >= 0 && _bsaAttend[at] != 0 && _bsaAttend[BsaHeadDims.Length + at] != 0;

    /// <summary>Dynamic shared memory of one attention block: the query tile, a key chunk, the chunk's weights (each row one float
    /// longer than its data) and the chunk's row numbers.</summary>
    internal static int BlockSparseAttentionSharedBytes(int headDim) =>
        sizeof(float) * ((BsaQueryTile + BsaKeyTile) * (headDim + 1) + BsaQueryTile * (BsaKeyTile + 1)) + sizeof(int) * BsaKeyTile;

    // Optional module: absence leaves the primitive unsupported on this backend instead of failing construction.
    private void LoadBlockSparseAttentionKernels()
    {
        string path = Ptx("block_sparse_attention");
        if (!CanLoad(path)) return;
        _blockSparseModule = LoadOwnedModule(path, required: false);
        _bsaMeansF32 = _blockSparseModule.GetFunction("bsa_means_f32");
        _bsaMeansBf16 = _blockSparseModule.GetFunction("bsa_means_bf16");
        _bsaSelect = _blockSparseModule.GetFunction("bsa_select");
        for (int i = 0; i < _bsaAttend.Length; i++)
        {
            int headDim = BsaHeadDims[i % BsaHeadDims.Length];
            nint entry = _blockSparseModule.GetFunction($"bsa_attend_{(i < BsaHeadDims.Length ? "f32" : "bf16")}_d{headDim}");
            int bytes = BlockSparseAttentionSharedBytes(headDim);
            // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8; a card or driver that refuses leaves the head dim unsupported.
            if (bytes > DefaultDynamicSharedBytes && (bytes > MaxDynamicSharedBytes || CudaDriverApi.cuFuncSetAttribute(entry, 8, bytes) != 0))
                continue;
            _bsaAttend[i] = entry;
        }
    }

    /// <summary>Stage 1: the mean row of every block of <paramref name="x"/> <c>[B, H, tokens, D]</c> into
    /// <paramref name="means"/> <c>[B, H, blocks, D]</c>, F32.</summary>
    internal unsafe void LaunchBlockSparseMeans(ulong means, ulong x, DType type, int batch, int heads, int tokens, int dim,
        int blocks, TokenGrid3D grid, TokenGrid3D box, nint stream)
    {
        ulong mA = means, xA = x;
        int tA = tokens, dA = dim, hA = heads, nA = blocks;
        int ghA = grid.Height, gwA = grid.Width, btA = box.Frames, bhA = box.Height, bwA = box.Width;
        void** a = stackalloc void*[11];
        a[0] = &mA; a[1] = &xA; a[2] = &tA; a[3] = &dA; a[4] = &hA; a[5] = &nA; a[6] = &ghA; a[7] = &gwA; a[8] = &btA; a[9] = &bhA;
        a[10] = &bwA;
        nint entry = type == DType.BF16 ? _bsaMeansBf16 : _bsaMeansF32;
        CudaDriverApi.cuLaunchKernel(entry, (uint)blocks, (uint)heads, (uint)batch, 128, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Stage 2: each query block's <paramref name="chosen"/> key blocks, ascending, into <paramref name="selected"/>
    /// <c>[B, H, queryBlocks, chosen]</c>, I32. <paramref name="chosen"/> is at least 1.</summary>
    internal unsafe void LaunchBlockSparseSelect(ulong selected, ulong queryMeans, ulong keyMeans, int batch, int heads, int dim,
        int queryBlocks, int keyBlocks, int chosen, nint stream)
    {
        ulong sA = selected, qA = queryMeans, kA = keyMeans;
        int dA = dim, hA = heads, qbA = queryBlocks, kbA = keyBlocks, cA = chosen;
        void** a = stackalloc void*[8];
        a[0] = &sA; a[1] = &qA; a[2] = &kA; a[3] = &dA; a[4] = &hA; a[5] = &qbA; a[6] = &kbA; a[7] = &cA;
        CudaDriverApi.cuLaunchKernel(_bsaSelect, (uint)queryBlocks, (uint)heads, (uint)batch, BsaSelectThreads, 1, 1,
            (uint)(keyBlocks * sizeof(uint)), stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Stage 3: each query attends the keys of its block's selected blocks. Q, K and V are <c>[B, H, tokens, D]</c> of
    /// <paramref name="type"/>; the output is F32, or BF16 when <paramref name="outputBf16"/> is set.</summary>
    internal unsafe void LaunchBlockSparseAttend(ulong output, ulong q, ulong k, ulong v, ulong selected, DType type, bool outputBf16,
        int batch, int heads, int dim, int queryTokens, int keyTokens, BlockSparseAttentionOptions options, float scale, nint stream)
    {
        TokenGrid3D qg = options.QueryGrid, qx = options.QueryBlock, kg = options.KeyGrid, kx = options.KeyBlock;
        ulong oA = output, qA = q, kA = k, vA = v, sA = selected;
        int qtA = queryTokens, ktA = keyTokens, hA = heads, qbA = options.QueryBlocks, cA = options.SelectedBlocks;
        int qghA = qg.Height, qgwA = qg.Width, qbtA = qx.Frames, qbhA = qx.Height, qbwA = qx.Width;
        int kghA = kg.Height, kgwA = kg.Width, kbtA = kx.Frames, kbhA = kx.Height, kbwA = kx.Width;
        float scaleA = scale;
        int obA = outputBf16 ? 1 : 0;
        void** a = stackalloc void*[22];
        a[0] = &oA; a[1] = &qA; a[2] = &kA; a[3] = &vA; a[4] = &sA; a[5] = &qtA; a[6] = &ktA; a[7] = &hA; a[8] = &qbA; a[9] = &cA;
        a[10] = &qghA; a[11] = &qgwA; a[12] = &qbtA; a[13] = &qbhA; a[14] = &qbwA; a[15] = &kghA; a[16] = &kgwA; a[17] = &kbtA;
        a[18] = &kbhA; a[19] = &kbwA; a[20] = &scaleA; a[21] = &obA;
        int at = Array.IndexOf(BsaHeadDims, dim) + (type == DType.BF16 ? BsaHeadDims.Length : 0);
        long tiles = (qx.Tokens + BsaQueryTile - 1) / BsaQueryTile;
        CudaDriverApi.cuLaunchKernel(_bsaAttend[at], checked((uint)(options.QueryBlocks * tiles)), (uint)heads, (uint)batch,
            BsaAttendThreads(dim), 1, 1, (uint)BlockSparseAttentionSharedBytes(dim), stream, (nint)a, 0).ThrowOnError();
    }
}
