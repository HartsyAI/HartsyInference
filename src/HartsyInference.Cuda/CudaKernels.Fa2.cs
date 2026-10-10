namespace HartsyInference.Cuda;

// Causal grouped-query FlashAttention-2 for LLM prefill (Kernels/lm/flash_attn_causal_f16.cu): F16 tensor cores, F32 accumulation.
public sealed partial class CudaKernels
{
    private CudaModule? _fa2Module, _decodeGqaModule, _kvScatterF16Module;
    private nint _kvScatterQkvF16, _kvScatterQkNormF16;

    /// <summary>True when kv_scatter_f16.ptx loaded: the fused graph-decode scatter kernels for an F16 key/value cache.</summary>
    public bool HasKvScatterF16 => _kvScatterQkvF16 != 0 && _kvScatterQkNormF16 != 0;
    private nint _fa2D128, _fa2D64, _kvToF16;
    private nint _decF32D128, _decF32D64, _decF16D128, _decF16D64;

    /// <summary>True when flash_attn_decode_gqa.ptx loaded.</summary>
    public bool HasDecodeGqa => _decF32D128 != 0 && _decF32D64 != 0 && _decF16D128 != 0 && _decF16D64 != 0;

    /// <summary>Head dimension and query-head group the grouped-query decode kernel is built for.</summary>
    public static bool DecodeGqaSupports(int d, int group) => (d == 128 || d == 64) && group >= 1 && group <= 16;

    /// <summary>Dynamic shared memory of one decode block: the query heads, a K tile (padded rows), a V tile and the probabilities.</summary>
    private static int DecodeSharedBytes(int d) => (16 * d + 32 * (d + 4) + 32 * d + 8 * 32) * sizeof(float);

    /// <summary>True when flash_attn_causal_f16.ptx loaded and the device has the m16n8k16 F16 tensor-core instruction (sm_80 and later).</summary>
    public bool HasFa2Causal => _fa2D128 != 0 && _fa2D64 != 0 && _kvToF16 != 0 && Sm >= 80;

    private void LoadFa2Kernels()
    {
        string path = Ptx("flash_attn_causal_f16");
        if (!File.Exists(path)) return;
        _fa2Module = LoadOwnedModule(path);
        _fa2D128 = _fa2Module.GetFunction("lm_fa2_causal_d128");
        _fa2D64 = _fa2Module.GetFunction("lm_fa2_causal_d64");
        _kvToF16 = _fa2Module.GetFunction("lm_kv_to_f16");
        string scatter = Ptx("kv_scatter_f16");
        if (File.Exists(scatter))
        {
            _kvScatterF16Module = LoadOwnedModule(scatter);
            _kvScatterQkvF16 = _kvScatterF16Module.GetFunction("lm_qkv_rope_scatter_f16kv");
            _kvScatterQkNormF16 = _kvScatterF16Module.GetFunction("lm_qknorm_rope_scatter_f16kv");
        }
        string decode = Ptx("flash_attn_decode_gqa");
        if (File.Exists(decode))
        {
            _decodeGqaModule = LoadOwnedModule(decode);
            _decF32D128 = _decodeGqaModule.GetFunction("lm_fa_decode_gqa_f32_d128");
            _decF32D64 = _decodeGqaModule.GetFunction("lm_fa_decode_gqa_f32_d64");
            _decF16D128 = _decodeGqaModule.GetFunction("lm_fa_decode_gqa_f16_d128");
            _decF16D64 = _decodeGqaModule.GetFunction("lm_fa_decode_gqa_f16_d64");
        }
    }

    /// <summary>Head dimensions the kernel is built for.</summary>
    public static bool Fa2SupportsHeadDim(int d) => d == 64 || d == 128;

    /// <summary>Dynamic shared memory of one block: a key tile and a value tile of 64 rows at the padded stride.</summary>
    private static int Fa2SharedBytes(int d) => 2 * 64 * (d + 8) * sizeof(ushort);

    /// <summary>Attention of a query block against F16 keys and values with a causal (and optional sliding-window) mask.</summary>
    /// <param name="keyStride">Sequence stride (positions) of the key and value buffers; only the first <paramref name="kvLen"/> are read.</param>
    public unsafe void LaunchFa2Causal(ulong output, ulong q, ulong k, ulong v, int b, int hq, int tq, int d, int hkv, int keyStride,
        int kvLen, int kvGroup, int qOffset, float scale, int window, nint stream)
    {
        nint fn = d == 128 ? _fa2D128 : _fa2D64;
        if (fn == 0 || !Fa2SupportsHeadDim(d)) throw new InvalidOperationException($"flash_attn_causal_f16 has no kernel for head dim {d}.");
        ulong oA = output, qA = q, kA = k, vA = v;
        int hqA = hq, tqA = tq, hkvA = hkv, lkA = keyStride, kvLenA = kvLen, groupA = kvGroup, offA = qOffset, winA = window;
        float scaleA = scale;
        void** a = stackalloc void*[13];
        a[0] = &oA; a[1] = &qA; a[2] = &kA; a[3] = &vA; a[4] = &hqA; a[5] = &tqA; a[6] = &hkvA; a[7] = &lkA; a[8] = &kvLenA;
        a[9] = &groupA; a[10] = &offA; a[11] = &scaleA; a[12] = &winA;
        CudaDriverApi.cuLaunchKernel(fn, (uint)((tq + 63) / 64), (uint)hq, (uint)b, 128, 1, 1, (uint)Fa2SharedBytes(d), stream, (nint)a, 0)
            .ThrowOnError();
    }

    /// <summary>Tight F16 copy of the first <paramref name="kvLen"/> positions of an F32 key/value cache laid out [rows, keyStride, d].</summary>
    public unsafe void LaunchKvToF16(ulong dstK, ulong dstV, ulong srcK, ulong srcV, int rows, int kvLen, int keyStride, int d, nint stream)
    {
        if (_kvToF16 == 0) throw new InvalidOperationException("flash_attn_causal_f16.ptx not present in the Ptx folder.");
        ulong dk = dstK, dv = dstV, sk = srcK, sv = srcV;
        int kvA = kvLen, lkA = keyStride, dA = d;
        void** a = stackalloc void*[7];
        a[0] = &dk; a[1] = &dv; a[2] = &sk; a[3] = &sv; a[4] = &kvA; a[5] = &lkA; a[6] = &dA;
        long per = (long)kvLen * d / 4;
        CudaDriverApi.cuLaunchKernel(_kvToF16, (uint)rows, (uint)((per + 255) / 256), 1, 256, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Partial online-softmax states of one decode row per sequence, one block per (split, KV head, sequence); merge them with the flash-attention combine kernel.</summary>
    /// <param name="devicePos">Device pointer to {kvLen, qOffset} for graph replay, or 0 to use the host values.</param>
    public unsafe void LaunchFlashDecodeGqa(ulong partialM, ulong partialL, ulong partialAcc, ulong q, ulong k, ulong v, int b, int hq, int d,
        int hkv, int keyStride, int kvLen, int kvGroup, int qOffset, float scale, float softcap, int window, int splits, int chunk,
        ulong devicePos, bool f16Kv, nint stream)
    {
        nint fn = f16Kv ? (d == 128 ? _decF16D128 : _decF16D64) : (d == 128 ? _decF32D128 : _decF32D64);
        if (fn == 0 || !DecodeGqaSupports(d, kvGroup)) throw new InvalidOperationException($"flash_attn_decode_gqa has no kernel for head dim {d}, group {kvGroup}.");
        ulong mA = partialM, lA = partialL, accA = partialAcc, qA = q, posA = devicePos, kA = k, vA = v;
        int hqA = hq, hkvA = hkv, lkA = keyStride, kvLenA = kvLen, groupA = kvGroup, offA = qOffset, winA = window, gA = splits, chunkA = chunk;
        float scaleA = scale, capA = softcap;
        void** a = stackalloc void*[18];
        a[0] = &mA; a[1] = &lA; a[2] = &accA; a[3] = &qA; a[4] = &hqA; a[5] = &hkvA; a[6] = &lkA; a[7] = &kvLenA; a[8] = &groupA;
        a[9] = &offA; a[10] = &scaleA; a[11] = &capA; a[12] = &winA; a[13] = &gA; a[14] = &chunkA; a[15] = &posA; a[16] = &kA; a[17] = &vA;
        CudaDriverApi.cuLaunchKernel(fn, (uint)splits, (uint)hkv, (uint)b, 256, 1, 1, (uint)DecodeSharedBytes(d), stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Fused graph-decode QKV epilogue writing an F16 key/value cache (q stays F32).</summary>
    public unsafe void LaunchQkvRopeScatterF16Kv(ulong qOut, ulong kCache, ulong vCache, ulong qIn, ulong kIn, ulong vIn,
        ulong cos, ulong sin, int nq, int nkv, int headDim, int rotaryDim, bool interleaved, int maxSeq, ulong devicePos, nint stream)
    {
        ulong qA = qOut, kA = kCache, vA = vCache, qiA = qIn, kiA = kIn, viA = vIn, cA = cos, sA = sin, dpA = devicePos;
        uint nqA = (uint)nq, nkvA = (uint)nkv, dA = (uint)headDim, rdA = (uint)rotaryDim, msA = (uint)maxSeq;
        int ilA = interleaved ? 1 : 0;
        void** args = stackalloc void*[15];
        args[0] = &qA; args[1] = &kA; args[2] = &vA; args[3] = &qiA; args[4] = &kiA; args[5] = &viA;
        args[6] = &cA; args[7] = &sA;
        args[8] = &nqA; args[9] = &nkvA; args[10] = &dA; args[11] = &rdA; args[12] = &ilA; args[13] = &msA; args[14] = &dpA;
        long total = ((long)nq + 2L * nkv) * headDim;
        uint grid = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_kvScatterQkvF16, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Fused per-head QK-norm + RoPE + scatter into an F16 key/value cache (q stays F32).</summary>
    public unsafe void LaunchQkNormRopeScatterF16Kv(ulong qOut, ulong kCache, ulong vCache, ulong qIn, ulong kIn, ulong vIn,
        ulong qNormW, ulong kNormW, ulong cos, ulong sin, int nq, int nkv, int headDim, int rotaryDim,
        bool interleaved, float eps, int maxSeq, ulong devicePos, nint stream)
    {
        ulong qA = qOut, kA = kCache, vA = vCache, qiA = qIn, kiA = kIn, viA = vIn, qwA = qNormW, kwA = kNormW,
            cA = cos, sA = sin, dpA = devicePos;
        uint nqA = (uint)nq, nkvA = (uint)nkv, dA = (uint)headDim, rdA = (uint)rotaryDim, msA = (uint)maxSeq;
        int ilA = interleaved ? 1 : 0;
        float epsA = eps;
        void** args = stackalloc void*[18];
        args[0] = &qA; args[1] = &kA; args[2] = &vA; args[3] = &qiA; args[4] = &kiA; args[5] = &viA;
        args[6] = &qwA; args[7] = &kwA; args[8] = &cA; args[9] = &sA;
        args[10] = &nqA; args[11] = &nkvA; args[12] = &dA; args[13] = &rdA; args[14] = &ilA;
        args[15] = &epsA; args[16] = &msA; args[17] = &dpA;
        uint grid = (uint)(nq + 2 * nkv);
        uint sharedMem = BlockSize * sizeof(float);
        CudaDriverApi.cuLaunchKernel(_kvScatterQkNormF16, grid, 1, 1, BlockSize, 1, 1, sharedMem, stream, (nint)args, 0).ThrowOnError();
    }
}
