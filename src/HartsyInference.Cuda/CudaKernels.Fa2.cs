namespace HartsyInference.Cuda;

// Causal grouped-query FlashAttention-2 for LLM prefill (Kernels/lm/flash_attn_causal_f16.cu): F16 tensor cores, F32 accumulation.
public sealed partial class CudaKernels
{
    private CudaModule? _fa2Module, _decodeGqaModule, _decodeGqaPagedModule, _decodeLengthsModule, _decodeMergeModule, _kvScatterF16Module, _normFastModule;
    private readonly nint[] _normFast = new nint[6], _addNormFast = new nint[6];   // index log2(K): K = 2, 4, 8, 16, 32 -> 1..5

    /// <summary>Test hook: route the RMSNorm Q8 launchers to the reference kernels even when the fast ones loaded.</summary>
    internal bool ForceReferenceNorm { get; set; }
    private nint _kvScatterQkvF16, _kvScatterQkNormF16, _kvAppendF16Sat, _castF32ToF16Sat;

    /// <summary>True when kv_scatter_f16.ptx loaded its saturating F32-to-F16 cast (the cuDNN SDPA input cast).</summary>
    public bool HasCastF32ToF16Sat => _castF32ToF16Sat != 0;

    /// <summary>True when kv_scatter_f16.ptx loaded: the fused graph-decode scatter kernels for an F16 key/value cache.</summary>
    public bool HasKvScatterF16 => _kvScatterQkvF16 != 0 && _kvScatterQkNormF16 != 0;
    private nint _fa2D128, _fa2D64, _kvToF16;
    private nint _decF32D128, _decF32D64, _decF16D128, _decF16D64;

    /// <summary>True when flash_attn_decode_gqa.ptx loaded.</summary>
    public bool HasDecodeGqa => _decF32D128 != 0 && _decF32D64 != 0 && _decF16D128 != 0 && _decF16D64 != 0;

    private nint _decPagedD128, _decPagedD64;

    /// <summary>True when flash_attn_decode_gqa_paged.ptx loaded: the grouped-query decode kernel that follows a page table.
    /// A build without the file leaves this false, and a paged call is then gathered into a contiguous buffer.</summary>
    public bool HasDecodeGqaPaged => _decPagedD128 != 0 && _decPagedD64 != 0;

    // Index: cache type (0 = F32, 1 = F16, 2 = BF16) * 2 + (head dim 128 ? 0 : 1).
    private readonly nint[] _decLengths = new nint[6];
    private nint _decLengthsCombine;

    /// <summary>True when flash_attn_decode_lengths.ptx loaded: the grouped-query decode that takes a key length per sequence
    /// and returns the LSE, over an F32 cache at least. A build without the file leaves this false, and such a call is then
    /// refused.</summary>
    public bool HasDecodeLengths => _decLengthsCombine != 0 && _decLengths[0] != 0 && _decLengths[1] != 0;

    /// <summary>Whether the loaded lengths decode reads keys and values of <paramref name="cache"/>: F32, and F16 and BF16
    /// when the PTX file has those entry points. A file built before them leaves the half types unread, and a call over
    /// such a cache is then refused; the F32 kernel and the rest of the backend are unaffected.</summary>
    public bool DecodeLengthsReads(HartsyInference.Core.Tensors.DType cache)
    {
        int type = DecodeLengthsCacheIndex(cache);
        return type >= 0 && _decLengths[type * 2] != 0 && _decLengths[type * 2 + 1] != 0;
    }

    private static int DecodeLengthsCacheIndex(HartsyInference.Core.Tensors.DType cache) =>
        cache == HartsyInference.Core.Tensors.DType.F32 ? 0 : cache == HartsyInference.Core.Tensors.DType.F16 ? 1
        : cache == HartsyInference.Core.Tensors.DType.BF16 ? 2 : -1;

    private nint _decMerge;

    /// <summary>True when flash_attn_decode_merge.ptx loaded: the speculative-decode merge. A build without the file leaves
    /// this false, and a merge is then refused.</summary>
    public bool HasDecodeMerge => _decMerge != 0;

    /// <summary>Head dimension and query-head group the grouped-query decode kernel is built for.</summary>
    public static bool DecodeGqaSupports(int d, int group) => (d == 128 || d == 64) && group >= 1 && group <= 16;

    /// <summary>Dynamic shared memory of one decode block: the query heads, a K tile (padded rows), a V tile and the probabilities.</summary>
    private static int DecodeSharedBytes(int d, int kvGroup, bool f16Kv) =>
        (Math.Min(kvGroup, 16) * d + 8 * 32) * sizeof(float) + (32 * (d + 4) + 32 * d) * (f16Kv ? sizeof(ushort) : sizeof(float));

    /// <summary>True when flash_attn_causal_f16.ptx loaded and the device has the m16n8k16 F16 tensor-core instruction (sm_80 and later).</summary>
    public bool HasFa2Causal => _fa2D128 != 0 && _fa2D64 != 0 && _kvToF16 != 0 && CudaArch.HasMmaK16(Sm);

    private void LoadFa2Kernels()
    {
        // Each module gates on its own load. A missing FA2 module (sm_80-only, so no Turing build) must not skip the modules
        // below it, which Turing runs and the F16 KV cache and cuDNN rows depend on.
        string path = Ptx("flash_attn_causal_f16");
        if (CanLoad(path))
        {
            _fa2Module = LoadOwnedModule(path);
            _fa2D128 = _fa2Module.GetFunction("lm_fa2_causal_d128");
            _fa2D64 = _fa2Module.GetFunction("lm_fa2_causal_d64");
            _kvToF16 = _fa2Module.GetFunction("lm_kv_to_f16");
        }
        string scatter = Ptx("kv_scatter_f16");
        if (CanLoad(scatter))
        {
            _kvScatterF16Module = LoadOwnedModule(scatter);
            _kvScatterQkvF16 = _kvScatterF16Module.GetFunction("lm_qkv_rope_scatter_f16kv");
            _kvScatterQkNormF16 = _kvScatterF16Module.GetFunction("lm_qknorm_rope_scatter_f16kv");
            _kvAppendF16Sat = _kvScatterF16Module.GetFunction("lm_kv_append_f16_sat");
            // A kv_scatter_f16.ptx built before this entry point lacks it: leave the handle at zero so HasCastF32ToF16Sat is false
            // and the cuDNN SDPA path declines, instead of failing the whole backend's construction.
            try { _castF32ToF16Sat = _kvScatterF16Module.GetFunction("cast_f32_to_f16_sat"); }
            catch (CudaException) { _castF32ToF16Sat = 0; }
        }
        string normFast = Ptx("lm_norm_q8_fast");
        if (CanLoad(normFast))
        {
            _normFastModule = LoadOwnedModule(normFast);
            for (int log = 1; log <= 5; log++)
            {
                _normFast[log] = _normFastModule.GetFunction($"lm_rmsnorm_q8_1_fast_k{1 << log}");
                _addNormFast[log] = _normFastModule.GetFunction($"lm_add_rmsnorm_q8_1_fast_k{1 << log}");
            }
        }
        string decode = Ptx("flash_attn_decode_gqa");
        if (CanLoad(decode))
        {
            _decodeGqaModule = LoadOwnedModule(decode);
            _decF32D128 = _decodeGqaModule.GetFunction("lm_fa_decode_gqa_f32_d128");
            _decF32D64 = _decodeGqaModule.GetFunction("lm_fa_decode_gqa_f32_d64");
            _decF16D128 = _decodeGqaModule.GetFunction("lm_fa_decode_gqa_f16_d128");
            _decF16D64 = _decodeGqaModule.GetFunction("lm_fa_decode_gqa_f16_d64");
        }
        string paged = Ptx("flash_attn_decode_gqa_paged");
        if (CanLoad(paged))
        {
            _decodeGqaPagedModule = LoadOwnedModule(paged);
            _decPagedD128 = _decodeGqaPagedModule.GetFunction("lm_fa_decode_gqa_paged_f32_d128");
            _decPagedD64 = _decodeGqaPagedModule.GetFunction("lm_fa_decode_gqa_paged_f32_d64");
        }
        string lengths = Ptx("flash_attn_decode_lengths");
        if (CanLoad(lengths))
        {
            _decodeLengthsModule = LoadOwnedModule(lengths);
            string[] types = ["f32", "f16", "bf16"];
            for (int t = 0; t < types.Length; t++)
            {
                // The half-type entry points came after the F32 ones: a PTX file built before them leaves their handles at
                // zero instead of failing the backend's construction.
                try
                {
                    _decLengths[t * 2] = _decodeLengthsModule.GetFunction($"lm_fa_decode_lengths_{types[t]}_d128");
                    _decLengths[t * 2 + 1] = _decodeLengthsModule.GetFunction($"lm_fa_decode_lengths_{types[t]}_d64");
                }
                catch (CudaException) when (t > 0)
                {
                    _decLengths[t * 2] = _decLengths[t * 2 + 1] = 0;
                }
            }
            _decLengthsCombine = _decodeLengthsModule.GetFunction("lm_fa_decode_lengths_combine");
        }
        string merge = Ptx("flash_attn_decode_merge");
        if (CanLoad(merge))
        {
            _decodeMergeModule = LoadOwnedModule(merge);
            _decMerge = _decodeMergeModule.GetFunction("lm_fa_decode_merge_f32");
        }
    }

    /// <summary>Head dimensions the kernel is built for.</summary>
    public static bool Fa2SupportsHeadDim(int d) => d == 64 || d == 128;

    /// <summary>The device limit on a grid's y and z extents.</summary>
    public const int MaxGridYz = 65_535;

    /// <summary>Whether a launch's grid y and z extents stay inside <see cref="MaxGridYz"/>.</summary>
    public static bool GridYzFits(long y, long z) => y is > 0 and <= MaxGridYz && z is > 0 and <= MaxGridYz;

    /// <summary>Whether <see cref="LaunchKvToF16"/>'s grid covers <paramref name="kvLen"/> positions of <paramref name="d"/> dims.</summary>
    public static bool KvToF16GridFits(long kvLen, long d) => (kvLen * d / 4 + 255) / 256 <= MaxGridYz;

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
        CudaDriverApi.cuLaunchKernel(fn, (uint)splits, (uint)hkv, (uint)b, 256, 1, 1, (uint)DecodeSharedBytes(d, kvGroup, f16Kv), stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary><see cref="LaunchFlashDecodeGqa"/> for one sequence whose F32 keys and values are rows of page buffers:
    /// row <c>r</c> is buffer row <c>(pageTable[r >> pageShift] << pageShift) + (r mod pageSize)</c>.</summary>
    /// <param name="poolRows">Rows of each head in the page buffers.</param>
    /// <param name="pageTable">Device pointer to the sequence's I32 page table.</param>
    /// <param name="pageShift">log2 of the page size.</param>
    public unsafe void LaunchFlashDecodeGqaPaged(ulong partialM, ulong partialL, ulong partialAcc, ulong q, ulong k, ulong v, int hq, int d,
        int hkv, int poolRows, int kvLen, int kvGroup, int qOffset, float scale, float softcap, int window, int splits, int chunk,
        ulong pageTable, int pageShift, nint stream)
    {
        nint fn = d == 128 ? _decPagedD128 : _decPagedD64;
        if (fn == 0 || !DecodeGqaSupports(d, kvGroup)) throw new InvalidOperationException($"flash_attn_decode_gqa_paged has no kernel for head dim {d}, group {kvGroup}.");
        ulong mA = partialM, lA = partialL, accA = partialAcc, qA = q, tableA = pageTable, kA = k, vA = v;
        int hqA = hq, hkvA = hkv, lkA = poolRows, kvLenA = kvLen, groupA = kvGroup, offA = qOffset, winA = window, gA = splits, chunkA = chunk;
        int shiftA = pageShift;
        float scaleA = scale, capA = softcap;
        void** a = stackalloc void*[19];
        a[0] = &mA; a[1] = &lA; a[2] = &accA; a[3] = &qA; a[4] = &hqA; a[5] = &hkvA; a[6] = &lkA; a[7] = &kvLenA; a[8] = &groupA;
        a[9] = &offA; a[10] = &scaleA; a[11] = &capA; a[12] = &winA; a[13] = &gA; a[14] = &chunkA; a[15] = &tableA; a[16] = &shiftA;
        a[17] = &kA; a[18] = &vA;
        CudaDriverApi.cuLaunchKernel(fn, (uint)splits, (uint)hkv, 1, 256, 1, 1, (uint)DecodeSharedBytes(d, kvGroup, f16Kv: false), stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Partial states of a decode whose sequences have their own key lengths: one block per (split, KV head, sequence
    /// and query row). Query row <c>r</c> of sequence <c>b</c> reads the first <c>min(lengths[b], capacity)</c> rows of its
    /// cache, or under the causal rule the rows up to position <c>length − rows + r</c>.</summary>
    /// <param name="lengths">Device pointer to one I32 length per sequence.</param>
    /// <param name="cache">The type of the keys and values: F32, F16 or BF16.</param>
    public unsafe void LaunchFlashDecodeLengths(ulong partialM, ulong partialL, ulong partialAcc, ulong q, ulong k, ulong v, ulong lengths,
        int batch, int rows, int hq, int d, int hkv, int capacity, int kvGroup, bool causal, float scale, int splits, int chunk,
        HartsyInference.Core.Tensors.DType cache, nint stream)
    {
        int type = DecodeLengthsCacheIndex(cache);
        nint fn = type < 0 ? 0 : _decLengths[type * 2 + (d == 128 ? 0 : 1)];
        if (fn == 0 || !DecodeGqaSupports(d, kvGroup))
            throw new InvalidOperationException($"flash_attn_decode_lengths has no kernel for head dim {d}, group {kvGroup}, a {cache} cache.");
        ulong mA = partialM, lA = partialL, accA = partialAcc, qA = q, lenA = lengths, kA = k, vA = v;
        int hqA = hq, hkvA = hkv, lkA = capacity, sA = rows, groupA = kvGroup, causalA = causal ? 1 : 0, gA = splits, chunkA = chunk;
        float scaleA = scale;
        void** a = stackalloc void*[16];
        a[0] = &mA; a[1] = &lA; a[2] = &accA; a[3] = &qA; a[4] = &hqA; a[5] = &hkvA; a[6] = &lkA; a[7] = &lenA; a[8] = &sA;
        a[9] = &groupA; a[10] = &causalA; a[11] = &scaleA; a[12] = &gA; a[13] = &chunkA; a[14] = &kA; a[15] = &vA;
        CudaDriverApi.cuLaunchKernel(fn, (uint)splits, (uint)hkv, (uint)(batch * rows), 256, 1, 1,
            (uint)DecodeSharedBytes(d, kvGroup, f16Kv: type != 0), stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Merges the partial states <see cref="LaunchFlashDecodeLengths"/> wrote: the output as <c>[B, S, Hq × D]</c> and,
    /// when <paramref name="lse"/> is not 0, each row's LSE as <c>[B, Hq, S]</c>.</summary>
    public unsafe void LaunchFlashDecodeLengthsCombine(ulong output, ulong lse, ulong partialM, ulong partialL, ulong partialAcc, int batch,
        int rows, int hq, int d, int splits, nint stream)
    {
        if (_decLengthsCombine == 0) throw new InvalidOperationException("flash_attn_decode_lengths.ptx not present in the Ptx folder.");
        ulong oA = output, lseA = lse, mA = partialM, lA = partialL, accA = partialAcc;
        int hqA = hq, sA = rows, dA = d, gA = splits;
        void** a = stackalloc void*[9];
        a[0] = &oA; a[1] = &lseA; a[2] = &mA; a[3] = &lA; a[4] = &accA; a[5] = &hqA; a[6] = &sA; a[7] = &dA; a[8] = &gA;
        CudaDriverApi.cuLaunchKernel(_decLengthsCombine, (uint)((long)batch * rows * hq), 1, 1, (uint)d, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>The speculative-decode merge: one block per output row, in the <c>[B, S, Hq × D]</c> output's order.</summary>
    /// <param name="mask">Device pointer to one I32 mask per step row.</param>
    public unsafe void LaunchFlashDecodeMerge(ulong merged, ulong prefix, ulong lse, ulong q, ulong k, ulong v, ulong mask, int batch, int rows,
        int hq, int hkv, int d, float scale, nint stream)
    {
        if (_decMerge == 0) throw new InvalidOperationException("flash_attn_decode_merge.ptx not present in the Ptx folder.");
        ulong mA = merged, pA = prefix, lseA = lse, qA = q, kA = k, vA = v, maskA = mask;
        int hqA = hq, hkvA = hkv, sA = rows, dA = d;
        float scaleA = scale;
        void** a = stackalloc void*[12];
        a[0] = &mA; a[1] = &pA; a[2] = &lseA; a[3] = &qA; a[4] = &kA; a[5] = &vA; a[6] = &maskA; a[7] = &hqA; a[8] = &hkvA; a[9] = &sA;
        a[10] = &dA; a[11] = &scaleA;
        CudaDriverApi.cuLaunchKernel(_decMerge, (uint)((long)batch * rows * hq), 1, 1, 32, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
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

    /// <summary>The fast-kernel table slot for a row of <paramref name="normDim"/> elements, or 0 when the row is not 256 * 2^j, j in 1..5, or the fast kernels are off.</summary>
    private int NormFastSlot(int normDim)
    {
        if (ForceReferenceNorm || !HartsyInference.Core.Configuration.EngineKnobs.NormFast.Value || normDim % 256 != 0) return 0;
        int k = normDim / 256;
        if (k < 2 || k > 32 || (k & (k - 1)) != 0) return 0;
        int log = System.Numerics.BitOperations.Log2((uint)k);
        return _normFast[log] != 0 && _addNormFast[log] != 0 ? log : 0;
    }
}
