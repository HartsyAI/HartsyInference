namespace HartsyInference.Cuda;

// INT8 attention v2 (Kernels/attention/int8_attn_v2.cu): the stats, quant and attention launches over a caller's workspace.
public sealed partial class CudaKernels
{
    private const int Int8V2StatsThreads = 128, Int8V2QuantThreads = 256;
    private const int Int8V2QueryRowsPerQuantBlock = 64;

    private CudaModule? _int8V2Module;
    private nint _int8V2StatsF16, _int8V2StatsF32, _int8V2Ex2;
    private readonly nint[] _int8V2Quant = new nint[4];      // [f32 input · 2 + F16 PV]
    private readonly nint[] _int8V2Attention = new nint[12]; // [(BR 128) · 6 + pv · 2 + F32 output]
    private readonly nint[] _int8V2Trace = new nint[4];      // [(BR 128) · 2 + tile V scales]

    /// <summary>Whether int8_attn_v2.ptx loaded with its u8 PV entries.</summary>
    internal bool HasInt8AttentionV2 { get; private set; }

    /// <summary>Whether int8_attn_v2.ptx loaded with its F16 PV entries, which need the shared-memory opt-in.</summary>
    internal bool HasInt8AttentionV2F16Pv { get; private set; }

    /// <summary>Dynamic shared memory of one INT8 v2 attention block: the K and V tiles, the tile's K scales and per-tile V scales.</summary>
    internal static int Int8V2SharedBytes(Int8AttentionV2Pv pv) =>
        Int8AttentionV2Layout.BlockCols * Int8AttentionV2Layout.HeadDim * (pv == Int8AttentionV2Pv.F16 ? 3 : 2)
        + (Int8AttentionV2Layout.BlockCols + Int8AttentionV2Layout.HeadDim) * sizeof(float);

    private void LoadInt8AttentionV2Kernels()
    {
        string path = Ptx("int8_attn_v2");
        if (!CanLoad(path)) return;
        _int8V2Module = LoadOwnedModule(path);
        _int8V2StatsF16 = _int8V2Module.GetFunction("int8v2_stats_f16");
        _int8V2StatsF32 = _int8V2Module.GetFunction("int8v2_stats_f32");
        _int8V2Ex2 = _int8V2Module.GetFunction("int8v2_ex2_probe");
        _int8V2Quant[0] = _int8V2Module.GetFunction("int8v2_quant_f16");
        _int8V2Quant[1] = _int8V2Module.GetFunction("int8v2_quant_f16_pvf16");
        _int8V2Quant[2] = _int8V2Module.GetFunction("int8v2_quant_f32");
        _int8V2Quant[3] = _int8V2Module.GetFunction("int8v2_quant_f32_pvf16");
        string[] pvNames = ["vseq", "vtile", "pvf16"];
        for (int br = 0; br < 2; br++)
            for (int pv = 0; pv < 3; pv++)
                for (int f32 = 0; f32 < 2; f32++)
                {
                    nint fn = _int8V2Module.GetFunction($"int8v2_attn_br{(br == 0 ? 64 : 128)}_{pvNames[pv]}_{(f32 == 0 ? "f16o" : "f32o")}");
                    int bytes = Int8V2SharedBytes((Int8AttentionV2Pv)pv);
                    // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8; only the F16 PV tile needs more than the default.
                    if (bytes > DefaultDynamicSharedBytes)
                    {
                        if (bytes > MaxDynamicSharedBytes) continue;
                        CudaDriverApi.cuFuncSetAttribute(fn, 8, bytes).ThrowOnError();
                    }
                    _int8V2Attention[br * 6 + pv * 2 + f32] = fn;
                }
        for (int br = 0; br < 2; br++)
            for (int tile = 0; tile < 2; tile++)
                _int8V2Trace[br * 2 + tile] = _int8V2Module.GetFunction($"int8v2_attn_br{(br == 0 ? 64 : 128)}_{(tile == 0 ? "vseq" : "vtile")}_trace");
        HasInt8AttentionV2 = true;
        HasInt8AttentionV2F16Pv = true;
        for (int i = 0; i < _int8V2Attention.Length; i++)
        {
            if (_int8V2Attention[i] != 0) continue;
            if ((i / 2) % 3 == (int)Int8AttentionV2Pv.F16) HasInt8AttentionV2F16Pv = false;
            else HasInt8AttentionV2 = false;
        }
    }

    /// <summary>Every loaded INT8 v2 entry with the registers and local (spill) bytes the driver's JIT gave it on this card.</summary>
    internal IEnumerable<(string Name, int Registers, int LocalBytes)> Int8V2EntryResources()
    {
        if (_int8V2Module is null) yield break;
        string[] pvNames = ["vseq", "vtile", "pvf16"];
        List<(string, nint)> entries = [("int8v2_stats_f16", _int8V2StatsF16), ("int8v2_stats_f32", _int8V2StatsF32),
            ("int8v2_quant_f16", _int8V2Quant[0]), ("int8v2_quant_f16_pvf16", _int8V2Quant[1]), ("int8v2_quant_f32", _int8V2Quant[2]),
            ("int8v2_quant_f32_pvf16", _int8V2Quant[3])];
        for (int br = 0; br < 2; br++)
            for (int pv = 0; pv < 3; pv++)
                for (int f32 = 0; f32 < 2; f32++)
                    entries.Add(($"int8v2_attn_br{(br == 0 ? 64 : 128)}_{pvNames[pv]}_{(f32 == 0 ? "f16o" : "f32o")}", _int8V2Attention[br * 6 + pv * 2 + f32]));
        foreach ((string name, nint fn) in entries)
        {
            if (fn == 0) continue;
            // CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3, CU_FUNC_ATTRIBUTE_NUM_REGS = 4.
            CudaDriverApi.cuFuncGetAttribute(out int local, 3, fn).ThrowOnError();
            CudaDriverApi.cuFuncGetAttribute(out int regs, 4, fn).ThrowOnError();
            yield return (name, regs, local);
        }
    }

    /// <summary>The K channel sums and per-sequence V absmax per 64-key chunk; each (b, h)'s last block folds them into the K
    /// mean and V scales. Grid (chunks, H, B).</summary>
    internal unsafe void LaunchInt8V2Stats(bool f32, ulong k, FlashStrides kStrides, ulong v, FlashStrides vStrides, ulong workspace,
        in Int8AttentionV2Layout layout, int batch, int heads, int validKeys, nint stream)
    {
        ulong kA = k, vA = v, kPartial = workspace + (ulong)layout.KPartial, vPartial = workspace + (ulong)layout.VPartial;
        ulong kMean = workspace + (ulong)layout.KMean, vScale = workspace + (ulong)layout.VScale, counters = workspace;
        FlashStrides ks = kStrides, vs = vStrides;
        uint hA = (uint)heads, kvA = (uint)validKeys, chunks = (uint)layout.Chunks;
        uint seqV = layout.Pv == Int8AttentionV2Pv.U8SequenceScales ? 1u : 0u;
        void** a = stackalloc void*[13];
        a[0] = &kA; a[1] = &ks; a[2] = &vA; a[3] = &vs; a[4] = &kPartial; a[5] = &vPartial; a[6] = &kMean; a[7] = &vScale;
        a[8] = &counters; a[9] = &hA; a[10] = &kvA; a[11] = &chunks; a[12] = &seqV;
        CudaDriverApi.cuLaunchKernel(f32 ? _int8V2StatsF32 : _int8V2StatsF16, chunks, (uint)heads, (uint)batch, Int8V2StatsThreads, 1, 1,
            0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Q rows, K tiles and V tiles into the workspace's bytes and scales. Grid (Q blocks + 2·tiles, H, B).</summary>
    internal unsafe void LaunchInt8V2Quant(bool f32, ulong q, FlashStrides qStrides, ulong k, FlashStrides kStrides, ulong v,
        FlashStrides vStrides, ulong workspace, in Int8AttentionV2Layout layout, int batch, int heads, int queryTokens, int validKeys,
        nint stream)
    {
        bool pvF16 = layout.Pv == Int8AttentionV2Pv.F16;
        ulong qA = q, kA = k, vA = v;
        FlashStrides qs = qStrides, ks = kStrides, vs = vStrides;
        ulong q8 = workspace + (ulong)layout.Q8, qScale = workspace + (ulong)layout.QScale, k8 = workspace + (ulong)layout.K8;
        ulong kScale = workspace + (ulong)layout.KScale, v8 = workspace + (ulong)layout.V8, vScale = workspace + (ulong)layout.VScale;
        ulong kMean = workspace + (ulong)layout.KMean;
        uint hA = (uint)heads, sqA = (uint)queryTokens, kvA = (uint)validKeys, tiles = (uint)layout.Tiles;
        uint qBlocks = (uint)((queryTokens + Int8V2QueryRowsPerQuantBlock - 1) / Int8V2QueryRowsPerQuantBlock);
        uint vTile = layout.Pv == Int8AttentionV2Pv.U8TileScales ? 1u : 0u;
        void** a = stackalloc void*[20];
        a[0] = &qA; a[1] = &qs; a[2] = &kA; a[3] = &ks; a[4] = &vA; a[5] = &vs; a[6] = &q8; a[7] = &qScale; a[8] = &k8; a[9] = &kScale;
        a[10] = &v8; a[11] = &vScale; a[12] = &kMean; a[13] = &vScale; a[14] = &hA; a[15] = &sqA; a[16] = &kvA; a[17] = &tiles;
        a[18] = &qBlocks; a[19] = &vTile;
        CudaDriverApi.cuLaunchKernel(_int8V2Quant[(f32 ? 2 : 0) + (pvF16 ? 1 : 0)], qBlocks + 2 * tiles, (uint)heads, (uint)batch,
            Int8V2QuantThreads, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>The attention kernel over the quantized workspace into a strided output. Grid (⌈Sq/BR⌉, H, B), 2·BR threads.</summary>
    /// <param name="trace">A trace entry's (b, h, query tile) and buffers; null runs the production entry.</param>
    internal unsafe void LaunchInt8V2Attention(int blockRows, bool f32Out, ulong output, FlashStrides outStrides, ulong workspace,
        in Int8AttentionV2Layout layout, int batch, int heads, int queryTokens, int validKeys, float alphaLog2e, nint stream,
        Int8V2TraceArgs? trace = null)
    {
        if (blockRows is not (64 or 128)) throw new ArgumentOutOfRangeException(nameof(blockRows), blockRows, "INT8 attention v2 tiles are 64 or 128 rows.");
        int br = blockRows == 128 ? 1 : 0;
        nint fn;
        if (trace is not null)
        {
            if (layout.Pv == Int8AttentionV2Pv.F16 || !f32Out) throw new ArgumentException("Trace entries run u8 PV with F32 output.");
            fn = _int8V2Trace[br * 2 + (layout.Pv == Int8AttentionV2Pv.U8TileScales ? 1 : 0)];
        }
        else
        {
            fn = _int8V2Attention[br * 6 + (int)layout.Pv * 2 + (f32Out ? 1 : 0)];
        }
        if (fn == 0) throw new InvalidOperationException($"int8_attn_v2.ptx has no {layout.Pv} entry for this device.");
        ulong oA = output;
        FlashStrides os = outStrides;
        ulong q8 = workspace + (ulong)layout.Q8, qScale = workspace + (ulong)layout.QScale, k8 = workspace + (ulong)layout.K8;
        ulong kScale = workspace + (ulong)layout.KScale, v8 = workspace + (ulong)layout.V8, vScale = workspace + (ulong)layout.VScale;
        uint hA = (uint)heads, sqA = (uint)queryTokens, kvA = (uint)validKeys, tiles = (uint)layout.Tiles;
        float scaleA = alphaLog2e;
        Int8V2TraceArgs traceA = trace ?? default;
        void** a = stackalloc void*[14];
        a[0] = &oA; a[1] = &os; a[2] = &q8; a[3] = &qScale; a[4] = &k8; a[5] = &kScale; a[6] = &v8; a[7] = &vScale; a[8] = &hA;
        a[9] = &sqA; a[10] = &kvA; a[11] = &tiles; a[12] = &scaleA; a[13] = &traceA;
        CudaDriverApi.cuLaunchKernel(fn, (uint)((queryTokens + blockRows - 1) / blockRows), (uint)heads, (uint)batch,
            (uint)(2 * blockRows), 1, 1, (uint)Int8V2SharedBytes(layout.Pv), stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>The attention kernels' exp2 of <paramref name="count"/> F32 arguments.</summary>
    internal unsafe void LaunchInt8V2Ex2Probe(ulong output, ulong input, int count, nint stream)
    {
        ulong oA = output, iA = input;
        uint nA = (uint)count;
        void** a = stackalloc void*[3];
        a[0] = &oA; a[1] = &iA; a[2] = &nA;
        CudaDriverApi.cuLaunchKernel(_int8V2Ex2, (uint)((count + 255) / 256), 1, 1, 256, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }
}
