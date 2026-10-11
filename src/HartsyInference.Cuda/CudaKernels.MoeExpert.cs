using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

// Expert-indexed dp4a GEMV kernels (Kernels/moe/moe_id_*.cu) and the slot combine (moe_combine_slots.cu): the routed-expert
// stage of MoE decode with the expert chosen from device memory, so it needs no host round trip and can be graph-captured.
public sealed partial class CudaKernels
{
    private CudaModule? _moeIdQ4kModule, _moeIdQ6kModule, _moeIdQ8_0Module, _moeCombineSlotsModule;
    private nint _moeGateUpIdQ4k, _moeDownIdQ4k, _moeGateUpIdQ6k, _moeDownIdQ6k, _moeGateUpIdQ8_0, _moeDownIdQ8_0;
    private nint _moeDownIdKsplitQ4k, _moeDownIdKsplitQ6k, _moeDownIdKsplitQ8_0;
    private nint _moeCombineSlotsF32;
    private CudaModule? _moePrefillModule;
    private nint _moeGatherRows16, _moeActMul16, _moeActMul16x16, _moeCombinePairsF32, _moeCastF16Bf16InPlace;

    /// <summary>True when the indexed expert GEMV exists for experts stored as <paramref name="dtype"/>.</summary>
    public bool HasMoeExpertKernel(DType dtype) => dtype switch
    {
        _ when dtype == DType.Q4_K => _moeGateUpIdQ4k != 0 && _moeDownIdQ4k != 0,
        _ when dtype == DType.Q6_K => _moeGateUpIdQ6k != 0 && _moeDownIdQ6k != 0,
        _ when dtype == DType.Q8_0 => _moeGateUpIdQ8_0 != 0 && _moeDownIdQ8_0 != 0,
        _ => false,
    };

    /// <summary>True when moe_combine_slots.ptx loaded.</summary>
    public bool HasMoeCombineSlots => _moeCombineSlotsF32 != 0;

    /// <summary>True when moe_prefill.ptx loaded: the gather, activation product and pair combine around the per-expert GEMMs.</summary>
    public bool HasMoePrefillKernels => _moeGatherRows16 != 0 && _moeActMul16 != 0 && _moeActMul16x16 != 0 && _moeCombinePairsF32 != 0;

    private void LoadMoeExpertKernels()
    {
        string q4k = Ptx("moe_id_q4k");
        if (File.Exists(q4k))
        {
            _moeIdQ4kModule = LoadOwnedModule(q4k);
            _moeGateUpIdQ4k = _moeIdQ4kModule.GetFunction("moe_gateup_id_q4k");
            _moeDownIdQ4k = _moeIdQ4kModule.GetFunction("moe_down_id_q4k");
            _moeDownIdKsplitQ4k = _moeIdQ4kModule.GetFunction("moe_down_id_ksplit_q4k");
        }
        string q6k = Ptx("moe_id_q6k");
        if (File.Exists(q6k))
        {
            _moeIdQ6kModule = LoadOwnedModule(q6k);
            _moeGateUpIdQ6k = _moeIdQ6kModule.GetFunction("moe_gateup_id_q6k");
            _moeDownIdQ6k = _moeIdQ6kModule.GetFunction("moe_down_id_q6k");
            _moeDownIdKsplitQ6k = _moeIdQ6kModule.GetFunction("moe_down_id_ksplit_q6k");
        }
        string q8 = Ptx("moe_id_q8_0");
        if (File.Exists(q8))
        {
            _moeIdQ8_0Module = LoadOwnedModule(q8);
            _moeGateUpIdQ8_0 = _moeIdQ8_0Module.GetFunction("moe_gateup_id_q8_0");
            _moeDownIdQ8_0 = _moeIdQ8_0Module.GetFunction("moe_down_id_q8_0");
            _moeDownIdKsplitQ8_0 = _moeIdQ8_0Module.GetFunction("moe_down_id_ksplit_q8_0");
        }
        string combine = Ptx("moe_combine_slots");
        if (File.Exists(combine))
        {
            _moeCombineSlotsModule = LoadOwnedModule(combine);
            _moeCombineSlotsF32 = _moeCombineSlotsModule.GetFunction("moe_combine_slots_f32");
        }
        string prefill = Ptx("moe_prefill");
        if (File.Exists(prefill))
        {
            _moePrefillModule = LoadOwnedModule(prefill);
            _moeGatherRows16 = _moePrefillModule.GetFunction("moe_gather_rows_16");
            _moeActMul16 = _moePrefillModule.GetFunction("moe_act_mul_16");
            _moeActMul16x16 = _moePrefillModule.GetFunction("moe_act_mul_16x16");
            _moeCombinePairsF32 = _moePrefillModule.GetFunction("moe_combine_pairs_f32");
            _moeCastF16Bf16InPlace = _moePrefillModule.GetFunction("moe_cast_f16_to_bf16_inplace");
        }
    }

    /// <summary>Gate and up projections with the activation fused, for every (token, slot) row: one warp per output row.</summary>
    /// <param name="n">Output rows per expert (the expert intermediate size).</param>
    /// <param name="k">Input length (the hidden size).</param>
    /// <param name="rows">Tokens times <paramref name="topk"/>.</param>
    public unsafe void LaunchMoeGateUpId(DType format, ulong act, ulong xq, ulong xd, ulong xs, ulong gateW, ulong upW, ulong ids,
        long expertStride, int n, int k, int topk, int rows, int numExperts, bool gelu, nint stream)
    {
        nint fn = format == DType.Q4_K ? _moeGateUpIdQ4k : format == DType.Q6_K ? _moeGateUpIdQ6k : _moeGateUpIdQ8_0;
        if (fn == 0) throw new InvalidOperationException($"moe_id kernels for {format} are not present in the Ptx folder.");
        ulong actA = act, xqA = xq, xdA = xd, xsA = xs, gA = gateW, uA = upW, iA = ids;
        long strideA = expertStride;
        int nA = n, kA = k, tA = topk, eA = numExperts, geluA = gelu ? 1 : 0;
        void** a = stackalloc void*[13];
        a[0] = &actA; a[1] = &xqA; a[2] = &xdA; a[3] = &xsA; a[4] = &gA; a[5] = &uA; a[6] = &iA; a[7] = &strideA;
        a[8] = &nA; a[9] = &kA; a[10] = &tA; a[11] = &eA; a[12] = &geluA;
        uint wpb = (uint)_wpbOverride;
        CudaDriverApi.cuLaunchKernel(fn, ((uint)n + wpb - 1) / wpb, (uint)rows, 1, 32, wpb, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Down projection for every (token, slot) row, the activation already quantized per row.</summary>
    public unsafe void LaunchMoeDownId(DType format, ulong output, ulong xq, ulong xd, ulong xs, ulong downW, ulong ids,
        long expertStride, int n, int k, int rows, int numExperts, nint stream)
    {
        nint fn = format == DType.Q4_K ? _moeDownIdQ4k : format == DType.Q6_K ? _moeDownIdQ6k : _moeDownIdQ8_0;
        if (fn == 0) throw new InvalidOperationException($"moe_id kernels for {format} are not present in the Ptx folder.");
        ulong oA = output, xqA = xq, xdA = xd, xsA = xs, wA = downW, iA = ids;
        long strideA = expertStride;
        int nA = n, kA = k, eA = numExperts;
        void** a = stackalloc void*[10];
        a[0] = &oA; a[1] = &xqA; a[2] = &xdA; a[3] = &xsA; a[4] = &wA; a[5] = &iA; a[6] = &strideA;
        a[7] = &nA; a[8] = &kA; a[9] = &eA;
        uint ksplit = MoeDownKsplitWarps(n, k, rows);
        nint split = format == DType.Q4_K ? _moeDownIdKsplitQ4k : format == DType.Q6_K ? _moeDownIdKsplitQ6k : _moeDownIdKsplitQ8_0;
        if (ksplit > 1 && split != 0)
        {
            CudaDriverApi.cuLaunchKernel(split, (uint)n, (uint)rows, 1, 32, ksplit, 1, 0, stream, (nint)a, 0).ThrowOnError();
            return;
        }
        uint wpb = (uint)_wpbOverride;
        CudaDriverApi.cuLaunchKernel(fn, ((uint)n + wpb - 1) / wpb, (uint)rows, 1, 32, wpb, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    // Warps per output row for the expert-indexed down projection. Long-K rows (Mixtral's ffn_down, K = 14336) run one row per
    // block split across 4 warps when the launch has few rows; numerics.gemvKsplit forces it off (0) or to W warps like the
    // dense GEMV's split.
    private static uint MoeDownKsplitWarps(int n, int k, int rows)
    {
        if (_ksplitOverride == 0) return 1;
        if (_ksplitOverride > 1) return (uint)Math.Min(_ksplitOverride, 16);
        return k >= 8192 && (long)n * rows <= 65536 ? 4u : 1u;
    }

    /// <summary>Weighted sum of each token's routed rows plus the optional shared-expert row; <paramref name="shared"/> and
    /// <paramref name="sharedGateLogit"/> are 0 when absent.</summary>
    public unsafe void LaunchMoeCombineSlots(ulong output, ulong slotOut, ulong topkWeight, ulong shared, ulong sharedGateLogit,
        int tokens, int hidden, int topk, nint stream)
    {
        if (_moeCombineSlotsF32 == 0) throw new InvalidOperationException("moe_combine_slots.ptx not present in the Ptx folder.");
        ulong oA = output, sA = slotOut, wA = topkWeight, shA = shared, gA = sharedGateLogit;
        int hA = hidden, kA = topk;
        void** a = stackalloc void*[7];
        a[0] = &oA; a[1] = &sA; a[2] = &wA; a[3] = &shA; a[4] = &gA; a[5] = &hA; a[6] = &kA;
        CudaDriverApi.cuLaunchKernel(_moeCombineSlotsF32, (uint)tokens, ((uint)hidden + 255) / 256, 1, 256, 1, 1, 0, stream, (nint)a, 0)
            .ThrowOnError();
    }

    /// <summary>Gathers activation rows into expert-major order as 16-bit values: <c>out[r] = x[perm[r]]</c>, zero where <c>perm[r] &lt; 0</c>.</summary>
    public unsafe void LaunchMoeGatherRows16(ulong output, ulong x, ulong perm, int rows, int hidden, bool bf16, nint stream)
    {
        if (_moeGatherRows16 == 0) throw new InvalidOperationException("moe_prefill.ptx not present in the Ptx folder.");
        ulong oA = output, xA = x, pA = perm;
        int rA = rows, hA = hidden, bA = bf16 ? 1 : 0;
        void** a = stackalloc void*[6];
        a[0] = &oA; a[1] = &xA; a[2] = &pA; a[3] = &rA; a[4] = &hA; a[5] = &bA;
        CudaDriverApi.cuLaunchKernel(_moeGatherRows16, (uint)rows, 1, 1, 256, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>16-bit <c>act(gate) * up</c> over <paramref name="count"/> values.</summary>
    public unsafe void LaunchMoeActMul16(ulong output, ulong gate, ulong up, long count, bool gelu, bool bf16, nint stream)
    {
        if (_moeActMul16 == 0) throw new InvalidOperationException("moe_prefill.ptx not present in the Ptx folder.");
        ulong oA = output, gA = gate, uA = up;
        long cA = count;
        int geluA = gelu ? 1 : 0, bA = bf16 ? 1 : 0;
        void** a = stackalloc void*[6];
        a[0] = &oA; a[1] = &gA; a[2] = &uA; a[3] = &cA; a[4] = &geluA; a[5] = &bA;
        CudaDriverApi.cuLaunchKernel(_moeActMul16, (uint)((count + 255) / 256), 1, 1, 256, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Weighted sum of each token's expert-major rows via the pair-slot map, plus the optional shared-expert row.</summary>
    public unsafe void LaunchMoeCombinePairs(ulong output, ulong expertOut, ulong pairSlot, ulong topkWeight, ulong shared,
        ulong sharedGateLogit, int tokens, int hidden, int topk, int expertRows, nint stream)
    {
        if (_moeCombinePairsF32 == 0) throw new InvalidOperationException("moe_prefill.ptx not present in the Ptx folder.");
        ulong oA = output, eA = expertOut, sA = pairSlot, wA = topkWeight, shA = shared, gA = sharedGateLogit;
        int hA = hidden, kA = topk, rA = expertRows;
        void** a = stackalloc void*[9];
        a[0] = &oA; a[1] = &eA; a[2] = &sA; a[3] = &wA; a[4] = &shA; a[5] = &gA; a[6] = &hA; a[7] = &kA; a[8] = &rA;
        CudaDriverApi.cuLaunchKernel(_moeCombinePairsF32, (uint)tokens, ((uint)hidden + 255) / 256, 1, 256, 1, 1, 0, stream, (nint)a, 0)
            .ThrowOnError();
    }

    /// <summary>16-bit <c>act(gate) * up</c> where gate and up are 16-bit too.</summary>
    public unsafe void LaunchMoeActMul16x16(ulong output, ulong gate, ulong up, long count, bool gelu, bool bf16, nint stream)
    {
        if (_moeActMul16x16 == 0) throw new InvalidOperationException("moe_prefill.ptx not present in the Ptx folder.");
        ulong oA = output, gA = gate, uA = up;
        long cA = count;
        int geluA = gelu ? 1 : 0, bA = bf16 ? 1 : 0;
        void** a = stackalloc void*[6];
        a[0] = &oA; a[1] = &gA; a[2] = &uA; a[3] = &cA; a[4] = &geluA; a[5] = &bA;
        CudaDriverApi.cuLaunchKernel(_moeActMul16x16, (uint)((count + 255) / 256), 1, 1, 256, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>True when moe_prefill.ptx carries the in-place F16 to BF16 cast.</summary>
    public bool HasCastF16ToBf16InPlace => _moeCastF16Bf16InPlace != 0;

    /// <summary>Converts <paramref name="count"/> F16 values at <paramref name="buffer"/> to BF16 in place.</summary>
    public unsafe void LaunchCastF16ToBf16InPlace(ulong buffer, long count, nint stream)
    {
        if (_moeCastF16Bf16InPlace == 0) throw new InvalidOperationException("moe_prefill.ptx not present in the Ptx folder.");
        ulong bA = buffer;
        long cA = count;
        void** a = stackalloc void*[2];
        a[0] = &bA; a[1] = &cA;
        long threads = Math.Max(count >> 3, 8);
        CudaDriverApi.cuLaunchKernel(_moeCastF16Bf16InPlace, (uint)((threads + 255) / 256), 1, 1, 256, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }
}
