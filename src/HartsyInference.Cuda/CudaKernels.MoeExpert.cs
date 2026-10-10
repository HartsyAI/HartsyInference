using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

// Expert-indexed dp4a GEMV kernels (Kernels/moe/moe_id_*.cu) and the slot combine (moe_combine_slots.cu): the routed-expert
// stage of MoE decode with the expert chosen from device memory, so it needs no host round trip and can be graph-captured.
public sealed partial class CudaKernels
{
    private CudaModule? _moeIdQ4kModule, _moeIdQ6kModule, _moeIdQ8_0Module, _moeCombineSlotsModule;
    private nint _moeGateUpIdQ4k, _moeDownIdQ4k, _moeGateUpIdQ6k, _moeDownIdQ6k, _moeGateUpIdQ8_0, _moeDownIdQ8_0;
    private nint _moeCombineSlotsF32;

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

    private void LoadMoeExpertKernels()
    {
        string q4k = Ptx("moe_id_q4k");
        if (File.Exists(q4k))
        {
            _moeIdQ4kModule = LoadOwnedModule(q4k);
            _moeGateUpIdQ4k = _moeIdQ4kModule.GetFunction("moe_gateup_id_q4k");
            _moeDownIdQ4k = _moeIdQ4kModule.GetFunction("moe_down_id_q4k");
        }
        string q6k = Ptx("moe_id_q6k");
        if (File.Exists(q6k))
        {
            _moeIdQ6kModule = LoadOwnedModule(q6k);
            _moeGateUpIdQ6k = _moeIdQ6kModule.GetFunction("moe_gateup_id_q6k");
            _moeDownIdQ6k = _moeIdQ6kModule.GetFunction("moe_down_id_q6k");
        }
        string q8 = Ptx("moe_id_q8_0");
        if (File.Exists(q8))
        {
            _moeIdQ8_0Module = LoadOwnedModule(q8);
            _moeGateUpIdQ8_0 = _moeIdQ8_0Module.GetFunction("moe_gateup_id_q8_0");
            _moeDownIdQ8_0 = _moeIdQ8_0Module.GetFunction("moe_down_id_q8_0");
        }
        string combine = Ptx("moe_combine_slots");
        if (File.Exists(combine))
        {
            _moeCombineSlotsModule = LoadOwnedModule(combine);
            _moeCombineSlotsF32 = _moeCombineSlotsModule.GetFunction("moe_combine_slots_f32");
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
        uint wpb = (uint)_wpbOverride;
        CudaDriverApi.cuLaunchKernel(fn, ((uint)n + wpb - 1) / wpb, (uint)rows, 1, 32, wpb, 1, 0, stream, (nint)a, 0).ThrowOnError();
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
}
