using HartsyInference.Core.Backends;

namespace HartsyInference.Cuda;

// Mixture-of-experts routing, dispatch, combine and top-k kernels (Kernels/moe, Kernels/lm/lm_topk_f32.cu).
public sealed partial class CudaKernels
{
    /// <summary>Widest row the radix top-k can order: its sort buffer is 2048 packed (key, index) entries in shared memory.</summary>
    public const int TopKMaxK = 2048;

    private CudaModule? _moeRouteModule;
    private nint _moeRouteF32;
    private nint _softplusF32;
    private CudaModule? _moeDispatchModule;
    private nint _moeDispatchCountI32;
    private nint _moeDispatchScanI32;
    private nint _moeDispatchScatterI32;
    private nint _moeCombineF32;
    private CudaModule? _topKModule;
    private nint _topKF32;

    /// <summary>True when moe_route.ptx and moe_dispatch.ptx loaded, so the MoE primitives can run on this device.</summary>
    public bool HasMoeKernels => _moeRouteF32 != 0 && _moeDispatchScatterI32 != 0;

    /// <summary>True when lm_topk_f32.ptx loaded.</summary>
    public bool HasTopKKernel => _topKF32 != 0;

    // Optional modules: absence leaves the primitives unsupported on this backend instead of failing construction.
    private void LoadMoeKernels()
    {
        string routePath = Ptx("moe_route");
        if (File.Exists(routePath))
        {
            _moeRouteModule = LoadOwnedModule(routePath);
            _moeRouteF32 = _moeRouteModule.GetFunction("moe_route_f32");
            _softplusF32 = _moeRouteModule.GetFunction("softplus_f32");
        }
        string dispatchPath = Ptx("moe_dispatch");
        if (File.Exists(dispatchPath))
        {
            _moeDispatchModule = LoadOwnedModule(dispatchPath);
            _moeDispatchCountI32 = _moeDispatchModule.GetFunction("moe_dispatch_count_i32");
            _moeDispatchScanI32 = _moeDispatchModule.GetFunction("moe_dispatch_scan_i32");
            _moeDispatchScatterI32 = _moeDispatchModule.GetFunction("moe_dispatch_scatter_i32");
            _moeCombineF32 = _moeDispatchModule.GetFunction("moe_combine_f32");
        }
        LoadMoeExpertKernels();
        string topKPath = Ptx("lm_topk_f32");
        if (File.Exists(topKPath))
        {
            _topKModule = LoadOwnedModule(topKPath);
            _topKF32 = _topKModule.GetFunction("lm_topk_f32");
        }
    }

    /// <summary>Routes <paramref name="tokens"/> tokens: one 128-thread block each. Optional pointers are 0 when absent.</summary>
    public unsafe void LaunchMoeRoute(ulong topkIdx, ulong topkWeight, ulong logits, ulong bias, ulong altBias,
        ulong tokenKinds, int tokens, in MoeRouteArgs args, nint stream)
    {
        if (_moeRouteF32 == 0) throw new InvalidOperationException("moe_route.ptx not present in the Ptx folder.");
        ulong idxA = topkIdx, wA = topkWeight, lA = logits, bA = bias, abA = altBias, kA = tokenKinds;
        int e = args.NumExperts, k = args.TopK, scoring = (int)args.Scoring, groups = args.GroupCount,
            kept = args.GroupsKept, renorm = args.Renormalize ? 1 : 0;
        float masked = args.MaskedGroupValue, eps = args.RenormEpsilon, scale = args.Scale, div = args.LogitDivisor;
        void** a = stackalloc void*[16];
        a[0] = &idxA; a[1] = &wA; a[2] = &lA; a[3] = &bA; a[4] = &abA; a[5] = &kA; a[6] = &e; a[7] = &k;
        a[8] = &scoring; a[9] = &groups; a[10] = &kept; a[11] = &masked; a[12] = &renorm; a[13] = &eps;
        a[14] = &scale; a[15] = &div;
        CudaDriverApi.cuLaunchKernel(_moeRouteF32, (uint)tokens, 1, 1, 128, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Elementwise softplus over <paramref name="count"/> floats; output may alias input.</summary>
    public unsafe void LaunchSoftplus(ulong output, ulong input, long count, nint stream)
    {
        if (_softplusF32 == 0) throw new InvalidOperationException("moe_route.ptx not present in the Ptx folder.");
        ulong oA = output, iA = input;
        long n = count;
        void** a = stackalloc void*[3];
        a[0] = &oA; a[1] = &iA; a[2] = &n;
        uint grid = (uint)((count + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_softplusF32, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Count, scan and scatter launches that build the expert-major dispatch from <paramref name="pairs"/> expert ids.</summary>
    public unsafe void LaunchMoeBuildDispatch(ulong counts, ulong offsets, ulong permutedToken, ulong pairSlot,
        ulong topkIdx, int pairs, int k, int numExperts, nint stream)
    {
        if (_moeDispatchScatterI32 == 0) throw new InvalidOperationException("moe_dispatch.ptx not present in the Ptx folder.");
        ulong cA = counts, oA = offsets, ptA = permutedToken, psA = pairSlot, iA = topkIdx;
        int pairsA = pairs, kA = k, eA = numExperts;
        void** count = stackalloc void*[3];
        count[0] = &cA; count[1] = &iA; count[2] = &pairsA;
        CudaDriverApi.cuLaunchKernel(_moeDispatchCountI32, (uint)numExperts, 1, 1, 256, 1, 1, 0, stream, (nint)count, 0).ThrowOnError();
        void** scan = stackalloc void*[3];
        scan[0] = &oA; scan[1] = &cA; scan[2] = &eA;
        CudaDriverApi.cuLaunchKernel(_moeDispatchScanI32, 1, 1, 1, 1, 1, 1, 0, stream, (nint)scan, 0).ThrowOnError();
        void** scatter = stackalloc void*[7];
        scatter[0] = &ptA; scatter[1] = &psA; scatter[2] = &iA; scatter[3] = &oA; scatter[4] = &pairsA;
        scatter[5] = &kA; scatter[6] = &eA;
        CudaDriverApi.cuLaunchKernel(_moeDispatchScatterI32, (uint)(numExperts + 1), 1, 1, 256, 1, 1, 0, stream, (nint)scatter, 0)
            .ThrowOnError();
    }

    /// <summary>Weighted gather-sum of each token's expert rows; one thread per (token, column). Slots outside
    /// [0, expertRows) are skipped (the CPU reference throws for slot >= expertRows).</summary>
    public unsafe void LaunchMoeCombine(ulong output, ulong expertOut, ulong pairSlot, ulong topkWeight, int tokens,
        int hidden, int k, bool accumulate, int expertRows, nint stream)
    {
        if (_moeCombineF32 == 0) throw new InvalidOperationException("moe_dispatch.ptx not present in the Ptx folder.");
        ulong oA = output, xA = expertOut, sA = pairSlot, wA = topkWeight;
        int hA = hidden, kA = k, accA = accumulate ? 1 : 0, rA = expertRows;
        void** a = stackalloc void*[8];
        a[0] = &oA; a[1] = &xA; a[2] = &sA; a[3] = &wA; a[4] = &hA; a[5] = &kA; a[6] = &accA; a[7] = &rA;
        uint gridY = (uint)((hidden + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_moeCombineF32, (uint)tokens, gridY, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Radix-select top-k, one block per row of an [rows, n] matrix.</summary>
    public unsafe void LaunchTopKLastDim(ulong values, ulong indices, ulong input, ulong validLengths, int rows,
        int n, int k, bool sortByIndex, nint stream)
    {
        if (_topKF32 == 0) throw new InvalidOperationException("lm_topk_f32.ptx not present in the Ptx folder.");
        ulong vA = values, iA = indices, xA = input, lA = validLengths;
        int nA = n, kA = k, sA = sortByIndex ? 1 : 0;
        void** a = stackalloc void*[7];
        a[0] = &vA; a[1] = &iA; a[2] = &xA; a[3] = &lA; a[4] = &nA; a[5] = &kA; a[6] = &sA;
        CudaDriverApi.cuLaunchKernel(_topKF32, (uint)rows, 1, 1, 256, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }
}
