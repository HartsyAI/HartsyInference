using HartsyInference.Core.Configuration;

namespace HartsyInference.Cuda;

// Graph-decode fusions (Kernels/lm/lm_decode_fused.cu): each kernel is bit-identical to the launch chain it replaces.
public sealed partial class CudaKernels
{
    private CudaModule? _decodeFusedModule;
    // Indexed by K = normDim / 256 (2..32).
    private readonly nint[] _normWide = new nint[33], _addNormWide = new nint[33], _moeCombineAddNorm = new nint[33];
    private nint _qkNormFullF32, _qkNormFullF16, _faCombineQ8;

    private void LoadDecodeFusedKernels()
    {
        string path = Ptx("lm_decode_fused");
        if (!File.Exists(path)) return;
        _decodeFusedModule = LoadOwnedModule(path);
        for (int k = 2; k <= 32; k++)
        {
            _normWide[k] = _decodeFusedModule.GetFunction($"lm_rmsnorm_q8_1_wide_k{k}");
            _addNormWide[k] = _decodeFusedModule.GetFunction($"lm_add_rmsnorm_q8_1_wide_k{k}");
            _moeCombineAddNorm[k] = _decodeFusedModule.GetFunction($"moe_combine_add_rmsnorm_q8_1_k{k}");
        }
        _qkNormFullF32 = _decodeFusedModule.GetFunction("lm_qknorm_full_rope_scatter_f32");
        _qkNormFullF16 = _decodeFusedModule.GetFunction("lm_qknorm_full_rope_scatter_f16kv");
        _faCombineQ8 = _decodeFusedModule.GetFunction("lm_flash_attn_f32_combine_q8_1");
    }

    /// <summary>K = normDim / 256 when a wide kernel row of that width exists (2..32), else 0.</summary>
    private static int FastNormK(int normDim) => normDim % 256 == 0 && normDim / 256 is >= 2 and <= 32 ? normDim / 256 : 0;

    /// <summary>Threads per block of the wide norm kernels for row width 256·<paramref name="k"/> (WIDE_NT in lm_decode_fused.cu).</summary>
    private static uint WideThreads(int k) => k < 4 ? (uint)(256 * k) : 1024u;

    /// <summary>The wide RMSNorm + Q8_1 kernels (bit-identical to the reference ones) for any row width that is a multiple of 256 up to 8192;
    /// false when not applicable, and the caller launches the register-resident or shared-memory kernel instead.</summary>
    private unsafe bool TryLaunchNormWide(bool add, int normDim, int totalRows, void** args, nint stream)
    {
        if (!WideNormEnabled) return false;
        int k = FastNormK(normDim);
        nint fn = k == 0 ? 0 : add ? _addNormWide[k] : _normWide[k];
        if (fn == 0) return false;
        CudaDriverApi.cuLaunchKernel(fn, (uint)totalRows, 1, 1, WideThreads(k), 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
        return true;
    }

    /// <summary>The wide kernels run only while both norm switches are on and the reference test hook is off: <c>numerics.lmNormFast=0</c>
    /// still returns every RMSNorm Q8 launch to the shared-memory tree.</summary>
    private bool WideNormEnabled => !ForceReferenceNorm && EngineKnobs.NormFast.Value && EngineKnobs.NormWide.Value;

    /// <summary>True when <see cref="LaunchMoeCombineAddRmsNormQ8"/> has a kernel for this row width and the wide norms are enabled
    /// (the fused kernel carries the wide norm's body).</summary>
    public bool HasMoeCombineAddRmsNormQ8(int normDim)
    {
        int k = FastNormK(normDim);
        return k > 0 && _moeCombineAddNorm[k] != 0 && WideNormEnabled;
    }

    /// <summary>One token: <c>residOut = a + combine(slots)</c>, <c>normOut = rmsnorm(residOut)·w</c> and its Q8_1 sidecar (bit-identical to
    /// moe_combine_slots_f32 + the residual add + the add-RMSNorm Q8 kernel). Gate on <see cref="HasMoeCombineAddRmsNormQ8"/>.</summary>
    public unsafe void LaunchMoeCombineAddRmsNormQ8(ulong residOut, ulong normOut, ulong xq, ulong xd, ulong xs, ulong a,
        ulong slotOut, ulong topkWeight, ulong shared, ulong sharedGateLogit, int topk, ulong weight, int normDim, float eps, nint stream)
    {
        nint fn = _moeCombineAddNorm[FastNormK(normDim)];
        if (fn == 0) throw new InvalidOperationException($"moe_combine_add_rmsnorm_q8_1 has no kernel for width {normDim}.");
        ulong rA = residOut, nA = normOut, xqA = xq, xdA = xd, xsA = xs, aA = a, sA = slotOut, wA = topkWeight, shA = shared,
            gA = sharedGateLogit, nwA = weight;
        int kA = topk;
        uint dA = (uint)normDim;
        float eA = eps;
        void** args = stackalloc void*[14];
        args[0] = &rA; args[1] = &nA; args[2] = &xqA; args[3] = &xdA; args[4] = &xsA; args[5] = &aA; args[6] = &sA; args[7] = &wA;
        args[8] = &shA; args[9] = &gA; args[10] = &kA; args[11] = &nwA; args[12] = &dA; args[13] = &eA;
        CudaDriverApi.cuLaunchKernel(fn, 1, 1, 1, WideThreads(FastNormK(normDim)), 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>True when the full-width QK-norm scatter kernel for this cache storage is loaded.</summary>
    public bool HasQkNormFullRopeScatter(bool f16Kv) => (f16Kv ? _qkNormFullF16 : _qkNormFullF32) != 0;

    /// <summary>Full-width QK-RMSNorm + RoPE + KV scatter (bit-identical to slices + two dit_rmsnorm_f32 rows + the rope-scatter):
    /// grid <c>nq + 2·nkv</c>, 256 threads, each head block reducing its whole section.</summary>
    public unsafe void LaunchQkNormFullRopeScatter(bool f16Kv, ulong qOut, ulong kCache, ulong vCache, ulong qIn, ulong kIn, ulong vIn,
        ulong qNormW, ulong kNormW, ulong cos, ulong sin, int nq, int nkv, int headDim, int rotaryDim,
        bool interleaved, float eps, int maxSeq, ulong devicePos, nint stream)
    {
        nint fn = f16Kv ? _qkNormFullF16 : _qkNormFullF32;
        if (fn == 0) throw new InvalidOperationException("lm_decode_fused.ptx is not loaded.");
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
        CudaDriverApi.cuLaunchKernel(fn, grid, 1, 1, BlockSize, 1, 1, BlockSize * sizeof(float), stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>True when the attention combine can emit the Q8_1 sidecar of a <paramref name="headDim"/>-wide output.</summary>
    public bool HasFlashAttentionCombineQ8(int headDim) => _faCombineQ8 != 0 && headDim % 32 == 0;

    /// <summary><see cref="LaunchFlashAttentionCombine"/> plus the Q8_1 sidecar (xq/xd/xs) of its output, bit-identical F32 output.</summary>
    public unsafe void LaunchFlashAttentionCombineQ8(ulong outPtr, ulong xq, ulong xd, ulong xs, ulong partialM, ulong partialL,
        ulong partialAcc, int batch, int hq, int tq, int headDim, int splits, nint stream)
    {
        ValidateFlashAttentionHeadDim(headDim);
        if (!HasFlashAttentionCombineQ8(headDim)) throw new InvalidOperationException($"No Q8_1 attention combine for head dim {headDim}.");
        ulong outArg = outPtr, xqA = xq, xdA = xd, xsA = xs, pmArg = partialM, plArg = partialL, paArg = partialAcc;
        uint bArg = (uint)batch, hqArg = (uint)hq, tqArg = (uint)tq, dArg = (uint)headDim, gArg = (uint)splits;
        void** args = stackalloc void*[12];
        args[0] = &outArg; args[1] = &xqA; args[2] = &xdA; args[3] = &xsA; args[4] = &pmArg; args[5] = &plArg; args[6] = &paArg;
        args[7] = &bArg; args[8] = &hqArg; args[9] = &tqArg; args[10] = &dArg; args[11] = &gArg;
        uint blockThreads = 32;
        while (blockThreads < (uint)headDim) blockThreads <<= 1;
        CudaDriverApi.cuLaunchKernel(_faCombineQ8, (uint)((long)batch * hq * tq), 1, 1, blockThreads, 1, 1, 0, stream, (nint)args, 0)
            .ThrowOnError();
    }
}
