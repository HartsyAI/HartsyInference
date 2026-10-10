namespace HartsyInference.Cuda;

// Causal grouped-query FlashAttention-2 for LLM prefill (Kernels/lm/flash_attn_causal_f16.cu): F16 tensor cores, F32 accumulation.
public sealed partial class CudaKernels
{
    private CudaModule? _fa2Module;
    private nint _fa2D128, _fa2D64, _kvToF16;

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
}
