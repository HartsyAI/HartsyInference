namespace HartsyInference.Cuda;

// Gated-DeltaNet prefill kernels (Kernels/lm/ssm_prefill_f32.cu): conv + split, per-head L2 norm, carry update,
// the row-parallel delta-rule scan and the gated norm, for a whole token sequence.
public sealed partial class CudaKernels
{
    /// <summary>State rows (warps) per block of the delta scan.</summary>
    private const uint SsmScanRowsPerBlock = 4;

    /// <summary>Rows (warps) per block of the row-wise L2 and gated norms.</summary>
    private const uint SsmRowsPerBlock = 8;

    /// <summary>Widest key head the scan holds in registers (eight columns per lane).</summary>
    public const int SsmPrefillMaxKeyDim = 256;

    private CudaModule? _ssmPrefillModule;
    private nint _ssmConvSplitPrefill, _ssmL2NormRows, _ssmConvHistoryUpdate, _ssmDeltaPrefill, _ssmGatedNormPrefill;

    /// <summary>True when ssm_prefill_f32.ptx loaded.</summary>
    public bool HasSsmPrefillKernels => _ssmDeltaPrefill != 0;

    // Optional module: its absence leaves prefill on the host reference instead of failing construction.
    private void LoadSsmPrefillKernels()
    {
        string path = Ptx("ssm_prefill_f32");
        if (!File.Exists(path))
            return;
        _ssmPrefillModule = LoadOwnedModule(path);
        _ssmConvSplitPrefill = _ssmPrefillModule.GetFunction("ssm_conv_split_prefill_f32");
        _ssmL2NormRows = _ssmPrefillModule.GetFunction("ssm_l2norm_rows_f32");
        _ssmConvHistoryUpdate = _ssmPrefillModule.GetFunction("ssm_conv_history_update_f32");
        _ssmDeltaPrefill = _ssmPrefillModule.GetFunction("ssm_delta_prefill_f32");
        _ssmGatedNormPrefill = _ssmPrefillModule.GetFunction("ssm_gated_norm_prefill_f32");
    }

    /// <summary>Conv + SiLU + split over <paramref name="seq"/> tokens, the per-head L2 norms of q and k, then the carry
    /// update (after the conv, which reads the old carry).</summary>
    public unsafe void LaunchSsmConvSplitPrefill(ulong q, ulong k, ulong v, ulong qkvMixed, ulong history, ulong convW,
        int convDim, int keyDim, int valueDim, int kernel, int seq, int kHeads, int skDim, float eps, float qScale, nint stream)
    {
        if (_ssmDeltaPrefill == 0) throw new InvalidOperationException("ssm_prefill_f32.ptx not present in the Ptx folder.");
        ulong qA = q, kA = k, vA = v, mA = qkvMixed, hA = history, wA = convW;
        uint convA = (uint)convDim, keyA = (uint)keyDim, valueA = (uint)valueDim, kernelA = (uint)kernel, seqA = (uint)seq;
        void** conv = stackalloc void*[11];
        conv[0] = &qA; conv[1] = &kA; conv[2] = &vA; conv[3] = &mA; conv[4] = &hA; conv[5] = &wA;
        conv[6] = &convA; conv[7] = &keyA; conv[8] = &valueA; conv[9] = &kernelA; conv[10] = &seqA;
        ulong total = (ulong)seq * (ulong)convDim;
        CudaDriverApi.cuLaunchKernel(_ssmConvSplitPrefill, (uint)((total + BlockSize - 1) / BlockSize), 1, 1, BlockSize, 1, 1,
            0, stream, (nint)conv, 0).ThrowOnError();
        LaunchSsmL2NormRows(q, skDim, seq * kHeads, eps, qScale, stream);
        LaunchSsmL2NormRows(k, skDim, seq * kHeads, eps, 1f, stream);
        void** update = stackalloc void*[5];
        update[0] = &hA; update[1] = &mA; update[2] = &convA; update[3] = &kernelA; update[4] = &seqA;
        CudaDriverApi.cuLaunchKernel(_ssmConvHistoryUpdate, (uint)((convDim + BlockSize - 1) / BlockSize), 1, 1, BlockSize, 1, 1,
            0, stream, (nint)update, 0).ThrowOnError();
    }

    /// <summary>The delta-rule scan over <paramref name="seq"/> tokens into <paramref name="output"/>, then the gated norm
    /// in place.</summary>
    public unsafe void LaunchSsmDeltaPrefill(ulong output, ulong state, ulong q, ulong k, ulong v, ulong z, ulong alphaRaw,
        ulong betaRaw, ulong dtBias, ulong ssmA, ulong normW, int seq, int hv, int sv, int sk, int repeat, float eps, nint stream)
    {
        if (_ssmDeltaPrefill == 0) throw new InvalidOperationException("ssm_prefill_f32.ptx not present in the Ptx folder.");
        if (sk > SsmPrefillMaxKeyDim) throw new NotSupportedException($"The CUDA delta-rule scan holds key heads up to {SsmPrefillMaxKeyDim} wide; got {sk}.");
        ulong oA = output, stA = state, qA = q, kA = k, vA = v, alA = alphaRaw, beA = betaRaw, dtA = dtBias, saA = ssmA;
        uint seqA = (uint)seq, hvA = (uint)hv, svA = (uint)sv, skA = (uint)sk, repA = (uint)repeat;
        void** scan = stackalloc void*[14];
        scan[0] = &oA; scan[1] = &stA; scan[2] = &qA; scan[3] = &kA; scan[4] = &vA; scan[5] = &alA; scan[6] = &beA;
        scan[7] = &dtA; scan[8] = &saA; scan[9] = &seqA; scan[10] = &hvA; scan[11] = &svA; scan[12] = &skA; scan[13] = &repA;
        uint gridX = (uint)((sv + SsmScanRowsPerBlock - 1) / SsmScanRowsPerBlock);
        CudaDriverApi.cuLaunchKernel(_ssmDeltaPrefill, gridX, (uint)hv, 1, 32 * SsmScanRowsPerBlock, 1, 1, 0, stream,
            (nint)scan, 0).ThrowOnError();
        ulong zA = z, nwA = normW;
        uint rowsA = (uint)(seq * hv);
        float epsA = eps;
        void** norm = stackalloc void*[6];
        norm[0] = &oA; norm[1] = &zA; norm[2] = &nwA; norm[3] = &rowsA; norm[4] = &svA; norm[5] = &epsA;
        CudaDriverApi.cuLaunchKernel(_ssmGatedNormPrefill, (rowsA + SsmRowsPerBlock - 1) / SsmRowsPerBlock, 1, 1, 32 * SsmRowsPerBlock,
            1, 1, 0, stream, (nint)norm, 0).ThrowOnError();
    }

    private unsafe void LaunchSsmL2NormRows(ulong x, int dim, int rows, float eps, float postScale, nint stream)
    {
        ulong xA = x;
        uint dimA = (uint)dim, rowsA = (uint)rows;
        float epsA = eps, scaleA = postScale;
        void** a = stackalloc void*[5];
        a[0] = &xA; a[1] = &dimA; a[2] = &rowsA; a[3] = &epsA; a[4] = &scaleA;
        CudaDriverApi.cuLaunchKernel(_ssmL2NormRows, (rowsA + SsmRowsPerBlock - 1) / SsmRowsPerBlock, 1, 1, 32 * SsmRowsPerBlock,
            1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }
}
