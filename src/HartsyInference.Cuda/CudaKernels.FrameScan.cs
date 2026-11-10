namespace HartsyInference.Cuda;

// Frame-wise delta-rule scan kernels (Kernels/dit/frame_scan.cu): the beta-scaled keys and the per-head state recurrence
// behind IBackend.FrameDeltaScan. The Gram, injection and readout GEMMs run through cuBLAS.
public sealed partial class CudaKernels
{
    /// <summary>State columns per recurrence block; matches FRAME_SCAN_COLS.</summary>
    private const int FrameScanCols = 16;

    /// <summary>Widest key head the recurrence holds (one thread per state row); matches FRAME_SCAN_MAX_KEY.</summary>
    public const int FrameScanMaxKeyDim = 128;

    private CudaModule? _frameScanModule;
    private nint _frameScanScaleKeys, _frameScanRecurrence, _frameScanRecurrenceColumn, _frameScanSum;

    /// <summary>True when frame_scan.ptx loaded.</summary>
    public bool HasFrameScanKernels => _frameScanRecurrence != 0;

    // Optional module: its absence leaves FrameDeltaScan on the host reference instead of failing construction.
    private void LoadFrameScanKernels()
    {
        string path = Ptx("frame_scan");
        if (!CanLoad(path))
            return;
        _frameScanModule = LoadOwnedModule(path);
        _frameScanScaleKeys = _frameScanModule.GetFunction("frame_scan_scale_keys_f32");
        _frameScanRecurrence = _frameScanModule.GetFunction("frame_scan_recurrence_f32");
        _frameScanRecurrenceColumn = _frameScanModule.GetFunction("frame_scan_recurrence_col_f32");
        _frameScanSum = _frameScanModule.GetFunction("frame_scan_sum_f32");
    }

    /// <summary><c>scaled[n, h, :] = beta[h, t, s] · key[n, h, :]</c> over every token.</summary>
    public unsafe void LaunchFrameScanScaleKeys(ulong scaled, ulong key, ulong beta, int frames, int spatial, int heads, int keyDim,
        nint stream)
    {
        if (_frameScanScaleKeys == 0) throw new InvalidOperationException("frame_scan.ptx not present in the Ptx folder.");
        ulong oA = scaled, kA = key, bA = beta;
        uint fA = (uint)frames, sA = (uint)spatial, hA = (uint)heads, dA = (uint)keyDim;
        void** args = stackalloc void*[7];
        args[0] = &oA; args[1] = &kA; args[2] = &bA; args[3] = &fA; args[4] = &sA; args[5] = &hA; args[6] = &dA;
        ulong total = (ulong)frames * (ulong)spatial * (ulong)heads * (ulong)keyDim;
        CudaDriverApi.cuLaunchKernel(_frameScanScaleKeys, (uint)((total + BlockSize - 1) / BlockSize), 1, 1, BlockSize, 1, 1, 0, stream,
            (nint)args, 0).ThrowOnError();
    }

    /// <summary>The forward-inclusive plus reverse-exclusive state of every head and frame, from the transposed Grams and the
    /// injections: both scans run at once (grid z = direction), the reverse into <paramref name="reverseScratch"/>, which is then
    /// added into <paramref name="states"/>. The forward scan starts from <paramref name="initial"/>
    /// (<c>[heads, keyDim, valueDim]</c>) when it is non-zero. The reverse state of the last frame is zero, so the last
    /// frame's slice of <paramref name="states"/> is the forward state after the scan.</summary>
    public unsafe void LaunchFrameScanRecurrence(ulong states, ulong reverseScratch, ulong gramT, ulong injection, ulong decay,
        ulong initial, int frames, int heads, int keyDim, int valueDim, nint stream)
    {
        if (_frameScanRecurrence == 0) throw new InvalidOperationException("frame_scan.ptx not present in the Ptx folder.");
        if (keyDim > FrameScanMaxKeyDim)
            throw new NotSupportedException($"The CUDA frame scan holds key heads up to {FrameScanMaxKeyDim} wide; got {keyDim}.");
        ulong stA = states, rvA = reverseScratch, gA = gramT, uA = injection, dA = decay, iA = initial;
        uint fA = (uint)frames, kA = (uint)keyDim, vA = (uint)valueDim;
        void** args = stackalloc void*[9];
        args[0] = &stA; args[1] = &rvA; args[2] = &gA; args[3] = &uA; args[4] = &dA; args[5] = &iA; args[6] = &fA; args[7] = &kA;
        args[8] = &vA;
        // A single value column (a denominator) takes the one-column build instead of padding to FrameScanCols.
        bool column = valueDim == 1;
        uint gridX = column ? 1u : (uint)((valueDim + FrameScanCols - 1) / FrameScanCols);
        CudaDriverApi.cuLaunchKernel(column ? _frameScanRecurrenceColumn : _frameScanRecurrence, gridX, (uint)heads, 2, FrameScanMaxKeyDim, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();

        ulong count = (ulong)heads * (ulong)frames * (ulong)keyDim * (ulong)valueDim;
        void** sum = stackalloc void*[3];
        sum[0] = &stA; sum[1] = &rvA; sum[2] = &count;
        CudaDriverApi.cuLaunchKernel(_frameScanSum, (uint)((count + BlockSize - 1) / BlockSize), 1, 1, BlockSize, 1, 1, 0, stream, (nint)sum, 0)
            .ThrowOnError();
    }
}
