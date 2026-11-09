namespace HartsyInference.Cuda;

// Frame-wise delta-rule scan kernels (Kernels/dit/frame_scan.cu): the beta-scaled keys and the per-head state recurrence
// behind IBackend.FrameDeltaScan. The Gram, injection and readout GEMMs run through cuBLAS.
public sealed partial class CudaKernels
{
    /// <summary>State columns per recurrence block; matches FRAME_SCAN_COLS.</summary>
    private const int FrameScanCols = 32;

    /// <summary>Widest key head the recurrence holds (one thread per state row); matches FRAME_SCAN_MAX_KEY.</summary>
    public const int FrameScanMaxKeyDim = 128;

    private CudaModule? _frameScanModule;
    private nint _frameScanScaleKeys, _frameScanRecurrence, _frameScanRecurrenceColumn;

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
    /// injections.</summary>
    public unsafe void LaunchFrameScanRecurrence(ulong states, ulong gramT, ulong injection, ulong decay, int frames, int heads,
        int keyDim, int valueDim, nint stream)
    {
        if (_frameScanRecurrence == 0) throw new InvalidOperationException("frame_scan.ptx not present in the Ptx folder.");
        if (keyDim > FrameScanMaxKeyDim)
            throw new NotSupportedException($"The CUDA frame scan holds key heads up to {FrameScanMaxKeyDim} wide; got {keyDim}.");
        ulong stA = states, gA = gramT, uA = injection, dA = decay;
        uint fA = (uint)frames, kA = (uint)keyDim, vA = (uint)valueDim;
        void** args = stackalloc void*[7];
        args[0] = &stA; args[1] = &gA; args[2] = &uA; args[3] = &dA; args[4] = &fA; args[5] = &kA; args[6] = &vA;
        // A single value column (a denominator) takes the one-column build instead of padding to FrameScanCols.
        bool column = valueDim == 1;
        uint gridX = column ? 1u : (uint)((valueDim + FrameScanCols - 1) / FrameScanCols);
        CudaDriverApi.cuLaunchKernel(column ? _frameScanRecurrenceColumn : _frameScanRecurrence, gridX, (uint)heads, 1, FrameScanMaxKeyDim, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }
}
