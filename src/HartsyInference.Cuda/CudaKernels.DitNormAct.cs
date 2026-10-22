namespace HartsyInference.Cuda;

// Fused DiT norm/activation kernels (Kernels/dit/dit_norm_act.cu): weighted RMSNorm + adaLN modulate, SiTU-GLU, and the
// LayerNorm and gated residual that join an F32 residual stream to F16 branches.
public sealed partial class CudaKernels
{
    private CudaModule? _ditNormActModule;
    private nint _rmsNormModulateF32F32, _rmsNormModulateF32F16, _rmsNormModulateF16F16, _rmsNormModulateF16F32;
    private nint _situGluF32, _situGluF16;
    private nint _layerNormModulateF32F16, _gatedResidualF32F16;

    /// <summary>True when dit_norm_act.ptx loaded.</summary>
    public bool HasDitNormActKernels => _rmsNormModulateF32F32 != 0 && _situGluF32 != 0;

    /// <summary>True when the F32-stream / F16-branch kernels loaded.</summary>
    public bool HasMixedStreamKernels => _layerNormModulateF32F16 != 0 && _gatedResidualF32F16 != 0;

    // Optional module: its absence leaves the ops on the host reference instead of failing construction.
    private void LoadDitNormActKernels()
    {
        string path = Ptx("dit_norm_act");
        if (!CanLoad(path))
            return;
        _ditNormActModule = LoadOwnedModule(path);
        _rmsNormModulateF32F32 = _ditNormActModule.GetFunction("rms_norm_modulate_f32_f32");
        _rmsNormModulateF32F16 = _ditNormActModule.GetFunction("rms_norm_modulate_f32_f16");
        _rmsNormModulateF16F16 = _ditNormActModule.GetFunction("rms_norm_modulate_f16_f16");
        _rmsNormModulateF16F32 = _ditNormActModule.GetFunction("rms_norm_modulate_f16_f32");
        _situGluF32 = _ditNormActModule.GetFunction("situ_glu_f32");
        _situGluF16 = _ditNormActModule.GetFunction("situ_glu_f16");
        _layerNormModulateF32F16 = _ditNormActModule.GetFunction("layer_norm_modulate_f32_f16");
        _gatedResidualF32F16 = _ditNormActModule.GetFunction("gated_residual_f32_f16");
    }

    /// <summary>Launches the fused weighted RMSNorm + modulate: one 256-thread block per row; a null
    /// <paramref name="shift"/> (0) skips the shift.</summary>
    public unsafe void LaunchRmsNormModulate(ulong output, ulong input, ulong weight, ulong scale, ulong shift, int dim,
        int rows, int rowsPerMod, float eps, bool inputF16, bool outputF16, nint stream)
    {
        nint function = (inputF16, outputF16) switch
        {
            (false, false) => _rmsNormModulateF32F32,
            (false, true) => _rmsNormModulateF32F16,
            (true, true) => _rmsNormModulateF16F16,
            (true, false) => _rmsNormModulateF16F32,
        };
        if (function == 0) throw new InvalidOperationException("dit_norm_act.ptx not present in the Ptx folder.");
        ulong oA = output, iA = input, wA = weight, scA = scale, shA = shift;
        uint dimA = (uint)dim, rowsA = (uint)rows, perModA = (uint)rowsPerMod;
        float epsA = eps;
        void** a = stackalloc void*[9];
        a[0] = &oA; a[1] = &iA; a[2] = &wA; a[3] = &scA; a[4] = &shA; a[5] = &dimA; a[6] = &rowsA; a[7] = &perModA; a[8] = &epsA;
        CudaDriverApi.cuLaunchKernel(function, (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Launches the LayerNorm of an F32 row into F16, <c>(x − mean)·rsqrt(var + eps)·w·(1 + scale) + shift</c>: one
    /// 256-thread block per row; a null (0) <paramref name="weight"/>, <paramref name="scale"/> or <paramref name="shift"/> drops its term.</summary>
    public unsafe void LaunchLayerNormModulateF32ToF16(ulong output, ulong input, ulong weight, ulong scale, ulong shift, int dim,
        int rows, int rowsPerMod, float eps, nint stream)
    {
        if (_layerNormModulateF32F16 == 0) throw new InvalidOperationException("dit_norm_act.ptx lacks layer_norm_modulate_f32_f16.");
        ulong oA = output, iA = input, wA = weight, scA = scale, shA = shift;
        uint dimA = (uint)dim, rowsA = (uint)rows, perModA = (uint)rowsPerMod;
        float epsA = eps;
        void** a = stackalloc void*[9];
        a[0] = &oA; a[1] = &iA; a[2] = &wA; a[3] = &scA; a[4] = &shA; a[5] = &dimA; a[6] = &rowsA; a[7] = &perModA; a[8] = &epsA;
        CudaDriverApi.cuLaunchKernel(_layerNormModulateF32F16, (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Gated residual over the last dim with an F32 residual and output and an F16 value (gate stays F32).</summary>
    public void LaunchGatedResidualLastDimF32F16(ulong output, ulong residual, ulong value, ulong gate, int seqLen, int dim, long total,
        nint stream)
    {
        if (_gatedResidualF32F16 == 0) throw new InvalidOperationException("dit_norm_act.ptx lacks gated_residual_f32_f16.");
        LaunchGatedResidualImpl(_gatedResidualF32F16, output, residual, value, gate, seqLen, dim, total, stream);
    }

    /// <summary>Launches SiTU-GLU over <paramref name="rows"/> rows of <c>[u | g]</c>, each half <paramref name="inner"/> wide.</summary>
    public unsafe void LaunchSituGlu(ulong output, ulong input, int inner, long rows, bool f16, nint stream)
    {
        nint function = f16 ? _situGluF16 : _situGluF32;
        if (function == 0) throw new InvalidOperationException("dit_norm_act.ptx not present in the Ptx folder.");
        ulong oA = output, iA = input;
        uint innerA = (uint)inner;
        ulong total = (ulong)rows * (ulong)inner;
        void** a = stackalloc void*[4];
        a[0] = &oA; a[1] = &iA; a[2] = &innerA; a[3] = &total;
        uint grid = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(function, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }
}
