namespace HartsyInference.Cuda;

// Fused DiT activation kernels (Kernels/dit/dit_norm_act.cu): SiTU-GLU and the gated residual that joins an F32 residual
// stream to an F16 branch.
public sealed partial class CudaKernels
{
    private CudaModule? _ditNormActModule;
    private nint _situGluF32, _situGluF16, _gatedResidualF32F16;

    /// <summary>True when dit_norm_act.ptx loaded.</summary>
    public bool HasDitNormActKernels => _situGluF32 != 0;

    /// <summary>True when the F32-stream / F16-branch gated residual loaded.</summary>
    public bool HasMixedStreamKernels => _gatedResidualF32F16 != 0;

    // Optional module: its absence leaves the ops on the host reference instead of failing construction.
    private void LoadDitNormActKernels()
    {
        string path = Ptx("dit_norm_act");
        if (!CanLoad(path))
            return;
        _ditNormActModule = LoadOwnedModule(path);
        _situGluF32 = _ditNormActModule.GetFunction("situ_glu_f32");
        _situGluF16 = _ditNormActModule.GetFunction("situ_glu_f16");
        _gatedResidualF32F16 = _ditNormActModule.GetFunction("gated_residual_f32_f16");
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
