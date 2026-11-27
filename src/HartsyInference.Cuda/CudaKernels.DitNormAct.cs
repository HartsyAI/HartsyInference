namespace HartsyInference.Cuda;

// The DiT gated residual that joins an F16 branch to an F32 stream (Kernels/dit/dit_norm_act.cu).
public sealed partial class CudaKernels
{
    private CudaModule? _ditNormActModule;
    private nint _gatedResidualF32F16;

    /// <summary>True when the F32-stream / F16-branch gated residual loaded.</summary>
    public bool HasMixedStreamKernels => _gatedResidualF32F16 != 0;

    // Optional module: its absence leaves the ops on the host reference instead of failing construction.
    private void LoadDitNormActKernels()
    {
        string path = Ptx("dit_norm_act");
        if (!CanLoad(path))
            return;
        _ditNormActModule = LoadOwnedModule(path);
        _gatedResidualF32F16 = _ditNormActModule.GetFunction("gated_residual_f32_f16");
    }

    /// <summary>Gated residual over the last dim with an F32 residual and output and an F16 value (gate stays F32).</summary>
    public void LaunchGatedResidualLastDimF32F16(ulong output, ulong residual, ulong value, ulong gate, int seqLen, int dim, long total,
        nint stream)
    {
        if (_gatedResidualF32F16 == 0) throw new InvalidOperationException("dit_norm_act.ptx lacks gated_residual_f32_f16.");
        LaunchGatedResidualImpl(_gatedResidualF32F16, output, residual, value, gate, seqLen, dim, total, stream);
    }

}
