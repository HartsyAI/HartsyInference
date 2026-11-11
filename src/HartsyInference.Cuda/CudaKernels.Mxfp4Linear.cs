namespace HartsyInference.Cuda;

// MXFP4-packed linear (Kernels/lm/mxfp4_linear_f32.cu): y = x · Wᵀ with the weight kept packed, one block per output row.
public sealed partial class CudaKernels
{
    private const uint Mxfp4LinearThreads = 256;

    private CudaModule? _mxfp4Module;
    private nint _mxfp4Linear;

    /// <summary>True when mxfp4_linear_f32.ptx loaded.</summary>
    public bool HasMxfp4LinearKernel => _mxfp4Linear != 0;

    // Optional module: its absence leaves the packed linear on the host reference instead of failing construction.
    private void LoadMxfp4LinearKernel()
    {
        string path = Ptx("mxfp4_linear_f32");
        if (!CanLoad(path))
            return;
        _mxfp4Module = LoadOwnedModule(path);
        _mxfp4Linear = _mxfp4Module.GetFunction("mxfp4_linear_f32");
    }

    /// <summary>Launches the packed linear over <paramref name="n"/> input rows into <paramref name="output"/>.</summary>
    public unsafe void LaunchMxfp4Linear(ulong output, ulong input, ulong packed, ulong scales, int n, int rows, int cols, nint stream)
    {
        if (_mxfp4Linear == 0) throw new InvalidOperationException("mxfp4_linear_f32.ptx not present in the Ptx folder.");
        ulong oA = output, iA = input, pA = packed, sA = scales;
        uint nA = (uint)n, rA = (uint)rows, cA = (uint)cols;
        void** args = stackalloc void*[7];
        args[0] = &oA; args[1] = &iA; args[2] = &pA; args[3] = &sA; args[4] = &nA; args[5] = &rA; args[6] = &cA;
        CudaDriverApi.cuLaunchKernel(_mxfp4Linear, (uint)rows, 1, 1, Mxfp4LinearThreads, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }
}
