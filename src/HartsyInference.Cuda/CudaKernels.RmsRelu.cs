namespace HartsyInference.Cuda;

// RMSNorm + ReLU + scale in one pass (Kernels/dit/rms_relu.cu), behind IBackend.RmsNormRelu.
public sealed partial class CudaKernels
{
    private const uint RmsReluRowsPerBlock = 8;
    private CudaModule? _rmsReluModule;
    private nint _rmsNormRelu;

    /// <summary>True when rms_relu.ptx loaded.</summary>
    public bool HasRmsReluKernels => _rmsNormRelu != 0;

    // Optional module: its absence leaves RmsNormRelu on the composed default.
    private void LoadRmsReluKernels()
    {
        string path = Ptx("rms_relu");
        if (!CanLoad(path))
            return;
        _rmsReluModule = LoadOwnedModule(path);
        _rmsNormRelu = _rmsReluModule.GetFunction("rms_norm_relu_f32");
    }

    /// <summary>One warp per row.</summary>
    public unsafe void LaunchRmsNormRelu(ulong output, ulong input, ulong weight, int rows, int width, float eps, float scale, nint stream)
    {
        if (_rmsNormRelu == 0) throw new InvalidOperationException("rms_relu.ptx not present in the Ptx folder.");
        ulong oA = output, iA = input, wA = weight;
        uint rA = (uint)rows, dA = (uint)width;
        float eA = eps, sA = scale;
        void** args = stackalloc void*[7];
        args[0] = &oA; args[1] = &iA; args[2] = &wA; args[3] = &rA; args[4] = &dA; args[5] = &eA; args[6] = &sA;
        CudaDriverApi.cuLaunchKernel(_rmsNormRelu, (rA + RmsReluRowsPerBlock - 1) / RmsReluRowsPerBlock, 1, 1, 32 * RmsReluRowsPerBlock, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }
}
