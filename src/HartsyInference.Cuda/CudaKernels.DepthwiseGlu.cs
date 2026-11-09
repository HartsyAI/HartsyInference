namespace HartsyInference.Cuda;

// Channels-last depthwise convolution fused with a GLU (Kernels/conv/depthwise_glu.cu), behind IBackend.DepthwiseConvGlu.
public sealed partial class CudaKernels
{
    private CudaModule? _depthwiseGluModule;
    private nint _depthwiseGluF32, _depthwiseGluBf16;

    /// <summary>True when depthwise_glu.ptx loaded.</summary>
    public bool HasDepthwiseGluKernels => _depthwiseGluBf16 != 0;

    // Optional module: its absence leaves DepthwiseConvGlu on the host reference instead of failing construction.
    private void LoadDepthwiseGluKernels()
    {
        string path = Ptx("depthwise_glu");
        if (!CanLoad(path))
            return;
        _depthwiseGluModule = LoadOwnedModule(path);
        _depthwiseGluF32 = _depthwiseGluModule.GetFunction("depthwise_glu_f32");
        _depthwiseGluBf16 = _depthwiseGluModule.GetFunction("depthwise_glu_bf16");
    }

    /// <summary>One thread per output element: the value and gate taps of its channel, then <c>value · SiLU(gate)</c>.</summary>
    public unsafe void LaunchDepthwiseGlu(ulong output, ulong input, ulong taps, ulong bias, bool bf16, int frames, int height, int width,
        int hidden, int kernel, nint stream)
    {
        nint function = bf16 ? _depthwiseGluBf16 : _depthwiseGluF32;
        if (function == 0) throw new InvalidOperationException("depthwise_glu.ptx not present in the Ptx folder.");
        ulong oA = output, iA = input, tA = taps, bA = bias;
        uint fA = (uint)frames, hA = (uint)height, wA = (uint)width, cA = (uint)hidden, kA = (uint)kernel;
        void** args = stackalloc void*[9];
        args[0] = &oA; args[1] = &iA; args[2] = &tA; args[3] = &bA; args[4] = &fA; args[5] = &hA; args[6] = &wA; args[7] = &cA; args[8] = &kA;
        ulong total = (ulong)frames * (ulong)height * (ulong)width * (ulong)hidden;
        CudaDriverApi.cuLaunchKernel(function, (uint)((total + BlockSize - 1) / BlockSize), 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0)
            .ThrowOnError();
    }
}
