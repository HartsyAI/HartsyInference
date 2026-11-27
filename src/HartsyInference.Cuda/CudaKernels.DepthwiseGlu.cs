namespace HartsyInference.Cuda;

// Channels-last depthwise convolution fused with a GLU (Kernels/conv/depthwise_glu.cu), behind IBackend.DepthwiseConvGlu.
public sealed partial class CudaKernels
{
    private CudaModule? _depthwiseGluModule;
    private nint _depthwiseGluF32, _depthwiseGluBf16, _depthwiseGluBf16Vec;

    /// <summary>Channels per thread of the vectorized kernel; matches DG_VEC.</summary>
    private const int DepthwiseGluVec = 8;

    /// <summary>Pixels per run of the vectorized kernel; matches DG_RUN.</summary>
    private const int DepthwiseGluRun = 4;

    /// <summary>True when depthwise_glu.ptx loaded.</summary>
    public bool HasDepthwiseGluKernels => _depthwiseGluBf16 != 0;

    // Optional module: a missing PTX does not fail construction; DepthwiseConvGlu then throws NotSupportedException when it is called.
    private void LoadDepthwiseGluKernels()
    {
        string path = Ptx("depthwise_glu");
        if (!CanLoad(path))
            return;
        _depthwiseGluModule = LoadOwnedModule(path);
        _depthwiseGluF32 = _depthwiseGluModule.GetFunction("depthwise_glu_f32");
        _depthwiseGluBf16 = _depthwiseGluModule.GetFunction("depthwise_glu_bf16");
        _depthwiseGluBf16Vec = _depthwiseGluModule.GetFunction("depthwise_glu_bf16_vec");
    }

    /// <summary>The depthwise conv and GLU. A BF16 3×3 with whole 8-channel groups and whole 4-pixel runs takes the vectorized
    /// kernel (one thread per 8 channels of an image row); anything else, one thread per output element.</summary>
    public unsafe void LaunchDepthwiseGlu(ulong output, ulong input, ulong taps, ulong bias, bool bf16, int frames, int height, int width,
        int hidden, int kernel, bool inputSilu, nint stream)
    {
        ulong oA = output, iA = input, tA = taps, bA = bias;
        uint fA = (uint)frames, hA = (uint)height, wA = (uint)width, cA = (uint)hidden, kA = (uint)kernel;
        int siluA = inputSilu ? 1 : 0;
        // The vectorized kernel puts frames·height on gridDim.y, which CUDA caps at 65,535 blocks; a taller input takes the general kernel.
        if (bf16 && kernel == 3 && hidden % DepthwiseGluVec == 0 && width % DepthwiseGluRun == 0 && _depthwiseGluBf16Vec != 0
            && (long)frames * height <= ChannelsLastLimits.MaxGridDimYZ)
        {
            void** vec = stackalloc void*[9];
            vec[0] = &oA; vec[1] = &iA; vec[2] = &tA; vec[3] = &bA; vec[4] = &fA; vec[5] = &hA; vec[6] = &wA; vec[7] = &cA; vec[8] = &siluA;
            const uint threads = 128;
            uint groups = (uint)(hidden / DepthwiseGluVec);
            CudaDriverApi.cuLaunchKernel(_depthwiseGluBf16Vec, (groups + threads - 1) / threads, (uint)(frames * height), 1, threads, 1, 1, 0,
                stream, (nint)vec, 0).ThrowOnError();
            return;
        }

        nint function = bf16 ? _depthwiseGluBf16 : _depthwiseGluF32;
        if (function == 0) throw new InvalidOperationException("depthwise_glu.ptx not present in the Ptx folder.");
        void** args = stackalloc void*[10];
        args[0] = &oA; args[1] = &iA; args[2] = &tA; args[3] = &bA; args[4] = &fA; args[5] = &hA; args[6] = &wA; args[7] = &cA; args[8] = &kA;
        args[9] = &siluA;
        ulong total = (ulong)frames * (ulong)height * (ulong)width * (ulong)hidden;
        CudaDriverApi.cuLaunchKernel(function, (uint)((total + BlockSize - 1) / BlockSize), 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0)
            .ThrowOnError();
    }
}
