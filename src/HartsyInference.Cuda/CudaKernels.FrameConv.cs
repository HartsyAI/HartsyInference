namespace HartsyInference.Cuda;

// Frame-axis depthwise convolution on token-major data (Kernels/dit/frame_conv.cu), behind IBackend.DepthwiseConvFrames.
public sealed partial class CudaKernels
{
    private CudaModule? _frameConvModule;
    private nint _depthwiseConvFrames;

    /// <summary>True when frame_conv.ptx loaded.</summary>
    public bool HasFrameConvKernels => _depthwiseConvFrames != 0;

    // Optional module: its absence leaves DepthwiseConvFrames on the host reference instead of failing construction.
    private void LoadFrameConvKernels()
    {
        string path = Ptx("frame_conv");
        if (!CanLoad(path))
            return;
        _frameConvModule = LoadOwnedModule(path);
        _depthwiseConvFrames = _frameConvModule.GetFunction("depthwise_conv_frames_f32");
    }

    /// <summary>One thread per output element.</summary>
    public unsafe void LaunchDepthwiseConvFrames(ulong output, ulong input, ulong weight, int frames, int spatial, int channels, int taps, int pad,
        nint stream)
    {
        if (_depthwiseConvFrames == 0) throw new InvalidOperationException("frame_conv.ptx not present in the Ptx folder.");
        ulong oA = output, iA = input, wA = weight;
        uint fA = (uint)frames, sA = (uint)spatial, cA = (uint)channels, tA = (uint)taps, pA = (uint)pad;
        void** args = stackalloc void*[8];
        args[0] = &oA; args[1] = &iA; args[2] = &wA; args[3] = &fA; args[4] = &sA; args[5] = &cA; args[6] = &tA; args[7] = &pA;
        ulong total = (ulong)frames * (ulong)spatial * (ulong)channels;
        CudaDriverApi.cuLaunchKernel(_depthwiseConvFrames, (uint)((total + BlockSize - 1) / BlockSize), 1, 1, BlockSize, 1, 1, 0, stream,
            (nint)args, 0).ThrowOnError();
    }
}
