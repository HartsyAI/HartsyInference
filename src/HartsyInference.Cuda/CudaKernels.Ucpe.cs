namespace HartsyInference.Cuda;

// UCPE token transform (Kernels/dit/ucpe.cu), behind IBackend.UcpeTransform.
public sealed partial class CudaKernels
{
    private CudaModule? _ucpeModule;
    private nint _ucpeTransform;

    /// <summary>True when ucpe.ptx loaded.</summary>
    public bool HasUcpeKernels => _ucpeTransform != 0;

    // Optional module: a missing PTX does not fail construction; UcpeTransform then throws NotSupportedException when it is called.
    private void LoadUcpeKernels()
    {
        string path = Ptx("ucpe");
        if (!CanLoad(path))
            return;
        _ucpeModule = LoadOwnedModule(path);
        _ucpeTransform = _ucpeModule.GetFunction("ucpe_transform_f32");
    }

    /// <summary>One thread per four channels of one token and head.</summary>
    public unsafe void LaunchUcpeTransform(ulong output, ulong input, ulong matrices, ulong cos, ulong sin, int tokens, int heads, int headDim,
        nint stream)
    {
        if (_ucpeTransform == 0) throw new InvalidOperationException("ucpe.ptx not present in the Ptx folder.");
        ulong oA = output, iA = input, mA = matrices, cA = cos, sA = sin;
        uint tA = (uint)tokens, hA = (uint)heads, dA = (uint)headDim;
        void** args = stackalloc void*[8];
        args[0] = &oA; args[1] = &iA; args[2] = &mA; args[3] = &cA; args[4] = &sA; args[5] = &tA; args[6] = &hA; args[7] = &dA;
        ulong total = (ulong)tokens * (ulong)heads * (ulong)(headDim / 4);
        CudaDriverApi.cuLaunchKernel(_ucpeTransform, (uint)((total + BlockSize - 1) / BlockSize), 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0)
            .ThrowOnError();
    }
}
