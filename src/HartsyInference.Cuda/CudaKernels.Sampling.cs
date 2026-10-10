namespace HartsyInference.Cuda;

// Device-side token draw from sorted top-k candidates (Kernels/lm/lm_sample_topk.cu).
public sealed partial class CudaKernels
{
    private CudaModule? _sampleTopKModule;
    private nint _sampleFromTopK;

    /// <summary>Largest candidate count the sampler kernel handles.</summary>
    public const int SampleMaxK = 64;

    /// <summary>True when lm_sample_topk.ptx and the radix top-k kernel are both loaded.</summary>
    public bool HasSampleKernel => _sampleFromTopK != 0 && HasTopKKernel;

    private void LoadSamplingKernels()
    {
        string path = Ptx("lm_sample_topk");
        if (!File.Exists(path)) return;
        _sampleTopKModule = LoadOwnedModule(path);
        _sampleFromTopK = _sampleTopKModule.GetFunction("lm_sample_from_topk");
    }

    /// <summary>One draw from the sorted top-<paramref name="k"/> values and ids; <paramref name="rng"/> is the device {seed, counter} pair.</summary>
    public unsafe void LaunchSampleFromTopK(ulong outToken, ulong vals, ulong idx, int k, float temperature, float topP, float minP, ulong rng, nint stream)
    {
        if (_sampleFromTopK == 0) throw new InvalidOperationException("lm_sample_topk.ptx not present in the Ptx folder.");
        ulong oA = outToken, vA = vals, iA = idx, rA = rng;
        int kA = k;
        float tA = temperature, pA = topP, mA = minP;
        void** a = stackalloc void*[8];
        a[0] = &oA; a[1] = &vA; a[2] = &iA; a[3] = &kA; a[4] = &tA; a[5] = &pA; a[6] = &mA; a[7] = &rA;
        CudaDriverApi.cuLaunchKernel(_sampleFromTopK, 1, 1, 1, 1, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }
}
