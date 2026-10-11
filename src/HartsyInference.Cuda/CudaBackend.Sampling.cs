using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// Device-side sampling for captured decode steps.
public sealed partial class CudaBackend
{
    /// <summary>Candidate slots of the two-stage top-k: values then ids, after the final k values and ids in the sampler buffer.</summary>
    private const int SampleCandidates = 2048;

    // Largest top-k each sampler buffer was allocated for: its value, id and candidate regions are laid out from that size.
    private readonly Dictionary<ulong, int> _rngMaxK = [];

    /// <inheritdoc />
    public bool DeviceSamplingSupported => _kernels is { HasSampleKernel: true };

    /// <inheritdoc />
    public int DeviceSamplingMaxTopK => CudaKernels.SampleMaxK;

    /// <inheritdoc />
    public unsafe ulong AllocDeviceRng(ulong seed, int maxTopK)
    {
        using OpScope _op = EnterOp();
        int k = Math.Clamp(maxTopK, 1, CudaKernels.SampleMaxK);
        ulong handle = CudaMemory.AllocatePersistent((nuint)(16 + 8 * k + 8 * SampleCandidates));
        ulong* init = stackalloc ulong[2] { seed == 0 ? 0x9E3779B97F4A7C15ul : seed, 0ul };
        CudaDriverApi.cuMemcpyHtoDAsync(handle, (nint)init, 16, _stream.Handle).ThrowOnError();
        CudaDriverApi.cuStreamSynchronize(_stream.Handle).ThrowOnError();
        lock (_rngMaxK) _rngMaxK[handle] = k;
        return handle;
    }

    /// <inheritdoc />
    public void FreeDeviceRng(ulong handle)
    {
        if (handle == 0) return;
        using OpScope _op = EnterOp();
        lock (_rngMaxK) _rngMaxK.Remove(handle);
        CudaMemory.Free(handle);
    }

    /// <inheritdoc />
    public void SampleTopKInto(ulong outputTokenId, Tensor logits, int topK, float temperature, float topP, float minP, ulong rngState)
    {
        using NvtxRange _nvtx = NvtxRange.Push("SampleTopK");
        if (outputTokenId == 0 || rngState == 0 || logits.DType != DType.F32 || topK < 1 || topK > CudaKernels.SampleMaxK)
            throw new NotSupportedException("SampleTopKInto requires F32 logits, a token-id buffer, an RNG buffer and 1 <= topK <= 64.");
        int allocatedK;
        lock (_rngMaxK) if (!_rngMaxK.TryGetValue(rngState, out allocatedK)) allocatedK = 0;
        if (topK > allocatedK)
            throw new ArgumentException($"topK {topK} exceeds the {allocatedK} the RNG buffer was allocated for.", nameof(topK));
        using OpScope _op = EnterOp();
        EnsureKernels();
        int n = (int)logits.Shape[logits.Shape.Rank - 1];
        ulong pIn = GpuTransferHelper.CopyToDevice(logits);
        try
        {
            ulong vals = rngState + 16;
            ulong idx = vals + (ulong)(4 * topK);
            // A wide row (a vocabulary) on one block leaves the other SMs idle: take the top k of each slice on its own block, then merge
            // the few hundred survivors on one. Narrow rows and tiny k keep the single pass.
            int slices = Math.Min(128, SampleCandidates / topK);
            if (_kernels!.HasTopKSlices && slices >= 4 && n >= 8192)
            {
                int width = (n + slices - 1) / slices;
                slices = (n + width - 1) / width;
                ulong candVals = idx + (ulong)(4 * topK);
                ulong candIdx = candVals + (ulong)(4 * SampleCandidates);
                _kernels.LaunchTopKSlices(candVals, candIdx, pIn, n, width, topK, slices, _stream.Handle);
                _kernels.LaunchTopKLastDim(vals, idx, candVals, 0, 1, slices * topK, topK, false, _stream.Handle);
                _kernels.LaunchSampleFromTopK(outputTokenId, vals, idx, candIdx, topK, temperature, topP, minP, rngState, _stream.Handle);
            }
            else
            {
                _kernels!.LaunchTopKLastDim(vals, idx, pIn, 0, 1, n, topK, false, _stream.Handle);
                _kernels.LaunchSampleFromTopK(outputTokenId, vals, idx, 0, topK, temperature, topP, minP, rngState, _stream.Handle);
            }
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pIn);   // a no-op for the cached activation a decode step passes
        }
    }
}
