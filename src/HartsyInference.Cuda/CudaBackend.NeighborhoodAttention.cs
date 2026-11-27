using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// 3D neighborhood attention on the device.
public sealed partial class CudaBackend
{
    /// <summary>Query-tiled neighborhood attention kernel; <c>numerics.neighborhoodAttentionTiled=false</c> falls back to the per-query one.
    /// Changes numerics: the tiled path is an online softmax over the tile's union window, not one dense pass.</summary>
    private static bool UseNeighborhoodAttentionTiled => EngineKnobs.NeighborhoodAttentionTiled.Value;

    /// <inheritdoc />
    /// <remarks>Prefers the query-tiled kernel, which stages a whole tile's shared window instead of re-reading one window per
    /// query; falls back to the per-query kernel on shapes it does not cover, and to the host reference when the module is not
    /// loaded or the per-query kernel's scores would not fit 48 KB of shared memory.</remarks>
    public void NeighborhoodAttention(Tensor output, Tensor q, Tensor k, Tensor v, in NeighborhoodWindow window, float scale)
    {
        using NvtxRange _nvtx = NvtxRange.Push("NeighborhoodAttention");
        NeighborhoodGeometry g = NeighborhoodAttentionContract.Validate(output, q, k, v, window, scale);
        EnsureKernels();
        (int tileH, int tileW) = _kernels is not null && UseNeighborhoodAttentionTiled ? _kernels.NeighborhoodAttentionTile(g) : (0, 0);
        if (_kernels is null || !_kernels.HasNeighborhoodAttentionKernels
            || (tileH == 0 && CudaKernels.NeighborhoodAttentionSharedBytes(g) > 48 * 1024))
        {
            NeighborhoodAttentionReference.Apply(output, q, k, v, window, scale);
            return;
        }

        using OpScope _op = EnterOp();
        ulong pOut = 0, pQ = 0, pK = 0, pV = 0;
        bool cached = false;
        try
        {
            pQ = GpuTransferHelper.CopyToDevice(q);
            pK = GpuTransferHelper.CopyToDevice(k);
            pV = GpuTransferHelper.CopyToDevice(v);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            if (tileH > 0) _kernels.LaunchNeighborhoodAttentionTiled(pOut, pQ, pK, pV, g, scale, tileH, tileW, _stream.Handle);
            else _kernels.LaunchNeighborhoodAttention(pOut, pQ, pK, pV, g, scale, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pQ);
            GpuTransferHelper.FreeDevice(pK);
            GpuTransferHelper.FreeDevice(pV);
        }
    }
}
