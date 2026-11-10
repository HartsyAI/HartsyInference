using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Gpu;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// Sparse-attention index building on the device: the exact per-row top-k and the window/top-k merge (Kernels/latent/sparse_index.cu).
public sealed partial class CudaBackend
{
    /// <inheritdoc />
    public void SparseTopK(Tensor indices, Tensor scores, Tensor limits, int topK, int offset)
    {
        using NvtxRange _nvtx = NvtxRange.Push("SparseTopK");
        (int rows, int cols, int k) = SparseTopKReference.Validate(indices, scores, limits, topK);
        if (rows == 0 || k == 0) return;
        using OpScope _op = EnterOp();
        EnsureKernels();
        if (!_kernels!.HasSparseIndexKernels) throw new NotSupportedException("CUDA SparseTopK needs sparse_index.ptx in the Ptx folder.");
        nuint bytes = GpuTransferHelper.ByteSize(indices);
        ulong pS = 0, pL = 0, pOut = 0;
        bool cached = false;
        try
        {
            pS = GpuTransferHelper.CopyToDevice(scores);
            pL = GpuTransferHelper.CopyToDevice(limits);
            pOut = GpuTransferHelper.AllocateDevice(bytes);
            _kernels.LaunchSparseTopK(pS, pL, pOut, rows, cols, k, offset, _stream.Handle);
            GpuTransferHelper.CacheActivation(indices, pOut, bytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            FreeAll(pS, pL);
        }
    }

    /// <inheritdoc />
    public void MergeWindowTopK(Tensor rows, Tensor topk, Tensor positions, int window)
    {
        using NvtxRange _nvtx = NvtxRange.Push("MergeWindowTopK");
        (int tokens, int cols, int width) = WindowTopKMergeReference.Validate(rows, topk, positions, window);
        if (tokens == 0) return;
        using OpScope _op = EnterOp();
        EnsureKernels();
        if (!_kernels!.HasSparseIndexKernels) throw new NotSupportedException("CUDA MergeWindowTopK needs sparse_index.ptx in the Ptx folder.");
        nuint bytes = GpuTransferHelper.ByteSize(rows);
        ulong pT = 0, pP = 0, pOut = 0;
        bool cached = false;
        try
        {
            pT = cols == 0 ? 0 : GpuTransferHelper.CopyToDevice(topk);
            pP = GpuTransferHelper.CopyToDevice(positions);
            pOut = GpuTransferHelper.AllocateDevice(bytes);
            _kernels.LaunchMergeWindowTopK(pT, pP, pOut, (long)tokens * width, cols, window, width, _stream.Handle);
            GpuTransferHelper.CacheActivation(rows, pOut, bytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            FreeAll(pT, pP);
        }
    }
}
