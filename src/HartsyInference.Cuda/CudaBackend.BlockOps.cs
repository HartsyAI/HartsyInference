using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Backends.Gpu;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// Walsh-Hadamard rotation, the block-scaled FP8 linear and the grouped linear, shared with the DSA indexer and ERNIE-5.5 weights.
public sealed partial class CudaBackend
{
    /// <inheritdoc />
    public void Hadamard(Tensor output, Tensor input, float scale)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Hadamard");
        int width = HadamardReference.Validate(output, input);
        using OpScope _op = EnterOp();
        EnsureKernels();
        if (!_kernels!.HasHadamardKernels) throw new NotSupportedException("CUDA Hadamard needs hadamard.ptx in the Ptx folder.");
        long count = input.ElementCount;
        if (count == 0) return;
        if (ReferenceEquals(input, output))
        {
            ulong pX = 0;
            try
            {
                pX = GpuTransferHelper.CopyToDevice(input);
                LaunchHadamard(pX, pX, count, width, scale);
                RecacheInPlace(input, pX);
            }
            finally
            {
                GpuTransferHelper.FreeDevice(pX);
            }
            return;
        }
        nuint bytes = GpuTransferHelper.ByteSize(output);
        ulong pIn = 0, pOut = 0;
        bool cached = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pOut = GpuTransferHelper.AllocateDevice(bytes);
            LaunchHadamard(pIn, pOut, count, width, scale);
            GpuTransferHelper.CacheActivation(output, pOut, bytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            FreeAll(pIn);
        }
    }

    /// <inheritdoc />
    public void Fp8BlockLinear(Tensor output, Tensor input, Tensor weight, Tensor weightScaleInv, Tensor? bias,
        int blockRows = 128, int blockCols = 128)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Fp8BlockLinear");
        (int m, int n, int k) = Fp8BlockLinearReference.Validate(output, input, weight, weightScaleInv, bias, blockRows, blockCols);
        if (m == 0 || n == 0) return;
        using OpScope _op = EnterOp();
        EnsureKernels();
        if (!_kernels!.HasFp8BlockLinearKernels) throw new NotSupportedException("CUDA Fp8BlockLinear needs fp8_block_linear.ptx in the Ptx folder.");
        nuint bytes = GpuTransferHelper.ByteSize(output);
        ulong pX = 0, pW = 0, pS = 0, pB = 0, pY = 0;
        bool cached = false;
        try
        {
            pX = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(weight);
            pS = GpuTransferHelper.CopyToDevice(weightScaleInv);
            pB = bias is null ? 0 : GpuTransferHelper.CopyToDevice(bias);
            pY = GpuTransferHelper.AllocateDevice(bytes);
            _kernels.LaunchFp8BlockLinear(pX, pW, pS, pB, pY, m, n, k, blockRows, blockCols, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pY, bytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pY);
            FreeAll(pX, pW, pS, pB);
        }
    }

    /// <inheritdoc />
    public void BlockDiagonalLinear(Tensor output, Tensor input, Tensor weight, Tensor? weightScaleInv, int groups,
        int blockRows = 128, int blockCols = 128)
    {
        using NvtxRange _nvtx = NvtxRange.Push("BlockDiagonalLinear");
        (int m, int r, int d) = BlockDiagonalLinearReference.Validate(output, input, weight, weightScaleInv, groups, blockRows, blockCols);
        if (m == 0) return;
        using OpScope _op = EnterOp();
        EnsureKernels();
        if (!_kernels!.HasBlockDiagonalLinearKernels) throw new NotSupportedException("CUDA BlockDiagonalLinear needs fp8_block_linear.ptx with its block-diagonal entries.");
        nuint bytes = GpuTransferHelper.ByteSize(output);
        ulong pX = 0, pW = 0, pS = 0, pY = 0;
        bool cached = false;
        try
        {
            pX = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(weight);
            pS = weightScaleInv is null ? 0 : GpuTransferHelper.CopyToDevice(weightScaleInv);
            pY = GpuTransferHelper.AllocateDevice(bytes);
            _kernels.LaunchBlockDiagonalLinear(pX, pW, pS, pY, m, groups * r, d, groups * d, r, blockRows, blockCols, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pY, bytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pY);
            FreeAll(pX, pW, pS);
        }
    }

    // Width-1 rows are a plain scale; wider rows take log2(width) butterfly stages, the last one carrying the scale.
    private void LaunchHadamard(ulong src, ulong dst, long count, int width, float scale)
    {
        nint stream = _stream.Handle;
        if (width == 1)
        {
            _kernels!.LaunchHadamardScale(src, dst, count, scale, stream);
            return;
        }
        long pairs = count / 2;
        ulong from = src;
        for (int h = 1; h < width; h <<= 1)
        {
            float stageScale = h << 1 == width ? scale : 1f;
            _kernels!.LaunchHadamardStage(from, dst, pairs, width, h, stageScale, stream);
            from = dst;
        }
    }
}
