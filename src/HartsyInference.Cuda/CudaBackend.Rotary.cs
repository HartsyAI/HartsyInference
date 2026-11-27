using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// Rotary position embedding: the generic kernel for every geometry, and the fused norm-then-rotate kernel where a
// call's rotation is the LTX-2 split pattern it implements.
public sealed partial class CudaBackend
{
    /// <inheritdoc />
    public void Rotary(in RotaryCall call)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Rotary");
        RotaryGeometry g = RotaryContract.Validate(call);
        using OpScope _op = EnterOp();
        EnsureKernels();
        ulong pX = 0, pCos = 0, pSin = 0;
        try
        {
            pX = GpuTransferHelper.CopyToDevice(call.X);
            pCos = GpuTransferHelper.CopyToDevice(call.Cos);
            pSin = GpuTransferHelper.CopyToDevice(call.Sin);
            _kernels!.LaunchRotary(pX, pCos, pSin, g.DevicePosition != 0 ? g.DevicePosition : pCos, g, call.X.DType, _stream.Handle);
            // In place: clear stale callbacks before re-caching (pitfall #17).
            call.X._gpuSyncCallback = null;
            call.X._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(call.X, pX, GpuTransferHelper.ByteSize(call.X));
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pCos);
            GpuTransferHelper.FreeDevice(pSin);
        }
    }

    /// <inheritdoc />
    public void RmsNormRotary(Tensor output, Tensor input, Tensor normWeight, float eps, Tensor? cos, Tensor? sin, in RotarySpec rotation)
    {
        if (input.Shape.Rank != 2)
            throw new ArgumentException($"RmsNormRotary takes token-major rows [seq, heads·headDim]; got {input.Shape}.", nameof(input));
        bool headMajor = output.Shape.Rank == 4;
        int seq = (int)input.Shape[0];
        int heads = headMajor ? (int)output.Shape[1] : 0;
        int headDim = headMajor ? (int)output.Shape[3] : 0;
        bool rotate = cos is not null && sin is not null;
        // The fused kernel implements one geometry: rotate-half over every head with one angle per pair, per-head lanes
        // [seq, heads·headDim/2] (LTX-2 split). Anything else, or a head count the fused kernel cannot derive from a
        // token-major output, composes the three ops.
        bool fused = (!rotate || IsSplitRotation(rotation, seq, ref heads, ref headDim)) && (headMajor || heads > 0)
            && (input.DType == DType.F32 || input.DType == DType.F16) && output.DType == input.DType
            && (256L + (long)heads * headDim) * sizeof(float) <= 48 * 1024;
        if (!fused)
        {
            IBackend.ComposeRmsNormRotary(this, output, input, normWeight, eps, cos, sin, rotation);
            return;
        }
        using NvtxRange _nvtx = NvtxRange.Push(headMajor ? "RmsNormRotaryHeadMajor" : "RmsNormRotaryTokenMajor");
        bool f16 = input.DType == DType.F16;
        bool ropeF16 = cos is not null && cos.DType == DType.F16;
        using OpScope _op = EnterOp();
        EnsureKernels();
        ulong pOut = 0, pIn = 0, pW = 0, pCos = 0, pSin = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(normWeight);
            if (rotate)
            {
                pCos = GpuTransferHelper.CopyToDevice(cos!);
                pSin = GpuTransferHelper.CopyToDevice(sin!);
            }
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            if (headMajor) _kernels!.LaunchLtx2QkNormRopeHeadMajor(pOut, pIn, pW, pCos, pSin, seq, heads, headDim, eps, _stream.Handle, f16, ropeF16);
            else _kernels!.LaunchLtx2QkNormRopeTokenMajor(pOut, pIn, pW, pCos, pSin, seq, heads, headDim, eps, _stream.Handle, f16, ropeF16);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pW);
            if (pCos != 0) GpuTransferHelper.FreeDevice(pCos);
            if (pSin != 0) GpuTransferHelper.FreeDevice(pSin);
        }
    }

    /// <summary>True when <paramref name="rotation"/> is the split rotation the fused kernel runs: every head of the rows,
    /// rotate-half, per-head lane tables. On success the head count and head dim it implies are returned.</summary>
    private static bool IsSplitRotation(in RotarySpec rotation, int seq, ref int heads, ref int headDim)
    {
        if (rotation.DevicePosition != 0) return false;
        int h = (int)rotation.X[1];
        int pairs = (int)rotation.X[5];
        int d = 2 * pairs;
        RotarySpec expected = RotaryViews.Create(RotaryViews.Rows(seq, h, d), RotaryViews.PerHeadLanes(seq, h, pairs), RotaryPairing.RotateHalf, RopeTableLayout.InterleavedPrefix);
        if (rotation.X != expected.X || rotation.Cos != expected.Cos || rotation.Sin != expected.Sin) return false;
        if (heads != 0 && (heads != h || headDim != d)) return false;
        heads = h;
        headDim = d;
        return true;
    }
}
