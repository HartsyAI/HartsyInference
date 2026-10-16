using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;
using HartsyInference.Gpu;

namespace HartsyInference.Cuda;

// Fused DiT norm/activation ops and the Gated-DeltaNet prefill on the device. Each validates through its Core
// reference so the errors match the host default, and falls back to it when the module is absent.
public sealed partial class CudaBackend
{
    /// <inheritdoc />
    public void SituGlu(Tensor output, Tensor input)
    {
        using NvtxRange _nvtx = NvtxRange.Push("SituGlu");
        (int inner, long rows) = SituGluReference.Validate(output, input);
        using OpScope _op = EnterOp();
        EnsureKernels();
        if (!_kernels!.HasDitNormActKernels)
        {
            SituGluReference.Apply(output, input);
            return;
        }
        ulong pIn = 0, pOut = 0;
        bool cached = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint bytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(bytes);
            _kernels.LaunchSituGlu(pOut, pIn, inner, rows, input.DType == DType.F16, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, bytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <inheritdoc />
    public void RmsNormModulate(Tensor output, Tensor input, Tensor weight, Tensor scale, Tensor? shift, float eps)
    {
        using NvtxRange _nvtx = NvtxRange.Push("RmsNormModulate");
        (int dim, long rows, long rowsPerBatch) = RmsNormModulateReference.Validate(output, input, weight, scale, shift);
        using OpScope _op = EnterOp();
        EnsureKernels();
        if (!_kernels!.HasDitNormActKernels)
        {
            RmsNormModulateReference.Apply(output, input, weight, scale, shift, eps);
            return;
        }
        ulong pIn = 0, pOut = 0, pW = 0, pScale = 0, pShift = 0;
        bool cached = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(weight);
            pScale = GpuTransferHelper.CopyToDevice(scale);
            if (shift is not null) pShift = GpuTransferHelper.CopyToDevice(shift);
            nuint bytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(bytes);
            _kernels.LaunchRmsNormModulate(pOut, pIn, pW, pScale, pShift, dim, checked((int)rows), checked((int)rowsPerBatch), eps,
                input.DType == DType.F16, output.DType == DType.F16, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, bytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pW);
            GpuTransferHelper.FreeDevice(pScale);
            GpuTransferHelper.FreeDevice(pShift);
        }
    }

    /// <inheritdoc />
    public void SsmConvSplitPrefill(Tensor q, Tensor k, Tensor v, Tensor qkvMixed, Tensor history, Tensor convW,
        int kernel, int kHeads, int skDim, int vHeads, int svDim, float eps, float qScale)
    {
        using NvtxRange _nvtx = NvtxRange.Push("SsmConvSplitPrefill");
        int seq = GatedDeltaNetReference.ValidateConvSplit(q, k, v, qkvMixed, history, convW, kernel, kHeads, skDim, vHeads, svDim);
        using OpScope _op = EnterOp();
        EnsureKernels();
        if (!_kernels!.HasSsmPrefillKernels)
        {
            GatedDeltaNetReference.ConvSplit(q, k, v, qkvMixed, history, convW, kernel, kHeads, skDim, vHeads, svDim, eps, qScale);
            return;
        }
        int keyDim = kHeads * skDim, valueDim = vHeads * svDim;
        ulong pQ = 0, pK = 0, pV = 0, pM = 0, pH = 0, pW = 0;
        bool cached = false;
        try
        {
            pM = GpuTransferHelper.CopyToDevice(qkvMixed);
            pH = GpuTransferHelper.CopyToDevice(history);
            pW = GpuTransferHelper.CopyToDevice(convW);
            pQ = GpuTransferHelper.AllocateDevice(GpuTransferHelper.ByteSize(q));
            pK = GpuTransferHelper.AllocateDevice(GpuTransferHelper.ByteSize(k));
            pV = GpuTransferHelper.AllocateDevice(GpuTransferHelper.ByteSize(v));
            _kernels.LaunchSsmConvSplitPrefill(pQ, pK, pV, pM, pH, pW, 2 * keyDim + valueDim, keyDim, valueDim, kernel, seq,
                kHeads, skDim, eps, qScale, _stream.Handle);
            GpuTransferHelper.CacheActivation(q, pQ, GpuTransferHelper.ByteSize(q));
            GpuTransferHelper.CacheActivation(k, pK, GpuTransferHelper.ByteSize(k));
            GpuTransferHelper.CacheActivation(v, pV, GpuTransferHelper.ByteSize(v));
            // The carry was advanced in place: keep it resident for the next chunk or decode step.
            GpuTransferHelper.CacheActivation(history, pH, GpuTransferHelper.ByteSize(history));
            cached = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pM);
            GpuTransferHelper.FreeDevice(pW);
            if (!cached)
            {
                GpuTransferHelper.FreeDevice(pQ);
                GpuTransferHelper.FreeDevice(pK);
                GpuTransferHelper.FreeDevice(pV);
            }
        }
    }

    /// <inheritdoc />
    public void SsmDeltaPrefill(Tensor output, Tensor state, Tensor q, Tensor k, Tensor v, Tensor z, Tensor alphaRaw,
        Tensor betaRaw, Tensor dtBias, Tensor ssmA, Tensor normW, int hv, int sv, int sk, int repeat, float eps)
    {
        using NvtxRange _nvtx = NvtxRange.Push("SsmDeltaPrefill");
        int seq = GatedDeltaNetReference.ValidateDeltaScan(output, state, q, k, v, z, alphaRaw, betaRaw, dtBias, ssmA, normW, hv, sv, sk, repeat);
        using OpScope _op = EnterOp();
        EnsureKernels();
        if (!_kernels!.HasSsmPrefillKernels || sk > CudaKernels.SsmPrefillMaxKeyDim)
        {
            GatedDeltaNetReference.DeltaScan(output, state, q, k, v, z, alphaRaw, betaRaw, dtBias, ssmA, normW, hv, sv, sk, repeat, eps);
            return;
        }
        ulong pO = 0, pSt = 0, pQ = 0, pK = 0, pV = 0, pZ = 0, pAl = 0, pBe = 0, pDt = 0, pSa = 0, pNw = 0;
        bool cached = false;
        try
        {
            pSt = GpuTransferHelper.CopyToDevice(state);
            pQ = GpuTransferHelper.CopyToDevice(q);
            pK = GpuTransferHelper.CopyToDevice(k);
            pV = GpuTransferHelper.CopyToDevice(v);
            pZ = GpuTransferHelper.CopyToDevice(z);
            pAl = GpuTransferHelper.CopyToDevice(alphaRaw);
            pBe = GpuTransferHelper.CopyToDevice(betaRaw);
            pDt = GpuTransferHelper.CopyToDevice(dtBias);
            pSa = GpuTransferHelper.CopyToDevice(ssmA);
            pNw = GpuTransferHelper.CopyToDevice(normW);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pO = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels.LaunchSsmDeltaPrefill(pO, pSt, pQ, pK, pV, pZ, pAl, pBe, pDt, pSa, pNw, seq, hv, sv, sk, repeat, eps, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pO, outBytes);
            // The recurrent state was advanced in place: keep it resident.
            GpuTransferHelper.CacheActivation(state, pSt, GpuTransferHelper.ByteSize(state));
            cached = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pQ); GpuTransferHelper.FreeDevice(pK); GpuTransferHelper.FreeDevice(pV);
            GpuTransferHelper.FreeDevice(pZ); GpuTransferHelper.FreeDevice(pAl); GpuTransferHelper.FreeDevice(pBe);
            GpuTransferHelper.FreeDevice(pDt); GpuTransferHelper.FreeDevice(pSa); GpuTransferHelper.FreeDevice(pNw);
            if (!cached) GpuTransferHelper.FreeDevice(pO);
        }
    }
}
