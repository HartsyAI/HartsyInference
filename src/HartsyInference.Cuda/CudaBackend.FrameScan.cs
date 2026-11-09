using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// IBackend.FrameDeltaScan on the device: every (head, frame) Gram and injection in two batched GEMMs, the sequential state
// recurrence in one kernel, and the token readout in a third batched GEMM, all addressing the token-major operands in
// place (row stride heads·dim), so nothing is transposed to head-major first.
public sealed partial class CudaBackend
{
    /// <inheritdoc cref="IBackend.FrameDeltaScan"/>
    public unsafe void FrameDeltaScan(Tensor output, Tensor query, Tensor key, Tensor? value, Tensor beta, Tensor decay, int frames,
        int spatial, int heads, int keyDim, int valueDim)
    {
        FrameDeltaScanReference.Validate(output, query, key, value, beta, decay, frames, spatial, heads, keyDim, valueDim);
        using NvtxRange _nvtx = NvtxRange.Push("FrameDeltaScan");
        using OpScope _op = EnterOp();
        EnsureKernels();
        if (!_kernels!.HasFrameScanKernels)
            throw new NotSupportedException("FrameDeltaScan needs frame_scan.ptx in the Ptx folder.");
        if (keyDim > CudaKernels.FrameScanMaxKeyDim)
            throw new NotSupportedException($"The CUDA frame scan holds key heads up to {CudaKernels.FrameScanMaxKeyDim} wide; got {keyDim}.");

        int count = heads * frames;
        long tokens = (long)frames * spatial;
        int keyRow = heads * keyDim;
        int valueRow = heads * valueDim;
        const int f32 = sizeof(float);
        ulong pQ = 0, pK = 0, pV = 0, pBeta = 0, pDecay = 0, pOut = 0, scaled = 0, gramT = 0, injection = 0, states = 0, reverse = 0, ones = 0,
            pointers = 0;
        bool cachedOutput = false;
        try
        {
            pQ = GpuTransferHelper.CopyToDevice(query);
            pK = GpuTransferHelper.CopyToDevice(key);
            pV = value is null ? 0 : GpuTransferHelper.CopyToDevice(value);
            pBeta = GpuTransferHelper.CopyToDevice(beta);
            pDecay = GpuTransferHelper.CopyToDevice(decay);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            scaled = GpuTransferHelper.AllocateDevice((nuint)(tokens * keyRow * f32));
            gramT = GpuTransferHelper.AllocateDevice((nuint)((long)count * keyDim * keyDim * f32));
            injection = GpuTransferHelper.AllocateDevice((nuint)((long)count * keyDim * valueDim * f32));
            states = GpuTransferHelper.AllocateDevice((nuint)((long)count * keyDim * valueDim * f32));
            reverse = GpuTransferHelper.AllocateDevice((nuint)((long)count * keyDim * valueDim * f32));
            if (value is null)
            {
                ones = GpuTransferHelper.AllocateDevice((nuint)(spatial * f32));
                float[] unit = new float[spatial];
                Array.Fill(unit, 1f);
                fixed (float* hostOnes = unit)
                    CudaMemory.CopyHostToDeviceAsync(ones, hostOnes, (nuint)(spatial * f32), _stream.Handle);
            }

            // Nine pointer arrays of `count` entries: Gram (A, B, C), injection (A, B, C), readout (A, B, C).
            ulong[] host = new ulong[9 * count];
            for (int h = 0; h < heads; h++)
            {
                for (int t = 0; t < frames; t++)
                {
                    int i = h * frames + t;
                    ulong keyOffset = (ulong)(((long)t * spatial * keyRow + (long)h * keyDim) * f32);
                    ulong valueOffset = (ulong)(((long)t * spatial * valueRow + (long)h * valueDim) * f32);
                    ulong gramOffset = (ulong)((long)i * keyDim * keyDim * f32);
                    ulong stateOffset = (ulong)((long)i * keyDim * valueDim * f32);
                    host[0 * count + i] = scaled + keyOffset;
                    host[1 * count + i] = pK + keyOffset;
                    host[2 * count + i] = gramT + gramOffset;
                    host[3 * count + i] = value is null ? ones : pV + valueOffset;
                    host[4 * count + i] = scaled + keyOffset;
                    host[5 * count + i] = injection + stateOffset;
                    host[6 * count + i] = states + stateOffset;
                    host[7 * count + i] = pQ + keyOffset;
                    host[8 * count + i] = pOut + valueOffset;
                }
            }

            pointers = GpuTransferHelper.AllocateDevice((nuint)(host.Length * sizeof(ulong)));
            fixed (ulong* hostPointers = host)
                CudaMemory.CopyHostToDeviceAsync(pointers, hostPointers, (nuint)(host.Length * sizeof(ulong)), _stream.Handle);
            ulong Slot(int slot) => pointers + (ulong)(slot * count * sizeof(ulong));

            using (NvtxRange.PushFine("FrameDeltaScan.scaleKeys"))
                _kernels.LaunchFrameScanScaleKeys(scaled, pK, pBeta, frames, spatial, heads, keyDim, _stream.Handle);

            int type = CublasApi.CUDA_R_32F;
            int compute = Compute32F(type);
            float alpha = 1f, zero = 0f;
            // Column-major views of the token-major operands: a frame's head slice [spatial, dim] (row stride heads·dim) is the
            // column-major [dim, spatial] matrix with leading dimension heads·dim.
            // gramT (row-major [Dk, Dk], gramT[j][r] = Σ_s kb[r]·k[j]); column-major that is kbᵀ·k = Kb_cm · K_cmᵀ.
            using (NvtxRange.PushFine("FrameDeltaScan.gram"))
                CublasApi.cublasGemmBatchedEx(_cublasHandle, CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_T, keyDim, keyDim, spatial, &alpha,
                    Slot(0), type, keyRow, Slot(1), type, keyRow, &zero, Slot(2), type, keyDim, count, compute,
                    CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
            // injection (row-major [Dk, Dv] = kbᵀ·v); column-major vᵀ·kb = V_cm · Kb_cmᵀ. A unit value is a ones row (lda 1).
            using (NvtxRange.PushFine("FrameDeltaScan.injection"))
                CublasApi.cublasGemmBatchedEx(_cublasHandle, CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_T, valueDim, keyDim, spatial, &alpha,
                    Slot(3), type, value is null ? 1 : valueRow, Slot(4), type, keyRow, &zero, Slot(5), type, valueDim, count, compute,
                    CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();

            using (NvtxRange.PushFine("FrameDeltaScan.recurrence"))
                _kernels.LaunchFrameScanRecurrence(states, reverse, gramT, injection, pDecay, frames, heads, keyDim, valueDim, _stream.Handle);

            // readout (row-major [S, Dv] = q·state); column-major stateᵀ·qᵀ = State_cm · Q_cm, written in place at row stride heads·Dv.
            using (NvtxRange.PushFine("FrameDeltaScan.readout"))
                CublasApi.cublasGemmBatchedEx(_cublasHandle, CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N, valueDim, spatial, keyDim, &alpha,
                    Slot(6), type, valueDim, Slot(7), type, keyRow, &zero, Slot(8), type, valueRow, count, compute,
                    CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pQ);
            GpuTransferHelper.FreeDevice(pK);
            if (pV != 0) GpuTransferHelper.FreeDevice(pV);
            GpuTransferHelper.FreeDevice(pBeta);
            GpuTransferHelper.FreeDevice(pDecay);
            foreach (ulong temp in new[] { scaled, gramT, injection, states, reverse, ones, pointers })
            {
                if (temp != 0) GpuTransferHelper.FreeDevice(temp);
            }

            if (!cachedOutput && pOut != 0) GpuTransferHelper.FreeDevice(pOut);
        }
    }
}
