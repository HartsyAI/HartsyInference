using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// Fused DiT norm/activation ops (SiTU-GLU, weighted RMSNorm + modulate) and the Gated-DeltaNet prefill. Each
// validates through its Core reference so the errors match the host default.
public sealed partial class VulkanBackend
{
    /// <summary>Invocations per row workgroup of the norm shaders.</summary>
    private const uint NormRowLocal = 256;

    /// <summary>Invocations per (state row, head) workgroup of the delta-rule scan; four columns each, so key heads up to 256.</summary>
    private const uint DeltaScanLocal = 64;

    /// <summary>Widest key head <c>ssm_delta_prefill</c> holds (<see cref="DeltaScanLocal"/> × its MAX_COLS of 4).</summary>
    private const int DeltaScanMaxKeyDim = 256;

    /// <inheritdoc/>
    public void SituGlu(Tensor output, Tensor input)
    {
        using OpScope _op = EnterOp();
        (int inner, long rows) = SituGluReference.Validate(output, input);
        long total = rows * inner;
        if (total == 0)
            return;
        VulkanBuffer inBuf = GetBuffer(input);
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(total * input.DType.SizeInBytes));
        try
        {
            VulkanKernel k = GetKernel("situ_glu" + DtypeSuffix(input.DType), storageBufferCount: 2, _default1DSpec);
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Authoring.PushConstants pc = new(pcBytes);
            pc.U32((uint)inner);
            pc.U32(checked((uint)total));
            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc.Written, GroupCount(total, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan SituGlu dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public void RmsNormModulate(Tensor output, Tensor input, Tensor weight, Tensor scale, Tensor? shift, float eps)
    {
        using OpScope _op = EnterOp();
        (int dim, long rows, long rowsPerBatch) = RmsNormModulateReference.Validate(output, input, weight, scale, shift);
        if (rows == 0)
            return;
        string suffix = input.DType == output.DType ? DtypeSuffix(input.DType) : DtypeSuffix(input.DType) + DtypeSuffix(output.DType);
        VulkanBuffer inBuf = GetBuffer(input);
        VulkanBuffer wBuf = GetBuffer(weight);
        VulkanBuffer scaleBuf = GetBuffer(scale);
        VulkanBuffer shiftBuf = shift is null ? scaleBuf : GetBuffer(shift);   // never read when hasShift is 0
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(output.ElementCount * output.DType.SizeInBytes));
        try
        {
            VulkanKernel k = GetKernel("rms_norm_modulate" + suffix, storageBufferCount: 5, RowSpec(NormRowLocal));
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Authoring.PushConstants pc = new(pcBytes);
            pc.U32((uint)dim);
            pc.U32(checked((uint)rows));
            pc.U32(checked((uint)rowsPerBatch));
            pc.F32(eps);
            pc.U32(shift is null ? 0u : 1u);
            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, wBuf.Handle, scaleBuf.Handle, shiftBuf.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc.Written, checked((uint)rows));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan RmsNormModulate dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public void SsmConvSplitPrefill(Tensor q, Tensor k, Tensor v, Tensor qkvMixed, Tensor history, Tensor convW,
        int kernel, int kHeads, int skDim, int vHeads, int svDim, float eps, float qScale)
    {
        using OpScope _op = EnterOp();
        int seq = GatedDeltaNetReference.ValidateConvSplit(q, k, v, qkvMixed, history, convW, kernel, kHeads, skDim, vHeads, svDim);
        int keyDim = kHeads * skDim, valueDim = vHeads * svDim, convDim = 2 * keyDim + valueDim;
        VulkanBuffer mixedBuf = GetBuffer(qkvMixed);
        VulkanBuffer histBuf = GetBuffer(history);
        VulkanBuffer tapsBuf = GetBuffer(convW);
        VulkanBuffer qBuf = _xfer.AllocateDevice((ulong)(q.ElementCount * sizeof(float)));
        VulkanBuffer kBuf = _xfer.AllocateDevice((ulong)(k.ElementCount * sizeof(float)));
        VulkanBuffer vBuf = _xfer.AllocateDevice((ulong)(v.ElementCount * sizeof(float)));
        try
        {
            VulkanKernel conv = GetKernel("ssm_conv_split_prefill", storageBufferCount: 6, _default1DSpec);
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Authoring.PushConstants pc = new(pcBytes);
            pc.U32((uint)convDim);
            pc.U32((uint)keyDim);
            pc.U32((uint)valueDim);
            pc.U32((uint)kernel);
            pc.U32((uint)seq);
            Span<ulong> convBufs = stackalloc ulong[] { qBuf.Handle, kBuf.Handle, vBuf.Handle, mixedBuf.Handle, histBuf.Handle, tapsBuf.Handle };
            Dispatch(conv, convBufs, pc.Written, GroupCount((long)seq * convDim, LocalX1D));

            DispatchL2NormRows(qBuf, skDim, seq * kHeads, eps, qScale);
            DispatchL2NormRows(kBuf, skDim, seq * kHeads, eps, 1f);

            VulkanKernel update = GetKernel("ssm_conv_history_update", storageBufferCount: 2, _default1DSpec);
            Span<byte> updateBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Authoring.PushConstants updatePc = new(updateBytes);
            updatePc.U32((uint)convDim);
            updatePc.U32((uint)kernel);
            updatePc.U32((uint)seq);
            Span<ulong> updateBufs = stackalloc ulong[] { histBuf.Handle, mixedBuf.Handle };
            Dispatch(update, updateBufs, updatePc.Written, GroupCount(convDim, LocalX1D));

            CacheOutput(q, qBuf);
            CacheOutput(k, kBuf);
            CacheOutput(v, vBuf);
            // The carry was advanced in place on its own buffer: re-cache it so the next reader sees the new rows.
            CacheOutput(history, histBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan SsmConvSplitPrefill dispatch failed", ex);
            qBuf.Dispose();
            kBuf.Dispose();
            vBuf.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public void SsmDeltaPrefill(Tensor output, Tensor state, Tensor q, Tensor k, Tensor v, Tensor z, Tensor alphaRaw,
        Tensor betaRaw, Tensor dtBias, Tensor ssmA, Tensor normW, int hv, int sv, int sk, int repeat, float eps)
    {
        using OpScope _op = EnterOp();
        int seq = GatedDeltaNetReference.ValidateDeltaScan(output, state, q, k, v, z, alphaRaw, betaRaw, dtBias, ssmA, normW, hv, sv, sk, repeat);
        if (sk > DeltaScanMaxKeyDim)
        {
            GatedDeltaNetReference.DeltaScan(output, state, q, k, v, z, alphaRaw, betaRaw, dtBias, ssmA, normW, hv, sv, sk, repeat, eps);
            return;
        }
        VulkanBuffer stateBuf = GetBuffer(state);
        VulkanBuffer qBuf = GetBuffer(q);
        VulkanBuffer kBuf = GetBuffer(k);
        VulkanBuffer vBuf = GetBuffer(v);
        VulkanBuffer zBuf = GetBuffer(z);
        VulkanBuffer alphaBuf = GetBuffer(alphaRaw);
        VulkanBuffer betaBuf = GetBuffer(betaRaw);
        VulkanBuffer biasBuf = GetBuffer(dtBias);
        VulkanBuffer decayBuf = GetBuffer(ssmA);
        VulkanBuffer normBuf = GetBuffer(normW);
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(output.ElementCount * sizeof(float)));
        try
        {
            VulkanKernel scan = GetKernel("ssm_delta_prefill", storageBufferCount: 9, RowSpec(DeltaScanLocal));
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Authoring.PushConstants pc = new(pcBytes);
            pc.U32((uint)seq);
            pc.U32((uint)hv);
            pc.U32((uint)sv);
            pc.U32((uint)sk);
            pc.U32((uint)repeat);
            Span<ulong> scanBufs = stackalloc ulong[]
            {
                outBuf.Handle, stateBuf.Handle, qBuf.Handle, kBuf.Handle, vBuf.Handle, alphaBuf.Handle, betaBuf.Handle,
                biasBuf.Handle, decayBuf.Handle,
            };
            Dispatch(scan, scanBufs, pc.Written, (uint)sv, (uint)hv);

            VulkanKernel norm = GetKernel("ssm_gated_norm_prefill", storageBufferCount: 3, RowSpec(NormRowLocal));
            Span<byte> normBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Authoring.PushConstants normPc = new(normBytes);
            normPc.U32(checked((uint)(seq * hv)));
            normPc.U32((uint)sv);
            normPc.F32(eps);
            Span<ulong> normBufs = stackalloc ulong[] { outBuf.Handle, zBuf.Handle, normBuf.Handle };
            Dispatch(norm, normBufs, normPc.Written, checked((uint)(seq * hv)));

            CacheOutput(output, outBuf);
            // The recurrent state was advanced in place on its own buffer: re-cache it.
            CacheOutput(state, stateBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan SsmDeltaPrefill dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    private void DispatchL2NormRows(VulkanBuffer buffer, int dim, int rows, float eps, float postScale)
    {
        VulkanKernel k = GetKernel("ssm_l2norm_rows", storageBufferCount: 1, RowSpec(NormRowLocal));
        Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
        Authoring.PushConstants pc = new(pcBytes);
        pc.U32((uint)dim);
        pc.U32((uint)rows);
        pc.F32(eps);
        pc.F32(postScale);
        Span<ulong> bufs = stackalloc ulong[] { buffer.Handle };
        Dispatch(k, bufs, pc.Written, (uint)rows);
    }

    /// <summary>Spec constants for a row-per-workgroup shader of <paramref name="local"/> invocations.</summary>
    private static SpecConstant[] RowSpec(uint local) =>
        [SpecConstant.UInt(0, local), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1)];
}
