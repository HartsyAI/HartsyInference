using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

/// <summary>Query-tiled flash attention on <c>VK_NV_cooperative_matrix2</c>, addressed by strides so head-major and
/// token-major Q/K/V run without a permute and grouped-query K/V are read without being repeated.</summary>
public sealed partial class VulkanBackend
{
    private const int FlashCm2QueryTile = 64;
    private const int FlashCm2KeyTile = 64;

    /// <inheritdoc/>
    public bool SupportsTokenMajorAttention => FlashCm2Available;

    /// <inheritdoc/>
    public bool SupportsTokenMajorGqaAttention => FlashCm2Available;

    private bool FlashCm2Available => EnableCoopMat2 && Vk.HasCooperativeMatrix2;

    /// <inheritdoc/>
    public void ScaledDotProductAttentionTokenMajor(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask,
        int heads, int headDim, float scale, bool allowF16 = false) =>
        ScaledDotProductAttentionTokenMajor(output, query, key, value, mask, heads, heads, headDim, scale, allowF16);

    /// <inheritdoc/>
    public void ScaledDotProductAttentionTokenMajor(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask,
        int heads, int kvHeads, int headDim, float scale, bool allowF16 = false)
    {
        using OpScope _op = EnterOp();
        long qWidth = (long)heads * headDim;
        long kvWidth = (long)kvHeads * headDim;
        if (qWidth <= 0 || kvWidth <= 0 || query.ElementCount % qWidth != 0 || key.ElementCount % kvWidth != 0
            || value.ElementCount != key.ElementCount || output.ElementCount != query.ElementCount)
        {
            throw new ArgumentException(
                $"Token-major attention needs Q/output as [S, {heads}*{headDim}] and K/V as [S, {kvHeads}*{headDim}]; got "
                + $"Q {query.Shape}, K {key.Shape}, V {value.Shape}, out {output.Shape}.");
        }
        int sq = (int)(query.ElementCount / qWidth);
        int skv = (int)(key.ElementCount / kvWidth);
        using Tensor? expandedMask = mask is not null && mask.DType == DType.F32 && mask.ElementCount == skv && sq > 1
            ? ExpandKeyOnlyMask(mask, sq, skv) : null;
        if (!CanUseFlashCm2(headDim, heads, kvHeads, sq, skv, expandedMask ?? mask))
        {
            AttendTokenMajorViaHeadMajor(output, query, key, value, mask, heads, kvHeads, headDim, sq, skv, scale, allowF16);
            return;
        }
        mask = expandedMask ?? mask;
        uint qRow = (uint)qWidth;
        uint kvRow = (uint)kvWidth;
        DispatchFlashCm2(output, query, key, value, mask, scale, batch: 1, heads, kvHeads, sq, skv, headDim,
            new AttnStrides(qRow, (uint)headDim, 0), new AttnStrides(kvRow, (uint)headDim, 0),
            new AttnStrides(kvRow, (uint)headDim, 0), new AttnStrides(qRow, (uint)headDim, 0));
    }

    /// <summary>Shapes the cooperative-matrix-2 kernel does not serve: permute to [1,H,S,D], attend, permute back.</summary>
    private void AttendTokenMajorViaHeadMajor(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask,
        int heads, int kvHeads, int headDim, int sq, int skv, float scale, bool allowF16)
    {
        TensorShape qShape = new TensorShape(1, heads, sq, headDim);
        TensorShape kvShape = new TensorShape(1, kvHeads, skv, headDim);
        using Tensor qMh = new Tensor(qShape, query.DType);
        using Tensor kMh = new Tensor(kvShape, key.DType);
        using Tensor vMh = new Tensor(kvShape, value.DType);
        using Tensor attnMh = new Tensor(qShape, output.DType);
        Permute0213(qMh, query, sq, heads, headDim);
        Permute0213(kMh, key, skv, kvHeads, headDim);
        Permute0213(vMh, value, skv, kvHeads, headDim);
        ScaledDotProductAttention(attnMh, qMh, kMh, vMh, mask, scale, allowF16);
        Permute0213(output, attnMh, heads, sq, headDim);
    }

    /// <summary>Whether the cooperative-matrix-2 kernel serves this shape.</summary>
    private bool CanUseFlashCm2(int headDim, int hq, int hkv, int sq, int skv, Tensor? mask) =>
        FlashCm2Available && headDim is 64 or 128 && hkv > 0 && hq % hkv == 0
        && (mask is null || (mask.DType == DType.F32 && mask.ElementCount == (long)sq * skv));

    /// <summary>Runs <c>sdpa_flash_cm2</c> in F16 over strided Q/K/V/O; strides are in elements.</summary>
    private void DispatchFlashCm2(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale,
        int batch, int hq, int hkv, int sq, int skv, int headDim, AttnStrides qs, AttnStrides ks, AttnStrides vs, AttnStrides os)
    {
        (VulkanBuffer qRes, VulkanBuffer? qOwned) = CastIfNeeded(query, GetBuffer(query), DType.F16);
        (VulkanBuffer kRes, VulkanBuffer? kOwned) = CastIfNeeded(key, GetBuffer(key), DType.F16);
        (VulkanBuffer vRes, VulkanBuffer? vOwned) = CastIfNeeded(value, GetBuffer(value), DType.F16);
        VulkanBuffer? maskBuf = mask is null ? null : GetBuffer(mask);
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(output.ElementCount * DType.F16.SizeInBytes));
        try
        {
            ReadOnlySpan<SpecConstant> spec = new SpecConstant[]
            {
                SpecConstant.UInt(0, Vk.CoopMat2WorkgroupInvocations),
                SpecConstant.UInt(10, FlashCm2QueryTile),
                SpecConstant.UInt(11, FlashCm2KeyTile),
                SpecConstant.UInt(12, (uint)headDim),
            };
            VulkanKernel k = GetKernel(mask is null ? "sdpa_flash_cm2" : "sdpa_flash_cm2_mask", mask is null ? 4 : 5, spec);
            Span<byte> pc = stackalloc byte[17 * 4];
            BinaryWriteUInt(pc, 0, (uint)hq);
            BinaryWriteUInt(pc, 4, (uint)hkv);
            BinaryWriteUInt(pc, 8, (uint)sq);
            BinaryWriteUInt(pc, 12, (uint)skv);
            BinaryWriteFloat(pc, 16, scale);
            WriteStrides(pc, 20, qs);
            WriteStrides(pc, 32, ks);
            WriteStrides(pc, 44, vs);
            WriteStrides(pc, 56, os);
            Span<ulong> bufs = mask is null
                ? stackalloc ulong[] { qRes.Handle, kRes.Handle, vRes.Handle, outBuf.Handle }
                : stackalloc ulong[] { qRes.Handle, kRes.Handle, vRes.Handle, maskBuf!.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, (uint)((sq + FlashCm2QueryTile - 1) / FlashCm2QueryTile), (uint)hq, (uint)batch);
            CacheOutputCastingFrom(output, outBuf, DType.F16, "sdpa_flash_cm2");
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan sdpa_flash_cm2 dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
        finally
        {
            if (qOwned is not null) _xfer.FreeDevice(qOwned);
            if (kOwned is not null) _xfer.FreeDevice(kOwned);
            if (vOwned is not null) _xfer.FreeDevice(vOwned);
        }
    }

    private static void WriteStrides(Span<byte> pc, int offset, AttnStrides s)
    {
        BinaryWriteUInt(pc, offset, s.Row);
        BinaryWriteUInt(pc, offset + 4, s.Head);
        BinaryWriteUInt(pc, offset + 8, s.Batch);
    }

    /// <summary>Element strides between consecutive sequence rows, heads and batch items.</summary>
    private readonly record struct AttnStrides(uint Row, uint Head, uint Batch);
}
