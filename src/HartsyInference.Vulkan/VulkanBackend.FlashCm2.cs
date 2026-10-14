using HartsyInference.Core.Configuration;
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

    /// <summary>Routes causal prefill attention through <c>sdpa_flash_cm2</c>; see <see cref="TryFlashCm2CausalPrefill"/>.</summary>
    public bool EnableFlashCm2Prefill { get; set; } = EngineKnobs.VkFlashCm2Prefill.Value;

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

    /// <summary>Smallest query count worth a 64-row cooperative-matrix tile; decode (one row) and short chunks stay on the scalar kernel.</summary>
    private const int FlashCm2PrefillMinRows = 64;

    /// <summary>Causal prefill calls served by <c>sdpa_flash_cm2</c> on this backend (tests and profiling).</summary>
    public long FlashCm2PrefillDispatches { get; private set; }

    /// <summary>Causal prefill on <c>sdpa_flash_cm2_causal</c>: the query rows sit at absolute positions <c>qOffset + i</c>,
    /// and the kernel computes the causal (and sliding-window) mask from those positions. No mask is built on the host, held
    /// across layers or uploaded, so a prefill of many layers moves no mask bytes. Returns false (leaving
    /// <paramref name="output"/> untouched) when the knob is off or the shape gate misses.</summary>
    private bool TryFlashCm2CausalPrefill(Tensor output, Tensor query, Tensor key, Tensor value, int kvLen, int kvGroup,
        int qOffset, float scale, int slidingWindow)
    {
        if (!EnableFlashCm2Prefill || !FlashCm2Available || query.Shape.Rank != 4 || key.Shape.Rank != 4
            || query.Shape[0] != 1 || query.DType != DType.F32 || key.DType != DType.F32 || value.DType != DType.F32)
            return false;
        int hq = (int)query.Shape[1], sq = (int)query.Shape[2], headDim = (int)query.Shape[3];
        if (sq < FlashCm2PrefillMinRows || kvGroup <= 0 || hq % kvGroup != 0 || kvLen <= 0 || kvLen > key.Shape[2]
            || qOffset < 0 || slidingWindow < 0)
            return false;
        int hkv = hq / kvGroup;
        if (!CanUseFlashCm2(headDim, hq, hkv, sq, kvLen, mask: null)) return false;
        // Shape policy, not a memory bound any more: above the bound the scalar kernel runs, as it does on main.
        // FlashCm2PrefillPolicy says why the bound stays.
        if (!FlashCm2PrefillPolicy.MaskFits(sq, kvLen)) return false;

        long kvCap = key.Shape[2];
        AttnStrides qs = new((uint)headDim, (uint)(sq * headDim), (uint)((long)sq * headDim * hq));
        AttnStrides ks = new((uint)headDim, (uint)(kvCap * headDim), (uint)(kvCap * headDim * hkv));
        DispatchFlashCm2(output, query, key, value, mask: null, scale, batch: 1, hq, hkv, sq, kvLen, headDim,
            qs, ks, ks, qs, new CausalWindow(qOffset, slidingWindow));
        FlashCm2PrefillDispatches++;
        return true;
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

    /// <summary>The causal (and sliding-window) mask a cm2 prefill computes in the kernel: query row <c>i</c> sits at absolute
    /// position <see cref="QOffset"/> + i, and <see cref="Window"/> 0 means no window.</summary>
    private readonly record struct CausalWindow(int QOffset, int Window);

    /// <summary>Runs <c>sdpa_flash_cm2</c> in F16 over strided Q/K/V/O; strides are in elements. With <paramref name="causal"/>
    /// set it runs <c>sdpa_flash_cm2_causal</c>, which computes that mask itself, so <paramref name="mask"/> must be null.</summary>
    private void DispatchFlashCm2(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale,
        int batch, int hq, int hkv, int sq, int skv, int headDim, AttnStrides qs, AttnStrides ks, AttnStrides vs, AttnStrides os,
        CausalWindow? causal = null)
    {
        if (causal is not null && mask is not null)
            throw new ArgumentException("A causal cm2 dispatch computes its own mask and takes no mask buffer.", nameof(mask));
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
            string kernelName = causal is not null ? "sdpa_flash_cm2_causal" : mask is null ? "sdpa_flash_cm2" : "sdpa_flash_cm2_mask";
            VulkanKernel k = GetKernel(kernelName, mask is null ? 4 : 5, spec);
            // 17 words as before, plus qOffset and window for the causal variant (zero for the others, which do not read them).
            Span<byte> pc = stackalloc byte[19 * 4];
            BinaryWriteUInt(pc, 0, (uint)hq);
            BinaryWriteUInt(pc, 4, (uint)hkv);
            BinaryWriteUInt(pc, 8, (uint)sq);
            BinaryWriteUInt(pc, 12, (uint)skv);
            BinaryWriteFloat(pc, 16, scale);
            WriteStrides(pc, 20, qs);
            WriteStrides(pc, 32, ks);
            WriteStrides(pc, 44, vs);
            WriteStrides(pc, 56, os);
            BinaryWriteUInt(pc, 68, (uint)(causal?.QOffset ?? 0));
            BinaryWriteUInt(pc, 72, (uint)(causal?.Window ?? 0));
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
