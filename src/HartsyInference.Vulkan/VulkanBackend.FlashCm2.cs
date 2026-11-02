using HartsyInference.Core.Backends;
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

    private bool FlashCm2Available => EnableCoopMat2 && Vk.HasCooperativeMatrix2 && Vk.HasCoopMat2AttentionFeatures
        && FitsGranularity(FlashCm2QueryTile, Vk.CoopMat2MGranularity) && FitsGranularity(FlashCm2KeyTile, Vk.CoopMat2NGranularity)
        && FitsGranularity(FlashCm2KeyTile, Vk.CoopMat2KGranularity);

    /// <summary>A tile edge the configuration's granularity divides (QKᵀ is Br×Bc over D, PV is Br×D over Bc).</summary>
    private static bool FitsGranularity(int edge, uint granularity) => granularity != 0 && edge % granularity == 0;

    /// <summary>Routes causal prefill attention through <c>sdpa_flash_cm2</c>; see <see cref="TryFlashCm2CausalPrefill"/>.</summary>
    public bool EnableFlashCm2Prefill { get; set; } = EngineKnobs.VkFlashCm2Prefill.Value;

    /// <inheritdoc/>
    public void ScaledDotProductAttentionTokenMajor(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask,
        int heads, int headDim, float scale, bool allowF16 = false) =>
        ScaledDotProductAttentionTokenMajor(output, query, key, value, mask, heads, heads, headDim, scale, allowF16);

    private void ScaledDotProductAttentionTokenMajor(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask,
        int heads, int headDim, float scale, AttentionNumerics numerics) =>
        ScaledDotProductAttentionTokenMajor(output, query, key, value, mask, heads, heads, headDim, scale, numerics);

    /// <inheritdoc/>
    public void ScaledDotProductAttentionTokenMajor(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask,
        int heads, int kvHeads, int headDim, float scale, bool allowF16 = false) =>
        ScaledDotProductAttentionTokenMajor(output, query, key, value, mask, heads, kvHeads, headDim, scale, AttentionNumerics.FromLegacy(allowF16));

    private void ScaledDotProductAttentionTokenMajor(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask,
        int heads, int kvHeads, int headDim, float scale, AttentionNumerics numerics)
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
        if (mask is not null) _ = AttentionMaskLayout.Of(mask, 1, heads, sq, skv);
        // Vulkan reads a token-major operand as one sequence whatever its rank, so the call is described that way.
        AttentionCall call = new()
        {
            Query = new(query, AttentionLayout.TokenMajor, 1, heads, sq, headDim, 0, qWidth * sq, headDim, qWidth),
            Key = new(key, AttentionLayout.TokenMajor, 1, kvHeads, skv, headDim, 0, kvWidth * skv, headDim, kvWidth),
            Value = new(value, AttentionLayout.TokenMajor, 1, kvHeads, skv, headDim, 0, kvWidth * skv, headDim, kvWidth),
            Output = new(output, AttentionLayout.TokenMajor, 1, heads, sq, headDim, 0, qWidth * sq, headDim, qWidth),
            Scale = scale, Mask = AttentionMask.Additive(mask), Numerics = numerics,
            Route = AttentionRoute.TokenMajor,
        };
        RunAttentionRows(TokenMajorAttentionRows, in call);
    }

    /// <summary>Smallest query count worth a 64-row cooperative-matrix tile; decode (one row) and short chunks stay on the scalar kernel.</summary>
    private const int FlashCm2PrefillMinRows = 64;

    /// <summary>Causal prefill calls served by <c>sdpa_flash_cm2</c> on this backend (tests and profiling).</summary>
    public long FlashCm2PrefillDispatches { get; private set; }

    /// <summary>Causal prefill on <c>sdpa_flash_cm2_causal</c>: the query rows sit at absolute positions <c>qOffset + i</c>,
    /// and the kernel computes the causal (and sliding-window) mask from those positions. No mask is built on the host, held
    /// across layers or uploaded, so a prefill of many layers moves no mask bytes. Returns false (leaving
    /// <paramref name="output"/> untouched) when the knob is off or the shape gate misses. Like the decode path, it needs the
    /// first query row inside the keys (<c>qOffset &lt; kvLen</c>).</summary>
    private bool TryFlashCm2CausalPrefill(Tensor output, Tensor query, Tensor key, Tensor value, int kvLen, int kvGroup,
        int qOffset, float scale, int slidingWindow)
    {
        if (!EnableFlashCm2Prefill || !FlashCm2Available || query.Shape.Rank != 4 || key.Shape.Rank != 4
            || query.Shape[0] != 1 || query.DType != DType.F32 || key.DType != DType.F32 || value.DType != DType.F32)
            return false;
        int hq = (int)query.Shape[1], sq = (int)query.Shape[2], headDim = (int)query.Shape[3];
        if (sq < FlashCm2PrefillMinRows || kvGroup <= 0 || hq % kvGroup != 0 || kvLen <= 0 || kvLen > key.Shape[2]
            || qOffset < 0 || qOffset >= kvLen || slidingWindow < 0)
            return false;
        int hkv = hq / kvGroup;
        if (!CanUseFlashCm2(headDim, hq, hkv, sq, kvLen, mask: null)) return false;
        // Shape policy, not a memory bound any more: above the bound the scalar kernel runs, as it does on main.
        // FlashCm2PrefillPolicy says why the bound stays.
        if (!FlashCm2PrefillPolicy.MaskFits(sq, kvLen)) return false;

        long kvCap = key.Shape[2];
        AttnStrides qs = new((uint)headDim, (uint)(sq * headDim), (uint)((long)sq * headDim * hq));
        AttnStrides ks = new((uint)headDim, (uint)(kvCap * headDim), (uint)(kvCap * headDim * hkv));
        DispatchFlashCm2Causal(output, query, key, value, scale, batch: 1, hq, hkv, sq, kvLen, headDim,
            qs, ks, ks, qs, new CausalWindow(qOffset, slidingWindow));
        FlashCm2PrefillDispatches++;
        return true;
    }

    /// <summary>Shapes the cooperative-matrix-2 kernel does not serve: permute to [1,H,S,D], attend, permute back.</summary>
    private void AttendTokenMajorViaHeadMajor(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask,
        int heads, int kvHeads, int headDim, int sq, int skv, float scale, AttentionNumerics numerics)
    {
        if (query.DType == DType.BF16 || key.DType == DType.BF16 || value.DType == DType.BF16 || output.DType == DType.BF16)
        {
            // The permutes have no BF16 kernel, and BF16 has more range than F16: permute F32 copies and compute in F32.
            Tensor q32 = F32Copy(query), k32 = F32Copy(key), v32 = F32Copy(value);
            Tensor o32 = output.DType == DType.F32 ? output : new Tensor(output.Shape, DType.F32);
            try
            {
                AttendTokenMajorViaHeadMajor(o32, q32, k32, v32, mask, heads, kvHeads, headDim, sq, skv, scale,
                    numerics with { LegacyAssertedBounded = false });
                if (!ReferenceEquals(o32, output)) CastToBf16(output, o32);
            }
            finally
            {
                if (!ReferenceEquals(q32, query)) q32.Dispose();
                if (!ReferenceEquals(k32, key)) k32.Dispose();
                if (!ReferenceEquals(v32, value)) v32.Dispose();
                if (!ReferenceEquals(o32, output)) o32.Dispose();
            }
            return;
        }
        TensorShape qShape = new TensorShape(1, heads, sq, headDim);
        TensorShape kvShape = new TensorShape(1, kvHeads, skv, headDim);
        using Tensor qMh = new Tensor(qShape, query.DType);
        using Tensor kMh = new Tensor(kvShape, key.DType);
        using Tensor vMh = new Tensor(kvShape, value.DType);
        using Tensor attnMh = new Tensor(qShape, output.DType);
        Permute0213(qMh, query, sq, heads, headDim);
        Permute0213(kMh, key, skv, kvHeads, headDim);
        Permute0213(vMh, value, skv, kvHeads, headDim);
        ScaledDotProductAttention(attnMh, qMh, kMh, vMh, mask, scale, numerics);
        Permute0213(output, attnMh, heads, sq, headDim);
    }

    /// <summary><paramref name="t"/> itself when it is F32, else an F32 copy the caller disposes.</summary>
    private Tensor F32Copy(Tensor t)
    {
        if (t.DType == DType.F32) return t;
        (_, VulkanBuffer? owned) = CastIfNeeded(t, GetBuffer(t), DType.F32);
        if (owned is null) return t.CastTo(DType.F32);   // the cached weight cast stays cached; CastTo returns a fresh copy
        Tensor copy = new Tensor(t.Shape, DType.F32);
        CacheOutput(copy, owned);
        return copy;
    }

    /// <summary>Whether the cooperative-matrix-2 kernel serves this shape.</summary>
    /// <remarks>Any mask the entry validated: a key-only one reaches the kernel as the <c>[Sq, Skv]</c> copy
    /// <see cref="WithRowsForCm2"/> makes, since a zero row stride in a cooperative-matrix tensor layout is unproven.</remarks>
    private bool CanUseFlashCm2(int headDim, int hq, int hkv, int sq, int skv, Tensor? mask) =>
        FlashCm2Available && headDim is 64 or 128 && FitsGranularity(headDim, Vk.CoopMat2KGranularity)
        && FitsGranularity(headDim, Vk.CoopMat2NGranularity) && hkv > 0 && hq % hkv == 0 && (mask is null || mask.DType == DType.F32);

    /// <summary>A key-only mask expanded to one row per query for the cm2 kernel, else null.</summary>
    private static Tensor? WithRowsForCm2(Tensor? mask, int batch, int heads, int sq, int skv) =>
        mask is not null && sq > 1 && AttentionMaskLayout.Of(mask, batch, heads, sq, skv).RowStride == 0
            ? ExpandKeyOnlyMask(mask, sq, skv) : null;

    /// <summary>The causal (and sliding-window) mask a cm2 prefill computes in the kernel: query row <c>i</c> sits at absolute
    /// position <see cref="QOffset"/> + i, and <see cref="Window"/> 0 means no window.</summary>
    private readonly record struct CausalWindow(int QOffset, int Window);

    /// <summary>Runs <c>sdpa_flash_cm2</c> in F16 over strided Q/K/V/O, or <c>sdpa_flash_cm2_mask</c> when a mask is given. Strides
    /// are in elements. The causal prefill has its own entry, <see cref="DispatchFlashCm2Causal"/>, so a mask and a causal
    /// window cannot be passed together.</summary>
    private void DispatchFlashCm2(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale,
        int batch, int hq, int hkv, int sq, int skv, int headDim, AttnStrides qs, AttnStrides ks, AttnStrides vs, AttnStrides os)
    {
        using Tensor? expanded = WithRowsForCm2(mask, batch, hq, sq, skv);
        mask = expanded ?? mask;
        Span<byte> pc = stackalloc byte[(mask is null ? 17 : 20) * 4];
        WriteFlashCm2Common(pc, hq, hkv, sq, skv, scale, qs, ks, vs, os);
        if (mask is not null)
        {
            AttentionMaskLayout m = AttentionMaskLayout.Of(mask, batch, hq, sq, skv);
            // A one-row mask still gets a real row stride: only row 0 is in bounds, and no layout stride is ever 0.
            uint rowStride = m.RowStride == 0 ? (uint)skv : (uint)m.RowStride;
            WriteStrides(pc, 68, new AttnStrides(rowStride, (uint)m.HeadStride, (uint)m.BatchStride));
        }
        RunFlashCm2(output, query, key, value, mask, mask is null ? "sdpa_flash_cm2" : "sdpa_flash_cm2_mask", pc,
            batch, hq, sq, headDim);
    }

    /// <summary>Runs <c>sdpa_flash_cm2_causal</c>: the kernel computes the causal and sliding-window mask from
    /// <paramref name="causal"/>, so there is no mask buffer to pass.</summary>
    private void DispatchFlashCm2Causal(Tensor output, Tensor query, Tensor key, Tensor value, float scale,
        int batch, int hq, int hkv, int sq, int skv, int headDim, AttnStrides qs, AttnStrides ks, AttnStrides vs, AttnStrides os,
        CausalWindow causal)
    {
        Span<byte> pc = stackalloc byte[19 * 4];
        WriteFlashCm2Common(pc, hq, hkv, sq, skv, scale, qs, ks, vs, os);
        BinaryWriteUInt(pc, 68, (uint)causal.QOffset);
        BinaryWriteUInt(pc, 72, (uint)causal.Window);
        RunFlashCm2(output, query, key, value, null, "sdpa_flash_cm2_causal", pc, batch, hq, sq, headDim);
    }

    /// <summary>The 17 push-constant words every cm2 kernel reads: head and row counts, the scale and the Q/K/V/O strides.</summary>
    private static void WriteFlashCm2Common(Span<byte> pc, int hq, int hkv, int sq, int skv, float scale,
        AttnStrides qs, AttnStrides ks, AttnStrides vs, AttnStrides os)
    {
        BinaryWriteUInt(pc, 0, (uint)hq);
        BinaryWriteUInt(pc, 4, (uint)hkv);
        BinaryWriteUInt(pc, 8, (uint)sq);
        BinaryWriteUInt(pc, 12, (uint)skv);
        BinaryWriteFloat(pc, 16, scale);
        WriteStrides(pc, 20, qs);
        WriteStrides(pc, 32, ks);
        WriteStrides(pc, 44, vs);
        WriteStrides(pc, 56, os);
    }

    /// <summary>Casts Q/K/V to F16 where they are not, dispatches <paramref name="kernelName"/> over the query row tiles and the
    /// heads, and casts the output back to its own dtype. <paramref name="mask"/> is bound only for the masked kernel.</summary>
    private void RunFlashCm2(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, string kernelName,
        ReadOnlySpan<byte> pc, int batch, int hq, int sq, int headDim)
    {
        (VulkanBuffer qRes, VulkanBuffer? qOwned) = CastIfNeeded(query, GetBuffer(query), DType.F16, saturate: true);
        (VulkanBuffer kRes, VulkanBuffer? kOwned) = CastIfNeeded(key, GetBuffer(key), DType.F16, saturate: true);
        (VulkanBuffer vRes, VulkanBuffer? vOwned) = CastIfNeeded(value, GetBuffer(value), DType.F16, saturate: true);
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
            VulkanKernel k = GetKernel(kernelName, mask is null ? 4 : 5, spec);
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
