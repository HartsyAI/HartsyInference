using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// The attention dispatch as capability rows (docs/ATTENTION_CONTRACT.md §6), in the order each entry point's branches
// were tested, each admission the branch's predicate and each launch its body.
public sealed partial class VulkanBackend
{
    /// <summary>The dispatch branch the last attention call on this thread took (A.n head-major, B.n token-major,
    /// C.n FlashAttention, D.n device positions; nested as "B.2/A.1").</summary>
    [ThreadStatic] private static string? t_attentionRoute;
    internal static string? LastAttentionRoute => t_attentionRoute;

    /// <summary>The caller's FlashAttention <c>kvGroup</c> for the rows of the call in flight.</summary>
    [ThreadStatic] private static int t_flashKvGroup;

    /// <summary>The key-only mask expanded to every query row for the token-major call in flight; the cm2 row reads it,
    /// the permute fallback hands the head-major entry point the original (which expands it again).</summary>
    [ThreadStatic] private static Tensor? t_expandedMask;

    private AttentionKernelRow[]? _denseRows, _tokenMajorRows, _flashRows, _flashDeviceRows;

    private AttentionKernelRow[] DenseAttentionRows => _denseRows ??= BuildDenseRows();
    private AttentionKernelRow[] TokenMajorAttentionRows => _tokenMajorRows ??= BuildTokenMajorRows();
    private AttentionKernelRow[] FlashAttentionRows => _flashRows ??= BuildFlashRows();
    private AttentionKernelRow[] FlashAttentionDeviceRows => _flashDeviceRows ??= BuildFlashDeviceRows();

    private void RunAttentionRows(AttentionKernelRow[] rows, in AttentionCall call, int flashKvGroup = 0)
    {
        int saved = t_flashKvGroup;
        t_flashKvGroup = flashKvGroup;
        try
        {
            AttentionKernelRow row = AttentionPlanner.Run(rows, in call);
            if (row is DelegateAttentionRow { Branch: { } branch }) t_attentionRoute = branch;
            AttentionPlanLog.Ran("vulkan", row, in call);
        }
        finally
        {
            t_flashKvGroup = saved;
        }
    }

    private AttentionKernelRow[] RowsFor(AttentionRoute route) => route switch
    {
        AttentionRoute.Dense => DenseAttentionRows,
        AttentionRoute.TokenMajor or AttentionRoute.TokenMajorGqa => TokenMajorAttentionRows,
        AttentionRoute.Flash => FlashAttentionRows,
        AttentionRoute.FlashDevice => FlashAttentionDeviceRows,
        _ => throw new AttentionUnsupportedException("Vulkan plans route-pinned calls only."),
    };

    /// <inheritdoc/>
    /// <remarks>Re-enters through the call's legacy entry point, which validates it and runs its rows; the entry points never
    /// call this method, so the two cannot recurse.</remarks>
    public void Attention(in AttentionCall call)
    {
        if (call.LegacyGap() is { } gap)
            throw new AttentionUnsupportedException($"Vulkan attention serves a call its legacy entry point can carry; this one has {gap}.");
        Tensor o = call.Output.Storage, q = call.Query.Storage, k = call.Key.Storage, v = call.Value.Storage;
        switch (call.Route)
        {
            case AttentionRoute.Dense:
                ScaledDotProductAttention(o, q, k, v, call.Mask.Bias, call.Scale, call.Numerics.LegacyAssertedBounded);
                break;
            case AttentionRoute.TokenMajor:
                ScaledDotProductAttentionTokenMajor(o, q, k, v, call.Mask.Bias, call.Heads, call.QkHeadDim, call.Scale,
                    call.Numerics.LegacyAssertedBounded);
                break;
            case AttentionRoute.TokenMajorGqa:
                ScaledDotProductAttentionTokenMajor(o, q, k, v, call.Mask.Bias, call.Heads, call.KvHeads, call.QkHeadDim, call.Scale,
                    call.Numerics.LegacyAssertedBounded);
                break;
            case AttentionRoute.Flash:
                FlashAttention(o, q, k, v, call.ValidKvLength, call.KvGroup, call.Mask.Causal, call.Positions.QueryOffset, call.Scale,
                    call.Logits.Softcap, call.Logits.Sinks, call.Mask.LegacySlidingWindow, call.Logits.AlibiSlopes);
                break;
            case AttentionRoute.FlashDevice:
                FlashAttentionDev(o, q, k, v, call.ValidKvLength, call.KvGroup, call.Mask.Causal, call.Positions.QueryOffset, call.Scale,
                    call.Positions.DeviceHandle, call.Logits.Softcap, call.Mask.LegacySlidingWindow);
                break;
        }
    }

    /// <inheritdoc/>
    public AttentionPlan PlanAttention(in AttentionCall call)
    {
        // The token-major entry serves grouped K/V on its own rows and records the call as token-major; the rows read the
        // legacy kvLen, which a call without bounded positions leaves at 0 while Attention() passes all keys.
        AttentionCall planned = call.Route == AttentionRoute.TokenMajorGqa ? call with { Route = AttentionRoute.TokenMajor } : call;
        if (planned.Route is AttentionRoute.Flash or AttentionRoute.FlashDevice && !planned.Positions.Bounded)
            planned = planned with { Positions = planned.Positions with { KvLength = planned.KeyTokens, Bounded = true } };
        int saved = t_flashKvGroup;
        t_flashKvGroup = call.KvGroup;
        try { return AttentionPlanner.Plan(RowsFor(planned.Route), in planned); }
        finally { t_flashKvGroup = saved; }
    }

    /// <summary>B.1's admission. A key-only mask reaches cm2 as the <c>[Sq, Skv]</c> copy the token-major entry makes of it,
    /// so a plan made before that copy exists judges it by the copy's shape.</summary>
    private bool TokenMajorCm2Admits(in AttentionCall c)
    {
        Tensor? mask = t_expandedMask ?? c.Mask.Bias;
        bool expandsToFull = t_expandedMask is null && mask is not null && mask.DType == DType.F32
            && mask.ElementCount == c.KeyTokens && c.QueryTokens > 1;
        return CanUseFlashCm2(c.QkHeadDim, c.Heads, c.KvHeads, c.QueryTokens, c.KeyTokens, expandsToFull ? null : mask);
    }

    private static AttentionKernelCapability Cap(string id, AttentionRoute entry, AttentionDTypes q, AttentionDTypes kv,
        AttentionLayouts layouts) => new() { Id = id, Entry = entry, QueryTypes = q, KvTypes = kv, Layouts = layouts };

    private const AttentionMaskSupport AnyBias = AttentionMaskSupport.HeadSharedBias | AttentionMaskSupport.PerHeadBias;
    private const AttentionMaskSupport RowCausalWindow = AttentionMaskSupport.Causal | AttentionMaskSupport.Window;

    // ── head-major ScaledDotProductAttention: A.1-A.3 ──────────────────────────────────────────────────────────────

    private AttentionKernelRow[] BuildDenseRows()
    {
        const AttentionRoute E = AttentionRoute.Dense;
        const AttentionLayouts HM = AttentionLayouts.HeadMajor;
        return
        [
            new DelegateAttentionRow(Cap("cm2.flash", E, AttentionDTypes.Any, AttentionDTypes.Any, HM) with
                {
                    MinHeadDim = 64, MaxHeadDim = 128, Gqa = true, Masks = AnyBias, UsesQueryKey = AttentionFormats.F16,
                    UsesProbValue = AttentionFormats.F16, Lowering = AttentionLowering.Cast,
                }, "A.1",
                (in AttentionCall c) =>
                {
                    bool f16Inputs = c.Query.DType == DType.F16 && c.Key.DType == DType.F16 && c.Value.DType == DType.F16;
                    return (c.Numerics.LegacyAssertedBounded || f16Inputs)
                        && CanUseFlashCm2(c.QkHeadDim, c.Heads, c.KvHeads, c.QueryTokens, c.KeyTokens, c.Mask.Bias);
                },
                (in AttentionCall c) =>
                {
                    int hq = c.Heads, hkv = c.KvHeads, headDim = c.QkHeadDim, sq = c.QueryTokens, skv = c.KeyTokens;
                    uint qHead = (uint)(sq * headDim), kvHead = (uint)(skv * headDim);
                    DispatchFlashCm2(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Mask.Bias, c.Scale, c.Batch, hq, hkv, sq,
                        skv, headDim, new AttnStrides((uint)headDim, qHead, qHead * (uint)hq), new AttnStrides((uint)headDim, kvHead, kvHead * (uint)hkv),
                        new AttnStrides((uint)headDim, kvHead, kvHead * (uint)hkv), new AttnStrides((uint)headDim, qHead, qHead * (uint)hq));
                    return true;
                }),
            new DelegateAttentionRow(Cap("flash.scalar", E, AttentionDTypes.Any, AttentionDTypes.Any, HM) with
                {
                    MaxHeadDim = FlashMaxHeadDim, Gqa = true, Masks = AnyBias, Lowering = AttentionLowering.Cast,
                }, "A.2",
                static (in AttentionCall c) => c.QkHeadDim <= FlashMaxHeadDim && c.KvHeads > 0 && c.Heads % c.KvHeads == 0,
                (in AttentionCall c) =>
                {
                    DispatchFlashAttention(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Mask.Bias, c.Scale,
                        causal: false, qOffset: 0, slidingWindow: 0, skv: c.KeyTokens, kvGroup: c.Heads / c.KvHeads);
                    return true;
                }),
            new DelegateAttentionRow(Cap("naive", E, AttentionDTypes.Any, AttentionDTypes.Any, HM) with
                {
                    Gqa = true, Masks = AnyBias, Lowering = AttentionLowering.MaterializeScores | AttentionLowering.Cast,
                }, "A.3",
                static (in AttentionCall c) => true,
                (in AttentionCall c) =>
                {
                    ScaledDotProductAttentionNaive(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Mask.Bias, c.Scale,
                        c.Numerics.LegacyAssertedBounded);
                    return true;
                }),
        ];
    }

    // ── token-major ScaledDotProductAttentionTokenMajor: B.1-B.2 ──────────────────────────────────────────────────

    private AttentionKernelRow[] BuildTokenMajorRows()
    {
        const AttentionRoute E = AttentionRoute.TokenMajor;
        const AttentionLayouts TM = AttentionLayouts.TokenMajor;
        return
        [
            new DelegateAttentionRow(Cap("cm2.flash", E, AttentionDTypes.Any, AttentionDTypes.Any, TM) with
                {
                    MinHeadDim = 64, MaxHeadDim = 128, Gqa = true, Masks = AnyBias, UsesQueryKey = AttentionFormats.F16,
                    UsesProbValue = AttentionFormats.F16, Lowering = AttentionLowering.Cast,
                }, "B.1",
                (in AttentionCall c) => TokenMajorCm2Admits(in c),
                (in AttentionCall c) =>
                {
                    uint qRow = (uint)((long)c.Heads * c.QkHeadDim), kvRow = (uint)((long)c.KvHeads * c.QkHeadDim);
                    DispatchFlashCm2(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, t_expandedMask ?? c.Mask.Bias, c.Scale,
                        batch: 1, c.Heads, c.KvHeads, c.QueryTokens, c.KeyTokens, c.QkHeadDim,
                        new AttnStrides(qRow, (uint)c.QkHeadDim, 0), new AttnStrides(kvRow, (uint)c.QkHeadDim, 0),
                        new AttnStrides(kvRow, (uint)c.QkHeadDim, 0), new AttnStrides(qRow, (uint)c.QkHeadDim, 0));
                    return true;
                }),
            new DelegateAttentionRow(Cap("permute.dense", E, AttentionDTypes.Any, AttentionDTypes.Any, TM) with
                {
                    Gqa = true, Masks = AnyBias, Lowering = AttentionLowering.Permute,
                }, null,
                static (in AttentionCall c) => true,
                (in AttentionCall c) =>
                {
                    AttendTokenMajorViaHeadMajor(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Mask.Bias, c.Heads,
                        c.KvHeads, c.QkHeadDim, c.QueryTokens, c.KeyTokens, c.Scale, c.Numerics.LegacyAssertedBounded);
                    t_attentionRoute = AttentionRouteText.Nested("B.2/", t_attentionRoute);
                    return true;
                }),
        ];
    }

    // ── FlashAttention: C.1-C.4 ───────────────────────────────────────────────────────────────────────────────────

    private bool FlashOnDevice(in AttentionCall c) =>
        c.Logits.Softcap == 0f && c.Logits.Sinks is null && c.Logits.AlibiSlopes is null && c.QkHeadDim <= FlashMaxHeadDim;

    private AttentionKernelRow[] BuildFlashRows()
    {
        const AttentionRoute E = AttentionRoute.Flash;
        const AttentionLayouts HM = AttentionLayouts.HeadMajor;
        return
        [
            // Any dtype: the reference refuses non-F32 itself, with the exception callers have always seen.
            new DelegateAttentionRow(Cap("host.reference", E, AttentionDTypes.Any, AttentionDTypes.Any, HM) with
                {
                    Gqa = true, Masks = RowCausalWindow, LogitTerms = true, Lowering = AttentionLowering.HostReference,
                }, "C.1",
                (in AttentionCall c) => !FlashOnDevice(in c),
                static (in AttentionCall c) =>
                {
                    AttentionReference.FlashAttention(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Positions.KvLength,
                        t_flashKvGroup, c.Mask.Causal, c.Positions.QueryOffset, c.Scale, c.Logits.Softcap, c.Logits.Sinks,
                        c.Mask.LegacySlidingWindow, c.Logits.AlibiSlopes);
                    return true;
                }),
            new DelegateAttentionRow(Cap("cm2.causal", E, AttentionDTypes.Any, AttentionDTypes.Any, HM) with
                {
                    MinHeadDim = 64, MaxHeadDim = 128, Gqa = true, Masks = RowCausalWindow, MinQueryTokens = 64,
                    UsesQueryKey = AttentionFormats.F16, UsesProbValue = AttentionFormats.F16, Lowering = AttentionLowering.Cast,
                }, "C.2",
                (in AttentionCall c) => FlashOnDevice(in c) && c.Mask.Causal,
                (in AttentionCall c) =>
                {
                    return TryFlashCm2CausalPrefill(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Positions.KvLength,
                        t_flashKvGroup, c.Positions.QueryOffset, c.Scale, c.Mask.LegacySlidingWindow);
                }),
            new DelegateAttentionRow(Cap("flash.decode.split", E, AttentionDTypes.F32, AttentionDTypes.F32, HM) with
                {
                    MaxHeadDim = FlashMaxHeadDim, Gqa = true, Masks = RowCausalWindow, MaxQueryTokens = 1,
                }, "C.3",
                (in AttentionCall c) =>
                {
                    int kvLen = c.Positions.KvLength, qOffset = c.Positions.QueryOffset, window = c.Mask.LegacySlidingWindow;
                    return FlashOnDevice(in c) && kvLen >= 1 && kvLen <= c.KeyTokens && qOffset >= 0 && window >= 0
                        && (!c.Mask.Causal || qOffset < kvLen)
                        && CanUseFlashDecode(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, t_flashKvGroup);
                },
                (in AttentionCall c) =>
                {
                    DispatchFlashDecode(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Scale, c.Mask.Causal,
                        c.Positions.QueryOffset, c.Mask.LegacySlidingWindow, c.Positions.KvLength, t_flashKvGroup, devicePos: 0);
                    return true;
                }),
            new DelegateAttentionRow(Cap("flash.scalar", E, AttentionDTypes.Any, AttentionDTypes.Any, HM) with
                {
                    MaxHeadDim = FlashMaxHeadDim, Gqa = true, Masks = RowCausalWindow, Lowering = AttentionLowering.Cast,
                }, "C.4",
                (in AttentionCall c) => FlashOnDevice(in c),
                (in AttentionCall c) =>
                {
                    DispatchFlashAttention(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, mask: null, c.Scale, c.Mask.Causal,
                        c.Positions.QueryOffset, c.Mask.LegacySlidingWindow, skv: c.Positions.KvLength, t_flashKvGroup);
                    return true;
                }),
        ];
    }

    // ── FlashAttentionDev: D.1-D.4 ──────────────────────────────────────────────────────────────────────────────

    private bool DevicePositionUsable(in AttentionCall c) =>
        c.Positions.DeviceHandle != 0 && _scalarBuffers.ContainsKey(c.Positions.DeviceHandle);

    private bool DeviceKernelEligible(in AttentionCall c) =>
        DevicePositionUsable(in c) && c.Logits.Softcap == 0f && c.QkHeadDim <= FlashMaxHeadDim
        && c.Query.DType == DType.F32 && c.Key.DType == DType.F32 && c.Value.DType == DType.F32 && c.Output.DType == DType.F32;

    private AttentionKernelRow[] BuildFlashDeviceRows()
    {
        const AttentionRoute E = AttentionRoute.FlashDevice;
        const AttentionLayouts HM = AttentionLayouts.HeadMajor;
        return
        [
            new DelegateAttentionRow(Cap("flash.decode.split.dev", E, AttentionDTypes.F32, AttentionDTypes.F32, HM) with
                {
                    MaxHeadDim = FlashMaxHeadDim, Gqa = true, Masks = RowCausalWindow, LogitTerms = true, DevicePositions = true, CaptureSafe = true,
                }, "D.1",
                (in AttentionCall c) => DeviceKernelEligible(in c)
                    && CanUseFlashDecode(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, t_flashKvGroup),
                (in AttentionCall c) =>
                {
                    DispatchFlashDecode(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Scale, c.Mask.Causal, qOffset: 0,
                        c.Mask.LegacySlidingWindow, kvLen: 0, t_flashKvGroup, c.Positions.DeviceHandle);
                    return true;
                }),
            new DelegateAttentionRow(Cap("flash.scalar.dev", E, AttentionDTypes.F32, AttentionDTypes.F32, HM) with
                {
                    MaxHeadDim = FlashMaxHeadDim, Gqa = true, Masks = AttentionMaskSupport.Causal, LogitTerms = true, DevicePositions = true,
                    CaptureSafe = true,
                }, "D.2",
                (in AttentionCall c) => DeviceKernelEligible(in c) && c.Mask.LegacySlidingWindow == 0,
                (in AttentionCall c) =>
                {
                    DispatchFlashAttentionDev(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Scale, c.Mask.Causal,
                        t_flashKvGroup, c.Positions.DeviceHandle);
                    return true;
                }),
            // With a valid device position the host kvLen/qOffset are placeholders, so the eager path has nothing correct to run.
            new DelegateAttentionRow(Cap("refuse", E, AttentionDTypes.Any, AttentionDTypes.Any, HM) with
                {
                    Gqa = true, Masks = RowCausalWindow, LogitTerms = true, DevicePositions = true,
                }, "D.3",
                (in AttentionCall c) => DevicePositionUsable(in c),
                static (in AttentionCall c) => throw new NotSupportedException(
                    "FlashAttentionDev on Vulkan covers F32 attention without a soft-cap and with head dim <= "
                    + $"{FlashMaxHeadDim}; a sliding window needs a single query row whose head dim the subgroup divides into at most 4 dims per lane. "
                    + $"Got headDim={c.QkHeadDim}, softcap={c.Logits.Softcap}, window={c.Mask.LegacySlidingWindow}, dtype={c.Query.DType}. "
                    + "Leave graph decode off for this model.")),
            new DelegateAttentionRow(Cap("flash.eager", E, AttentionDTypes.Any, AttentionDTypes.Any, HM) with
                {
                    Gqa = true, Masks = RowCausalWindow, LogitTerms = true, DevicePositions = true,
                }, null,
                static (in AttentionCall c) => true,
                (in AttentionCall c) =>
                {
                    FlashAttention(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Positions.KvLength, t_flashKvGroup,
                        c.Mask.Causal, c.Positions.QueryOffset, c.Scale, c.Logits.Softcap, sink: null, c.Mask.LegacySlidingWindow, alibiSlopes: null);
                    t_attentionRoute = AttentionRouteText.Nested("D.4/", t_attentionRoute);
                    return true;
                }),
        ];
    }
}
