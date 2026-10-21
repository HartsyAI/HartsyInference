using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

// The attention dispatch as capability rows (docs/ATTENTION_CONTRACT.md §6). Each entry point's rows run in the order its
// old if-ladder tested them, each row's admission is that branch's predicate and its launch is that branch's body, so a
// call takes the branch it took before; AttentionRouteGoldenTests pins the routes.
public sealed partial class CudaBackend
{
    /// <summary>The caller's FlashAttention <c>kvGroup</c> for the rows of the call in flight: kernels and the device-path
    /// validator take the raw value, and the descriptor only knows <c>Hq / Hkv</c>.</summary>
    // TODO(attention phase 4): carry the raw kvGroup in AttentionCall and delete this side channel.
    [ThreadStatic] private static int t_flashKvGroup;

    private AttentionKernelRow[]? _denseRows, _tokenMajorRows, _tokenMajorGqaRows, _flashRows, _flashDeviceRows;

    private AttentionKernelRow[] DenseAttentionRows => _denseRows ??= BuildDenseRows();
    private AttentionKernelRow[] TokenMajorAttentionRows => _tokenMajorRows ??= BuildTokenMajorRows();
    private AttentionKernelRow[] TokenMajorGqaAttentionRows => _tokenMajorGqaRows ??= BuildTokenMajorGqaRows();
    private AttentionKernelRow[] FlashAttentionRows => _flashRows ??= BuildFlashRows();
    private AttentionKernelRow[] FlashAttentionDeviceRows => _flashDeviceRows ??= BuildFlashDeviceRows();

    /// <summary>Runs the first row of <paramref name="rows"/> that covers, admits and completes the call, and records its route.</summary>
    private void RunAttentionRows(AttentionKernelRow[] rows, in AttentionCall call, int flashKvGroup = 0)
    {
        int saved = t_flashKvGroup;
        t_flashKvGroup = flashKvGroup;
        try
        {
            AttentionKernelRow row = AttentionPlanner.Run(rows, in call);
            if (row is DelegateAttentionRow { Branch: { } branch }) t_attentionRoute = branch;
            AttentionPlanLog.Ran("cuda", row, in call);
        }
        finally
        {
            t_flashKvGroup = saved;
        }
    }

    /// <summary>The rows a call pinned to <paramref name="route"/> is planned over.</summary>
    private AttentionKernelRow[] RowsFor(AttentionRoute route) => route switch
    {
        AttentionRoute.Dense => DenseAttentionRows,
        AttentionRoute.TokenMajor => TokenMajorAttentionRows,
        AttentionRoute.TokenMajorGqa => TokenMajorGqaAttentionRows,
        AttentionRoute.Flash => FlashAttentionRows,
        AttentionRoute.FlashDevice => FlashAttentionDeviceRows,
        _ => throw new AttentionUnsupportedException("CUDA plans route-pinned calls only."),
    };

    /// <inheritdoc/>
    /// <remarks>Re-enters through the call's legacy entry point, which validates it and runs its rows; the entry points never
    /// call this method, so the two cannot recurse.</remarks>
    public void Attention(in AttentionCall call)
    {
        if (call.LegacyGap() is { } gap)
            throw new AttentionUnsupportedException($"CUDA attention serves a call its legacy entry point can carry; this one has {gap}.");
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
        // The rows read the legacy kvLen, which a call without bounded positions leaves at 0; Attention() passes all keys.
        AttentionCall planned = call.Route is AttentionRoute.Flash or AttentionRoute.FlashDevice && !call.Positions.Bounded
            ? call with { Positions = call.Positions with { KvLength = call.KeyTokens, Bounded = true } } : call;
        int saved = t_flashKvGroup;
        t_flashKvGroup = call.KvGroup;
        try { return AttentionPlanner.Plan(RowsFor(planned.Route), in planned); }
        finally { t_flashKvGroup = saved; }
    }

    private static AttentionKernelCapability Cap(string id, AttentionRoute entry, AttentionDTypes q, AttentionDTypes kv,
        AttentionLayouts layouts) => new() { Id = id, Entry = entry, QueryTypes = q, KvTypes = kv, Layouts = layouts };

    private const AttentionDTypes AnyFloat = AttentionDTypes.F32 | AttentionDTypes.F16 | AttentionDTypes.BF16;
    private const AttentionMaskSupport AnyBias = AttentionMaskSupport.HeadSharedBias | AttentionMaskSupport.PerHeadBias;

    // ── head-major ScaledDotProductAttention: S.1-S.15 ─────────────────────────────────────────────────────────────

    private AttentionKernelRow[] BuildDenseRows()
    {
        const AttentionRoute E = AttentionRoute.Dense;
        const AttentionLayouts HM = AttentionLayouts.HeadMajor;
        return
        [
            new DelegateAttentionRow(Cap("sage.int8.f16", E, AttentionDTypes.F16, AttentionDTypes.F16, HM) with
                {
                    MinHeadDim = 64, MaxHeadDim = 128, UsesQueryKey = AttentionFormats.Int8, UsesProbValue = AttentionFormats.F16, CaptureSafe = true,
                }, "S.1",
                (in AttentionCall c) => SageF16Preferred(c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Output.Storage, c.Mask.Bias,
                    c.QueryTokens, c.KeyTokens, c.QkHeadDim),
                (in AttentionCall c) => { SageAttentionInt8(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Scale); return true; }),
            new DelegateAttentionRow(Cap("flash.f16", E, AttentionDTypes.F16, AttentionDTypes.F16, HM) with
                {
                    MinHeadDim = 256, MaxHeadDim = 256, UsesQueryKey = AttentionFormats.F16, UsesProbValue = AttentionFormats.F16, CaptureSafe = true,
                }, "S.2",
                (in AttentionCall c) => FlashF16Eligible(c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Output.Storage, c.Mask.Bias, c.QkHeadDim),
                (in AttentionCall c) =>
                {
                    long h = c.Heads, sq = c.QueryTokens, skv = c.KeyTokens, d = c.QkHeadDim;
                    return TryFlashF16(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Scale, c.Batch, c.Heads, c.QueryTokens,
                        c.KeyTokens, c.QkHeadDim, HeadMajorStrides(h, sq, d), HeadMajorStrides(h, sq, d), HeadMajorStrides(h, skv, d),
                        HeadMajorStrides(h, skv, d));
                }),
            new DelegateAttentionRow(Cap("cudnn.f16", E, AttentionDTypes.F16, AttentionDTypes.F16, HM) with
                {
                    MinHeadDim = CudnnSdpa.MinHeadDim, MaxHeadDim = CudnnSdpa.MaxHeadDim, Masks = AttentionMaskSupport.HeadSharedBias,
                    UsesQueryKey = AttentionFormats.F16, UsesProbValue = AttentionFormats.F16,
                }, "S.3",
                (in AttentionCall c) => CudnnMaskCompatible(c.Mask.Bias, c.Batch, c.QueryTokens, c.KeyTokens)
                    && c.Query.DType == DType.F16 && c.Key.DType == DType.F16 && c.Value.DType == DType.F16 && c.Output.DType == DType.F16
                    && _sdpaCudnn && !_cudnnSdpaDead && CudnnSdpaDimEligible(c.QkHeadDim) && CudnnSdpa.ShapeSupported(c.QkHeadDim),
                (in AttentionCall c) => TryCudnnSdpa(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Mask.Bias, c.Scale)),
            // The F32 Sage rows narrow V to F16 with no range check (|V| > 65504 becomes Inf), hence the two opt-in flags.
            new DelegateAttentionRow(Cap("sage.int8.f32", E, AttentionDTypes.F32, AttentionDTypes.F32, HM) with
                {
                    MinHeadDim = 64, MaxHeadDim = 128, UsesQueryKey = AttentionFormats.Int8, UsesProbValue = AttentionFormats.F16, CaptureSafe = true,
                }, "S.4",
                (in AttentionCall c) => c.Query.DType == DType.F32 && SageF32ValueNarrowingEnabled && c.Mask.Bias is null
                    && (c.QkHeadDim == 64 || c.QkHeadDim == 128) && c.KeyTokens >= 2048 && SageF32KernelsReady(c.QueryTokens),
                (in AttentionCall c) => { SageAttentionInt8(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Scale); return true; }),
            new DelegateAttentionRow(Cap("cudnn.f32.sat", E, AttentionDTypes.F32, AttentionDTypes.F32, HM) with
                {
                    MinHeadDim = CudnnSdpa.MinHeadDim, MaxHeadDim = CudnnSdpa.MaxHeadDim, Masks = AttentionMaskSupport.HeadSharedBias,
                    UsesQueryKey = AttentionFormats.F16,
                    UsesProbValue = AttentionFormats.F16, Saturates = true, Lowering = AttentionLowering.Cast,
                }, "S.5",
                (in AttentionCall c) => c.Query.DType == DType.F32 && CudnnMaskCompatible(c.Mask.Bias, c.Batch, c.QueryTokens, c.KeyTokens)
                    && _sdpaCudnn && !_cudnnSdpaDead && CudnnSdpaDimEligible(c.QkHeadDim) && CudnnSdpa.ShapeSupported(c.QkHeadDim)
                    && (c.Numerics.LegacyAssertedBounded || _sdpaF16ForceOn) && !_sdpaF16Disabled,
                (in AttentionCall c) => TryCudnnSdpa(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Mask.Bias, c.Scale)),
            new DelegateAttentionRow(Cap("sage.int8.f32.short", E, AttentionDTypes.F32, AttentionDTypes.F32, HM) with
                {
                    MinHeadDim = 64, MaxHeadDim = 128, UsesQueryKey = AttentionFormats.Int8, UsesProbValue = AttentionFormats.F16, CaptureSafe = true,
                }, "S.6",
                (in AttentionCall c) => c.Mask.Bias is null && c.Query.DType == DType.F32 && SageF32ValueNarrowingEnabled
                    && (c.QkHeadDim == 64 || c.QkHeadDim == 128) && c.KeyTokens >= 1024 && SageF32KernelsReady(c.QueryTokens),
                (in AttentionCall c) => { SageAttentionInt8(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Scale); return true; }),
            new DelegateAttentionRow(Cap("fa2.tf32", E, AttentionDTypes.F32, AttentionDTypes.F32, HM) with
                {
                    MinHeadDim = 64, MaxHeadDim = 128, UsesQueryKey = AttentionFormats.Tf32, UsesProbValue = AttentionFormats.Tf32, CaptureSafe = true,
                }, "S.7",
                (in AttentionCall c) => c.Mask.Bias is null && c.Query.DType == DType.F32 && EngineKnobs.SdpaV2.Value
                    && FlashAttentionV2ContractSatisfied(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Mask.Bias, c.Scale, _allowTf32),
                (in AttentionCall c) => { FlashAttentionV2(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Scale); return true; }),
            new DelegateAttentionRow(Cap("flash.forced", E, AttentionDTypes.F32, AttentionDTypes.F32, HM), null,
                (in AttentionCall c) => c.Mask.Bias is null && c.Query.DType == DType.F32 && EngineKnobs.SdpaForceFlash.Value,
                (in AttentionCall c) =>
                {
                    FlashAttention(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.KeyTokens, kvGroup: 1, causal: false, qOffset: 0,
                        c.Scale);
                    t_attentionRoute = AttentionRouteText.Nested("S.8/", t_attentionRoute);
                    return true;
                }),
            new DelegateAttentionRow(Cap("tiled.f32", E, AttentionDTypes.F32, AttentionDTypes.F32, HM) with
                {
                    UsesQueryKey = AttentionFormats.Tf32, UsesProbValue = AttentionFormats.Tf32,
                }, "S.9",
                (in AttentionCall c) => c.Mask.Bias is null && c.Query.DType == DType.F32 && ScoresExceedHalfFree(in c),
                (in AttentionCall c) => { SdpaTiledF32(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Scale); return true; }),
            new DelegateAttentionRow(Cap("tiled.f32.keybias", E, AttentionDTypes.F32, AttentionDTypes.F32, HM) with
                {
                    Masks = AnyBias, UsesQueryKey = AttentionFormats.Tf32, UsesProbValue = AttentionFormats.Tf32,
                }, "S.10",
                (in AttentionCall c) => c.Query.DType == DType.F32 && MaskIsKeyOnly(c.Mask.Bias, c.QueryTokens) && ScoresExceedHalfFree(in c),
                (in AttentionCall c) =>
                {
                    SdpaTiledF32(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Scale, c.Mask.Bias);
                    return true;
                }),
            new DelegateAttentionRow(Cap("materialized", E, AnyFloat, AnyFloat, HM) with
                {
                    Masks = AnyBias, UsesQueryKey = AttentionFormats.Tf32 | AttentionFormats.F16, UsesProbValue = AttentionFormats.Tf32 | AttentionFormats.F16,
                    Lowering = AttentionLowering.MaterializeScores | AttentionLowering.Cast | AttentionLowering.ExpandMask,
                }, null,
                static (in AttentionCall c) => true,
                (in AttentionCall c) =>
                {
                    SdpaMaterialized(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Mask.Bias, c.Scale,
                        c.Numerics.LegacyAssertedBounded);
                    return true;
                }),
        ];
    }

    /// <summary>The F32 Sage branches' kernel check: the module loaded, and the v0 kernel's full-tile rule when v1 is absent.</summary>
    private bool SageF32KernelsReady(int sq)
    {
        EnsureKernels();
        return _kernels!.HasSageAttentionKernels && (_kernels.HasSageV1 || sq % 32 == 0);
    }

    /// <summary>The query-tiled gate: forced, or an F32 score matrix larger than half of free VRAM.</summary>
    private bool ScoresExceedHalfFree(in AttentionCall c)
    {
        ulong scoreBytes = (ulong)((long)c.Batch * c.Heads) * (ulong)c.QueryTokens * (ulong)c.KeyTokens * sizeof(float);
        (nuint free, _) = _context.GetMemoryInfo();
        return EngineKnobs.SdpaForceTiled.Value || scoreBytes > (ulong)free / 2;
    }

    // ── token-major ScaledDotProductAttentionTokenMajor: M.1-M.3, and the grouped-query lowering ────────────────────

    private AttentionKernelRow[] BuildTokenMajorRows()
    {
        const AttentionRoute E = AttentionRoute.TokenMajor;
        const AttentionLayouts TM = AttentionLayouts.TokenMajor;
        return
        [
            new DelegateAttentionRow(Cap("flash.f16", E, AttentionDTypes.F16, AttentionDTypes.F16, TM) with
                {
                    MinHeadDim = 256, MaxHeadDim = 256, UsesQueryKey = AttentionFormats.F16, UsesProbValue = AttentionFormats.F16, CaptureSafe = true,
                }, "M.1",
                (in AttentionCall c) => FlashF16Eligible(c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Output.Storage, c.Mask.Bias, c.QkHeadDim),
                (in AttentionCall c) =>
                {
                    CudaKernels.FlashStrides tokenMajor = new(0, (ulong)c.QkHeadDim, (ulong)c.Heads * (ulong)c.QkHeadDim);
                    return TryFlashF16(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Scale, 1, c.Heads, c.QueryTokens,
                        c.KeyTokens, c.QkHeadDim, tokenMajor, tokenMajor, tokenMajor, tokenMajor);
                }),
            new DelegateAttentionRow(Cap("cudnn.tokenmajor", E, AttentionDTypes.F16 | AttentionDTypes.F32,
                    AttentionDTypes.F16 | AttentionDTypes.F32, TM) with
                {
                    MinHeadDim = CudnnSdpa.MinHeadDim, MaxHeadDim = CudnnSdpa.MaxHeadDim, Masks = AttentionMaskSupport.HeadSharedBias,
                    UsesQueryKey = AttentionFormats.F16,
                    UsesProbValue = AttentionFormats.F16, Saturates = true,
                }, "M.2",
                (in AttentionCall c) => !(SageTokenMajorDetour && SageF16Preferred(c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Output.Storage,
                        c.Mask.Bias, c.QueryTokens, c.KeyTokens, c.QkHeadDim))
                    && CudnnMaskCompatible(c.Mask.Bias, 1, c.QueryTokens, c.KeyTokens)
                    && _sdpaCudnn && !_cudnnSdpaDead && CudnnSdpaDimEligible(c.QkHeadDim) && CudnnSdpa.ShapeSupported(c.QkHeadDim)
                    && (c.Query.DType == DType.F16
                        || (c.Query.DType == DType.F32 && (c.Numerics.LegacyAssertedBounded || _sdpaF16ForceOn) && !_sdpaF16Disabled)),
                (in AttentionCall c) =>
                {
                    using OpScope _op = EnterOp(nameof(ScaledDotProductAttentionTokenMajor));
                    EnsureKernels();
                    return TryCudnnSdpa(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Mask.Bias, c.Scale,
                        CudnnSdpa.SdpaLayout.TokenMajor, 1, c.Heads, c.QueryTokens, c.KeyTokens, c.QkHeadDim);
                }),
            new DelegateAttentionRow(Cap("permute.dense", E, AnyFloat, AnyFloat, TM) with
                {
                    Masks = AnyBias, Lowering = AttentionLowering.Permute,
                }, null,
                static (in AttentionCall c) => true,
                (in AttentionCall c) => { TokenMajorViaHeadMajor(in c); return true; }),
        ];
    }

    private AttentionKernelRow[] BuildTokenMajorGqaRows() =>
    [
        new DelegateAttentionRow(Cap("permute.widen.dense", AttentionRoute.TokenMajorGqa, AnyFloat, AnyFloat, AttentionLayouts.TokenMajor) with
            {
                Gqa = true, Masks = AnyBias, Lowering = AttentionLowering.Permute | AttentionLowering.WidenKv,
            }, null,
            static (in AttentionCall c) => true,
            (in AttentionCall c) => { GroupedTokenMajorViaHeadMajor(in c); return true; }),
    ];

    /// <summary>The sequence Krea2 issued before CUDA served this overload, in its order: permute Q, K and V, repeat K/V
    /// heads when grouped (dropping the head-major copies), head-major SDPA, permute out.</summary>
    private void GroupedTokenMajorViaHeadMajor(in AttentionCall c)
    {
        Tensor output = c.Output.Storage, query = c.Query.Storage, key = c.Key.Storage, value = c.Value.Storage;
        int heads = c.Heads, kvHeads = c.KvHeads, headDim = c.QkHeadDim, sq = c.QueryTokens, skv = c.KeyTokens;
        TensorShape qMhShape = new(1, heads, sq, headDim), kvMhShape = new(1, kvHeads, skv, headDim);
        Tensor? qMh = null, kMh = null, vMh = null, kRep = null, vRep = null, oMh = null;
        try
        {
            qMh = new Tensor(qMhShape, query.DType);
            Permute0213(qMh, query, sq, heads, headDim);
            kMh = new Tensor(kvMhShape, key.DType);
            Permute0213(kMh, key, skv, kvHeads, headDim);
            vMh = new Tensor(kvMhShape, value.DType);
            Permute0213(vMh, value, skv, kvHeads, headDim);
            if (kvHeads != heads)
            {
                kRep = new Tensor(new TensorShape(1, heads, skv, headDim), key.DType);
                RepeatKvHeads(kRep, kMh, kvHeads, heads / kvHeads);
                vRep = new Tensor(new TensorShape(1, heads, skv, headDim), value.DType);
                RepeatKvHeads(vRep, vMh, kvHeads, heads / kvHeads);
                kMh.Dispose();
                vMh.Dispose();
            }
            oMh = new Tensor(qMhShape, output.DType);
            ScaledDotProductAttention(oMh, qMh, kRep ?? kMh, vRep ?? vMh, c.Mask.Bias, c.Scale, c.Numerics.LegacyAssertedBounded);
            qMh.Dispose();
            (kRep ?? kMh).Dispose();
            (vRep ?? vMh).Dispose();
            Permute0213(output, oMh, heads, sq, headDim);
            t_attentionRoute = AttentionRouteText.Nested("MG/", t_attentionRoute);
        }
        finally
        {
            qMh?.Dispose(); kMh?.Dispose(); vMh?.Dispose(); kRep?.Dispose(); vRep?.Dispose(); oMh?.Dispose();
        }
    }

    /// <summary>Token-major attention as head-major SDPA (M.3): Q/K/V permuted in, the head-major entry point, the output
    /// permuted back, in the order the token-major entry always issued it.</summary>
    private void TokenMajorViaHeadMajor(in AttentionCall c)
    {
        Tensor output = c.Output.Storage, query = c.Query.Storage, key = c.Key.Storage, value = c.Value.Storage;
        int heads = c.Heads, headDim = c.QkHeadDim, sq = c.QueryTokens, skv = c.KeyTokens;
        Tensor qMh = new Tensor(new TensorShape(1, heads, sq, headDim), query.DType);
        Tensor kMh = new Tensor(new TensorShape(1, heads, skv, headDim), key.DType);
        Tensor vMh = new Tensor(new TensorShape(1, heads, skv, headDim), value.DType);
        Tensor oMh = new Tensor(new TensorShape(1, heads, sq, headDim), output.DType);
        try
        {
            Permute0213(qMh, query, sq, heads, headDim);
            Permute0213(kMh, key, skv, heads, headDim);
            Permute0213(vMh, value, skv, heads, headDim);
            ScaledDotProductAttention(oMh, qMh, kMh, vMh, c.Mask.Bias, c.Scale, c.Numerics.LegacyAssertedBounded);
            Permute0213(output, oMh, heads, sq, headDim);
            t_attentionRoute = AttentionRouteText.Nested("M.3/", t_attentionRoute);
        }
        finally
        {
            qMh.Dispose(); kMh.Dispose(); vMh.Dispose(); oMh.Dispose();
        }
    }

    // ── FlashAttention: F.1-F.7 ──────────────────────────────────────────────────────────────────────────────────

    private AttentionKernelRow[] BuildFlashRows()
    {
        const AttentionRoute E = AttentionRoute.Flash;
        const AttentionLayouts HM = AttentionLayouts.HeadMajor;
        const AttentionMaskSupport CW = AttentionMaskSupport.Causal | AttentionMaskSupport.Window;
        return
        [
            new DelegateAttentionRow(Cap("host.reference", E, AttentionDTypes.F32, AttentionDTypes.F32, HM) with
                {
                    MinHeadDim = 1025, Gqa = true, Masks = CW, LogitTerms = true, Lowering = AttentionLowering.HostReference,
                }, "F.1",
                static (in AttentionCall c) => !(c.QkHeadDim > 0 && c.QkHeadDim <= 1024),
                (in AttentionCall c) =>
                {
                    AttentionReference.FlashAttention(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Positions.KvLength,
                        t_flashKvGroup, c.Mask.Causal, c.Positions.QueryOffset, c.Scale, c.Logits.Softcap, c.Logits.Sinks,
                        c.Mask.LegacySlidingWindow, c.Logits.AlibiSlopes);
                    return true;
                }),
            // Causal prefill keeps 64: LLM prefill below that head dim has not been measured on cuDNN.
            new DelegateAttentionRow(Cap("cudnn.causal", E, AttentionDTypes.F32, AttentionDTypes.F32, HM) with
                {
                    MinHeadDim = 64, MaxHeadDim = 256, Gqa = true, Masks = CW, MinQueryTokens = 2, UsesQueryKey = AttentionFormats.F16,
                    UsesProbValue = AttentionFormats.F16, Saturates = true,
                    Lowering = AttentionLowering.Cast | AttentionLowering.MaterializeMask | AttentionLowering.WidenKv,
                }, "F.2",
                (in AttentionCall c) =>
                {
                    Tensor key = c.Key.Storage, value = c.Value.Storage;
                    bool f16Kv = key.DType == DType.F16 && value.DType == DType.F16;
                    int kvLen = c.Positions.KvLength, d = c.QkHeadDim;
                    return c.Mask.Causal && c.QueryTokens > 1 && c.Batch == 1 && c.KeyTokens == kvLen
                        && c.Logits.Softcap == 0f && c.Logits.Sinks is null && c.Logits.AlibiSlopes is null
                        && c.Query.DType == DType.F32 && c.Output.DType == DType.F32 && !f16Kv && value.DType == DType.F32
                        && _kernels is not null && _kernels.HasCausalMaskKernel
                        && _sdpaCudnn && !_cudnnSdpaDead && CudnnSdpaDimEligible(d) && CudnnSdpa.ShapeSupported(d)
                        && CausalMaskFits(c.QueryTokens, kvLen, c.Heads, d, t_flashKvGroup);
                },
                (in AttentionCall c) => TryCausalPrefillSdpa(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Heads, c.KvHeads,
                    c.QueryTokens, c.Positions.KvLength, c.Positions.QueryOffset, c.Mask.LegacySlidingWindow, c.QkHeadDim, c.Scale)),
            new DelegateAttentionRow(Cap("fa2.causal", E, AttentionDTypes.F32, AttentionDTypes.F32 | AttentionDTypes.F16, HM) with
                {
                    MinHeadDim = 64, MaxHeadDim = 128, Gqa = true, Masks = CW, MinQueryTokens = Fa2MinQueryRows, UsesQueryKey = AttentionFormats.F16,
                    UsesProbValue = AttentionFormats.F16, Saturates = true, CaptureSafe = true,
                }, "F.3",
                (in AttentionCall c) =>
                {
                    Tensor key = c.Key.Storage, value = c.Value.Storage;
                    bool f16Kv = key.DType == DType.F16 && value.DType == DType.F16;
                    return c.Mask.Causal && c.QueryTokens >= Fa2MinQueryRows && c.Logits.Sinks is null && c.Logits.AlibiSlopes is null
                        && c.Logits.Softcap == 0f && c.Query.DType == DType.F32 && c.Output.DType == DType.F32
                        && (key.DType == DType.F32 || f16Kv) && value.DType == key.DType
                        && _kernels is { HasFa2Causal: true } && CudaKernels.Fa2SupportsHeadDim(c.QkHeadDim)
                        && c.Heads % Math.Max(1, c.KvHeads) == 0 && CudaKernels.GridYzFits(c.Heads, c.Batch)
                        && (f16Kv || CudaKernels.KvToF16GridFits(c.Positions.KvLength, c.QkHeadDim)) && EngineKnobs.Fa2Prefill.Value;
                },
                (in AttentionCall c) =>
                {
                    Tensor key = c.Key.Storage, value = c.Value.Storage;
                    bool f16Kv = key.DType == DType.F16 && value.DType == DType.F16;
                    int kvGroup = t_flashKvGroup;
                    FlashAttentionFa2(c.Output.Storage, c.Query.Storage, key, value, c.Batch, c.Heads, c.KvHeads, c.QueryTokens, c.KeyTokens,
                        c.QkHeadDim, c.Positions.KvLength, kvGroup <= 0 ? 1 : kvGroup, c.Positions.QueryOffset, c.Scale, c.Mask.LegacySlidingWindow, f16Kv);
                    return true;
                }),
            new DelegateAttentionRow(Cap("flash.decode", E, AttentionDTypes.F32, AttentionDTypes.F32 | AttentionDTypes.F16, HM) with
                {
                    MaxHeadDim = 1024, Gqa = true, Masks = CW, LogitTerms = true, CaptureSafe = true,
                }, null,
                static (in AttentionCall c) => true,
                (in AttentionCall c) =>
                {
                    FlashAttentionDecode(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Positions.KvLength, t_flashKvGroup,
                        c.Mask.Causal, c.Positions.QueryOffset, c.Scale, c.Logits.Softcap, c.Logits.Sinks, c.Mask.LegacySlidingWindow, c.Logits.AlibiSlopes);
                    return true;
                }),
        ];
    }

    // ── FlashAttentionDev: V.0-V.5 ──────────────────────────────────────────────────────────────────────────────

    private AttentionKernelRow[] BuildFlashDeviceRows()
    {
        const AttentionRoute E = AttentionRoute.FlashDevice;
        const AttentionMaskSupport CW = AttentionMaskSupport.Causal | AttentionMaskSupport.Window;
        return
        [
            new DelegateAttentionRow(Cap("flash.eager", E, AnyFloat, AnyFloat, AttentionLayouts.HeadMajor) with
                {
                    Gqa = true, Masks = CW, LogitTerms = true, DevicePositions = true,
                }, null,
                static (in AttentionCall c) =>
                {
                    Tensor key = c.Key.Storage, value = c.Value.Storage;
                    bool devF16Kv = key.DType == DType.F16 && value.DType == DType.F16;
                    return c.Positions.DeviceHandle == 0 || c.Query.DType != DType.F32 || c.Output.DType != DType.F32
                        || (!devF16Kv && (key.DType != DType.F32 || value.DType != DType.F32));
                },
                (in AttentionCall c) =>
                {
                    FlashAttention(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Positions.KvLength, t_flashKvGroup,
                        c.Mask.Causal, c.Positions.QueryOffset, c.Scale, c.Logits.Softcap, sink: null, c.Mask.LegacySlidingWindow, alibiSlopes: null);
                    t_attentionRoute = AttentionRouteText.Nested("V.0/", t_attentionRoute);
                    return true;
                }),
            new DelegateAttentionRow(Cap("flash.decode.device", E, AttentionDTypes.F32, AttentionDTypes.F32 | AttentionDTypes.F16,
                    AttentionLayouts.HeadMajor) with
                {
                    Gqa = true, Masks = CW, LogitTerms = true, DevicePositions = true, CaptureSafe = true,
                }, null,
                static (in AttentionCall c) => true,
                (in AttentionCall c) =>
                {
                    FlashAttentionDeviceDecode(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Positions.KvLength,
                        t_flashKvGroup, c.Mask.Causal, c.Positions.QueryOffset, c.Scale, c.Positions.DeviceHandle, c.Logits.Softcap,
                        c.Mask.LegacySlidingWindow);
                    return true;
                }),
        ];
    }

    /// <summary>The grouped-query token-major contract: Q/output carry <paramref name="heads"/> heads, K/V
    /// <paramref name="kvHeads"/>, which divides it.</summary>
    internal static void ValidateTokenMajorGqaAttentionContract(Tensor output, Tensor query, Tensor key, Tensor value,
        int heads, int kvHeads, int headDim, float scale)
    {
        if (heads <= 0 || kvHeads <= 0 || headDim <= 0 || heads % kvHeads != 0)
            throw new ArgumentException(
                $"Grouped token-major SDPA needs positive heads, kvHeads and headDim with kvHeads dividing heads; got heads={heads}, kvHeads={kvHeads}, headDim={headDim}.");
        // Krea2 hands rank-4 [1, S, H, D] operands and a rank-3 [1, S, H·D] output.
        static bool Fits(Tensor t, int h, int d) => t.Shape.Rank switch
        {
            2 => t.Shape[1] == (long)h * d,
            3 => t.Shape[0] == 1 && t.Shape[2] == (long)h * d,
            4 => t.Shape[0] == 1 && t.Shape[2] == h && t.Shape[3] == d,
            _ => false,
        };
        if (!Fits(output, heads, headDim) || !Fits(query, heads, headDim) || !Fits(key, kvHeads, headDim) || !Fits(value, kvHeads, headDim))
            throw new ArgumentException(
                $"Grouped token-major SDPA requires Q/output with {heads} heads and K/V with {kvHeads} heads of {headDim}; got output={output.Shape}, Q={query.Shape}, K={key.Shape}, V={value.Shape}.");
        if (TokenMajorRows(output) != TokenMajorRows(query) || TokenMajorRows(key) != TokenMajorRows(value))
            throw new ArgumentException($"Grouped token-major SDPA row counts disagree; got output={output.Shape}, Q={query.Shape}, K={key.Shape}, V={value.Shape}.");
        if (!float.IsFinite(scale))
            throw new ArgumentOutOfRangeException(nameof(scale), "SDPA scale must be finite.");
        if (query.DType != DType.F32 && query.DType != DType.F16 && query.DType != DType.BF16)
            throw new NotSupportedException($"SDPA supports F32/F16/BF16 tensors; got {query.DType}.");
        if (output.DType != query.DType || key.DType != query.DType || value.DType != query.DType)
            throw new ArgumentException(
                $"SDPA requires matching output/Q/K/V dtypes; got output={output.DType}, Q={query.DType}, K={key.DType}, V={value.DType}.");
        if (TokenMajorRows(query) <= 0 || TokenMajorRows(key) <= 0)
            throw new ArgumentException($"Grouped token-major SDPA needs at least one query and one key row; got Q={query.Shape}, K={key.Shape}.");
    }
}
