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
        if (EngineKnobs.AttentionCaptureDir.Value is { Length: > 0 } captureDir) CaptureAttentionCall(in call, captureDir);
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
        {
            // Only the cuDNN rows read views, so the route's other rows exclude themselves and the planner refuses what none takes.
            if (call.Route is not (AttentionRoute.Dense or AttentionRoute.TokenMajor or AttentionRoute.TokenMajorGqa) || call.TermsGap() is not null)
                throw new AttentionUnsupportedException($"CUDA attention serves a call its legacy entry point can carry; this one has {gap}.");
            RunViewCall(in call);
            return;
        }
        Tensor o = call.Output.Storage, q = call.Query.Storage, k = call.Key.Storage, v = call.Value.Storage;
        switch (call.Route)
        {
            case AttentionRoute.Dense:
                ScaledDotProductAttention(o, q, k, v, call.Mask.Bias, call.Scale, call.Numerics);
                break;
            case AttentionRoute.TokenMajor:
                ScaledDotProductAttentionTokenMajor(o, q, k, v, call.Mask.Bias, call.Heads, call.QkHeadDim, call.Scale, call.Numerics);
                break;
            case AttentionRoute.TokenMajorGqa:
                ScaledDotProductAttentionTokenMajor(o, q, k, v, call.Mask.Bias, call.Heads, call.KvHeads, call.QkHeadDim, call.Scale,
                    call.Numerics);
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

    /// <summary>A call the legacy entries cannot carry, planned over its route's rows with its own geometry checked first.</summary>
    private void RunViewCall(in AttentionCall call)
    {
        AttentionHostReference.Validate(in call);
        if (!float.IsFinite(call.Scale))
            throw new ArgumentOutOfRangeException(nameof(call), "SDPA scale must be finite.");
        Tensor output = call.Output.Storage;
        if (ReferenceEquals(output, call.Query.Storage) || ReferenceEquals(output, call.Key.Storage) || ReferenceEquals(output, call.Value.Storage))
            throw new ArgumentException("Attention output must not alias an input.", nameof(call));
        RunAttentionRows(RowsFor(call.Route), in call);
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

    // ── head-major ScaledDotProductAttention: S.1-S.16 (S.4, S.6-S.8 retired) ───────────────────────────────────────

    /// <summary>INT8 attention v2 on one route: the u8 PV row, then its F16-PV fallback.</summary>
    private AttentionKernelRow[] Int8V2Rows(AttentionRoute entry, AttentionLayouts layouts, string u8Branch, string f16Branch)
    {
        AttentionKernelCapability cap = Cap("int8.v2", entry, AttentionDTypes.F16 | AttentionDTypes.F32, AttentionDTypes.F16 | AttentionDTypes.F32,
            layouts) with
        {
            StridedViews = true, MinHeadDim = 128, MaxHeadDim = 128, UsesQueryKey = AttentionFormats.Int8, UsesProbValue = AttentionFormats.Int8,
            CaptureSafe = true,
        };
        return
        [
            new DelegateAttentionRow(cap, u8Branch, (in AttentionCall c) => Int8V2Admits(in c, f16Pv: false),
                (in AttentionCall c) => RunInt8AttentionV2(in c, f16Pv: false)),
            new DelegateAttentionRow(cap with { Id = "int8.v2.f16pv", UsesProbValue = AttentionFormats.F16 }, f16Branch,
                (in AttentionCall c) => Int8V2Admits(in c, f16Pv: true), (in AttentionCall c) => RunInt8AttentionV2(in c, f16Pv: true)),
        ];
    }

    private AttentionKernelRow[] BuildDenseRows()
    {
        const AttentionRoute E = AttentionRoute.Dense;
        const AttentionLayouts HM = AttentionLayouts.HeadMajor;
        const AttentionDTypes Half = AttentionDTypes.F16 | AttentionDTypes.BF16;
        return
        [
            .. Int8V2Rows(E, HM, "S.17", "S.18"),
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
            new DelegateAttentionRow(CudnnSdpaRows.Capability("cudnn.f16", E, AttentionDTypes.F16 | AttentionDTypes.BF16, CudnnSdpaRows.AnyView), "S.3",
                (in AttentionCall c) => CudnnSdpaRows.Covers(in c) && CudnnReady(in c),
                (in AttentionCall c) => TryCudnnSdpa(in c)),
            new DelegateAttentionRow(CudnnSdpaRows.Capability("cudnn.f32.sat", E, AttentionDTypes.F32, HM, strided: false) with
                {
                    Saturates = true, Lowering = AttentionLowering.Cast,
                }, "S.5",
                (in AttentionCall c) => c.Query.DType == DType.F32 && CudnnSdpaRows.MaskFits(in c) && CudnnSdpaRows.ViewsFit(in c) && CudnnReady(in c)
                    && (c.Numerics.LegacyAssertedBounded || _sdpaF16ForceOn) && !_sdpaF16Disabled,
                (in AttentionCall c) => TryCudnnSdpa(in c)),
            new DelegateAttentionRow(CudnnSdpaRows.Capability("cudnn.f32.bf16", E, AttentionDTypes.F32, HM, strided: false) with
                {
                    UsesQueryKey = AttentionFormats.Bf16, UsesProbValue = AttentionFormats.Bf16, NarrowsF32To = AttentionFormats.Bf16,
                    Lowering = AttentionLowering.Cast,
                }, "S.16",
                (in AttentionCall c) => c.Query.DType == DType.F32 && CudnnSdpaRows.MaskFits(in c) && CudnnSdpaRows.ViewsFit(in c)
                    && CudnnReady(in c, CudnnSdpa.SdpaDtype.BF16) && Bf16CastFits(in c),
                (in AttentionCall c) => TryCudnnSdpa(in c, CudnnSdpa.SdpaDtype.BF16)),
            new DelegateAttentionRow(Cap("tiled.f32", E, AttentionDTypes.F32, AttentionDTypes.F32, HM) with
                {
                    UsesQueryKey = SdpaF32Formats, UsesProbValue = SdpaF32Formats,
                }, "S.9",
                (in AttentionCall c) => c.Mask.Bias is null && c.Query.DType == DType.F32 && TilesScores(in c),
                (in AttentionCall c) => { SdpaTiledF32(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Scale); return true; }),
            new DelegateAttentionRow(Cap("tiled.f32.keybias", E, AttentionDTypes.F32, AttentionDTypes.F32, HM) with
                {
                    Masks = AnyBias, UsesQueryKey = SdpaF32Formats, UsesProbValue = SdpaF32Formats,
                }, "S.10",
                (in AttentionCall c) => c.Query.DType == DType.F32 && MaskIsKeyOnly(c.Mask.Bias, c.QueryTokens) && TilesScores(in c),
                (in AttentionCall c) =>
                {
                    SdpaTiledF32(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Scale, c.Mask.Bias);
                    return true;
                }),
            new DelegateAttentionRow(Cap("tiled.cast", E, Half, Half, HM) with
                {
                    UsesQueryKey = SdpaF32Formats, UsesProbValue = SdpaF32Formats, Lowering = AttentionLowering.Cast,
                }, "S.11",
                static (in AttentionCall c) => c.Mask.Bias is null && ScoresPastMaterialized(in c) && OneRowTiles(in c),
                (in AttentionCall c) => { SdpaTiledCast(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Scale); return true; }),
            new DelegateAttentionRow(Cap("materialized", E, AnyFloat, AnyFloat, HM) with
                {
                    Masks = AnyBias, UsesQueryKey = SdpaF32Formats | AttentionFormats.F16, UsesProbValue = SdpaF32Formats | AttentionFormats.F16,
                    Lowering = AttentionLowering.MaterializeScores | AttentionLowering.Cast | AttentionLowering.ExpandMask,
                }, null,
                static (in AttentionCall c) => !ScoresPastMaterialized(in c),
                (in AttentionCall c) =>
                {
                    SdpaMaterialized(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Mask.Bias, c.Scale,
                        c.Numerics.LegacyAssertedBounded);
                    return true;
                }),
            new DelegateAttentionRow(Cap("refuse", E, AnyFloat, AnyFloat, HM) with { Masks = AnyBias }, null,
                static (in AttentionCall c) => ScoresPastMaterialized(in c),
                static (in AttentionCall c) => throw new AttentionUnsupportedException(
                    $"No bounded CUDA row serves this {c.Query.DType} head-major attention: its {(long)c.Batch * c.Heads * c.QueryTokens * c.KeyTokens} "
                    + $"score elements are past the 2^32 ({MaterializedMaxScores}) the materialized row addresses, and past that a call runs "
                    + "query-tiled only when it is unmasked or F32 with a key-only mask and one query row across its heads, "
                    + $"{(long)c.Batch * c.Heads * c.KeyTokens} score elements, fits too.")),
        ];
    }

    /// <summary>The format the F32 SDPA GEMMs multiply in beyond F32: TF32 unless <c>numerics.sdpaNoTf32</c>.</summary>
    private AttentionFormats SdpaF32Formats => _sdpaNoTf32 ? AttentionFormats.None : AttentionFormats.Tf32;

    /// <summary>Score elements the materialized row addresses: its softmax forms each row's offset in 32 bits.</summary>
    internal const long MaterializedMaxScores = 1L << 32;

    /// <summary>The B·H·Sq·Skv score matrix is larger than the materialized row addresses.</summary>
    internal static bool ScoresPastMaterialized(in AttentionCall c) =>
        c.KeyTokens > 0 && (long)c.Batch * c.Heads * c.QueryTokens > MaterializedMaxScores / c.KeyTokens;

    /// <summary>A one-row tile, B·H·Skv scores, is within what the softmax addresses; every tiled row needs it.</summary>
    internal static bool OneRowTiles(in AttentionCall c) => (long)c.Batch * c.Heads * c.KeyTokens <= MaterializedMaxScores;

    /// <summary>The query-tiled gate: a one-row tile the softmax addresses, and forced, an F32 score matrix larger than half of free
    /// VRAM, or one the materialized row does not address.</summary>
    private bool TilesScores(in AttentionCall c)
    {
        if (!OneRowTiles(in c)) return false;
        if (EngineKnobs.SdpaForceTiled.Value || ScoresPastMaterialized(in c)) return true;
        ulong scoreBytes = (ulong)((long)c.Batch * c.Heads) * (ulong)c.QueryTokens * (ulong)c.KeyTokens * sizeof(float);
        return scoreBytes > (ulong)FreeVram() / 2;
    }

    /// <summary>The BF16 copies of Q, K, V and the output fit in half of free VRAM.</summary>
    private bool Bf16CastFits(in AttentionCall c)
    {
        ulong castBytes = sizeof(ushort) * ((ulong)c.Query.Storage.ElementCount + (ulong)c.Key.Storage.ElementCount
            + (ulong)c.Value.Storage.ElementCount + (ulong)c.Output.Storage.ElementCount);
        return castBytes <= (ulong)FreeVram() / 2;
    }

    /// <summary>Free device memory the attention rows budget against.</summary>
    private nuint FreeVram() => TestFreeVramOverride?.Invoke() ?? _context.GetMemoryInfo().freeBytes;

    // ── token-major ScaledDotProductAttentionTokenMajor: M.1-M.3, and the grouped-query lowering ────────────────────

    private AttentionKernelRow[] BuildTokenMajorRows()
    {
        const AttentionRoute E = AttentionRoute.TokenMajor;
        const AttentionLayouts TM = AttentionLayouts.TokenMajor;
        return
        [
            .. Int8V2Rows(E, TM, "M.4", "M.5"),
            new DelegateAttentionRow(Cap("flash.f16", E, AttentionDTypes.F16, AttentionDTypes.F16, TM) with
                {
                    MaxBatch = 1, MinHeadDim = 256, MaxHeadDim = 256, UsesQueryKey = AttentionFormats.F16, UsesProbValue = AttentionFormats.F16, CaptureSafe = true,
                }, "M.1",
                (in AttentionCall c) => FlashF16Eligible(c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Output.Storage, c.Mask.Bias, c.QkHeadDim),
                (in AttentionCall c) =>
                {
                    CudaKernels.FlashStrides tokenMajor = new(0, (ulong)c.QkHeadDim, (ulong)c.Heads * (ulong)c.QkHeadDim);
                    return TryFlashF16(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Scale, 1, c.Heads, c.QueryTokens,
                        c.KeyTokens, c.QkHeadDim, tokenMajor, tokenMajor, tokenMajor, tokenMajor);
                }),
            new DelegateAttentionRow(CudnnSdpaRows.Capability("cudnn.tokenmajor", E,
                    AttentionDTypes.F16 | AttentionDTypes.BF16 | AttentionDTypes.F32, CudnnSdpaRows.AnyView) with { Saturates = true }, "M.2",
                // The Sage detour is the permute row's; a view or batch that row cannot take stays on cuDNN.
                (in AttentionCall c) => !(SageTokenMajorDetour && WholeSequence(in c) && SageF16Preferred(c.Query.Storage, c.Key.Storage, c.Value.Storage,
                        c.Output.Storage, c.Mask.Bias, c.QueryTokens, c.KeyTokens, c.QkHeadDim))
                    && CudnnSdpaRows.MaskFits(in c) && CudnnSdpaRows.ViewsFit(in c) && CudnnReady(in c)
                    && (CudnnSdpaRows.HalfOperands(in c)
                        || (c.Query.DType == DType.F32 && (c.Numerics.LegacyAssertedBounded || _sdpaF16ForceOn) && !_sdpaF16Disabled)),
                (in AttentionCall c) => RunCudnnTokenMajor(in c)),
            new DelegateAttentionRow(Cap("permute.dense", E, AnyFloat, AnyFloat, TM) with
                {
                    MaxBatch = 1, Masks = AnyBias, Lowering = AttentionLowering.Permute,
                }, null,
                static (in AttentionCall c) => c.Mask.Bias is not { } mask || TokenMajorMaskFits(mask, c.Heads, c.QueryTokens, c.KeyTokens),
                (in AttentionCall c) => { TokenMajorViaHeadMajor(in c); return true; }),
        ];
    }

    private AttentionKernelRow[] BuildTokenMajorGqaRows() =>
    [
        // Grouped fp16/bf16 only; yields to INT8 v2, Sage and the F16 flash kernel, which take these calls after widening.
        new DelegateAttentionRow(CudnnSdpaRows.Capability("cudnn.tokenmajor.gqa", AttentionRoute.TokenMajorGqa,
                AttentionDTypes.F16 | AttentionDTypes.BF16, CudnnSdpaRows.AnyView, gqa: true), "G.1",
            (in AttentionCall c) => c.KvHeads < c.Heads && CudnnSdpaRows.Covers(in c)
                && (!WholeSequence(in c)
                    || (!Int8V2TakesWidened(in c)
                        && !SageF16Preferred(c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Output.Storage, c.Mask.Bias, c.QueryTokens,
                            c.KeyTokens, c.QkHeadDim)
                        && !FlashF16Eligible(c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Output.Storage, c.Mask.Bias, c.QkHeadDim)))
                && CudnnReady(in c),
            (in AttentionCall c) => RunCudnnTokenMajor(in c)),
        new DelegateAttentionRow(Cap("permute.widen.dense", AttentionRoute.TokenMajorGqa, AnyFloat, AnyFloat, AttentionLayouts.TokenMajor) with
            {
                MaxBatch = 1, Gqa = true, Masks = AnyBias, Lowering = AttentionLowering.Permute | AttentionLowering.WidenKv,
            }, null,
            // It runs the head-major entry, which takes the masks that entry takes.
            static (in AttentionCall c) => c.Mask.Bias is not { } mask || HeadMajorMaskFits(mask, c.Batch, c.Heads, c.QueryTokens, c.KeyTokens),
            (in AttentionCall c) => { GroupedTokenMajorViaHeadMajor(in c); return true; }),
    ];

    /// <summary>The knob, the session, the head dim and the health of the signature the call runs under, F32 operands cast to
    /// <paramref name="castTarget"/>.</summary>
    private bool CudnnReady(in AttentionCall c, CudnnSdpa.SdpaDtype castTarget = CudnnSdpa.SdpaDtype.F16) =>
        _sdpaCudnn && !_cudnnSdpaDead && CudnnSdpa.ShapeSupported(c.QkHeadDim)
        && _cudnnSdpaHealth.Eligible(CudnnSdpaSignature.Of(in c, castTarget), Environment.TickCount64);

    /// <summary>One sequence of whole tensors: all the permuting and widening rows take.</summary>
    private static bool WholeSequence(in AttentionCall c) =>
        c.Batch == 1 && c.Query.IsPlain && c.Key.IsPlain && c.Value.IsPlain && c.Output.IsPlain;

    /// <summary>The token-major launch shared by the equal-head and grouped rows.</summary>
    private bool RunCudnnTokenMajor(in AttentionCall c)
    {
        using OpScope _op = EnterOp(nameof(ScaledDotProductAttentionTokenMajor));
        EnsureKernels();
        return TryCudnnSdpa(in c);
    }

    /// <summary>The masks the permuting token-major row has always taken: F32, one block per head or shared, one row per query.</summary>
    private static bool TokenMajorMaskFits(Tensor mask, long heads, long sq, long skv) => mask.DType == DType.F32 && mask.Shape.Rank switch
    {
        2 => mask.Shape[0] == sq && mask.Shape[1] == skv,
        3 => (mask.Shape[0] == 1 || mask.Shape[0] == heads) && mask.Shape[1] == sq && mask.Shape[2] == skv,
        4 => mask.Shape[0] == 1 && (mask.Shape[1] == 1 || mask.Shape[1] == heads) && mask.Shape[2] == sq && mask.Shape[3] == skv,
        _ => false,
    };

    /// <summary>What <see cref="AttentionMaskLayout.Of"/> accepts, without throwing.</summary>
    private static bool HeadMajorMaskFits(Tensor mask, long batch, long heads, long sq, long skv) =>
        mask.DType == DType.F32 && mask.ElementCount <= int.MaxValue && sq * skv <= int.MaxValue
        && AttentionMaskLayout.Broadcasts(mask.Shape, batch, heads, sq, skv);

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
            ScaledDotProductAttention(oMh, qMh, kRep ?? kMh, vRep ?? vMh, c.Mask.Bias, c.Scale, c.Numerics);
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
            ScaledDotProductAttention(oMh, qMh, kMh, vMh, c.Mask.Bias, c.Scale, c.Numerics);
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
                    Lowering = AttentionLowering.Cast | AttentionLowering.MaterializeMask,
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
                        && _sdpaCudnn && !_cudnnSdpaDead && CudnnSdpa.ShapeSupported(d)
                        && _cudnnSdpaHealth.Eligible(CudnnSdpaSignature.For(DType.F32, c.Heads, c.KvHeads, d, new CudnnBias(1, 1, c.QueryTokens)),
                            Environment.TickCount64)
                        && CausalMaskFits(c.QueryTokens, kvLen, c.Heads, c.KvHeads, d);
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
