using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>What keys a cuDNN SDPA plan, what a failure is remembered against, and which calls a row's declaration lets through; no device.</summary>
public sealed class CudnnSdpaPolicyTests
{
    private const long H = 8, Sq = 64, Sk = 96, D = 128;

    private static CudnnSdpa.PlanKey Key(long hKv = H, CudnnSdpa.SdpaDtype dtype = CudnnSdpa.SdpaDtype.F16,
        bool tokenMajor = false, long biasB = 0, long biasH = 0, long biasSq = 0)
    {
        Func<long, long, long, long, CudnnSdpa.View> view = tokenMajor ? CudnnSdpa.View.TokenMajor : CudnnSdpa.View.HeadMajor;
        return new(dtype, D, view(2, H, Sq, D), view(2, hKv, Sk, D), view(2, hKv, Sk, D), view(2, H, Sq, D),
            CudnnSdpa.View.Bias(new CudnnBias(biasB, biasH, biasSq), Sk));
    }

    // ── plan key ──────────────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void PlanKey_HasNoScale_SoOnePlanServesEveryScale()
    {
        Assert.DoesNotContain(typeof(CudnnSdpa.PlanKey).GetProperties(), p => p.Name.Contains("Scale", StringComparison.OrdinalIgnoreCase));
        Assert.Equal(Key(), Key());
        Assert.Equal(Key().GetHashCode(), Key().GetHashCode());
    }

    [Fact]
    public void PlanKey_DistinguishesKvHeads()
    {
        Assert.NotEqual(Key(hKv: H), Key(hKv: 2));
        Assert.NotEqual(Key(hKv: 2), Key(hKv: 4));
    }

    [Fact]
    public void PlanKey_DistinguishesDtypeAndLayout()
    {
        Assert.NotEqual(Key(), Key(dtype: CudnnSdpa.SdpaDtype.BF16));
        Assert.NotEqual(Key(), Key(tokenMajor: true));
    }

    [Theory]
    [InlineData(1, 1, 1, 1, 1, Sq)]    // key-only against per-query
    [InlineData(1, 1, Sq, 1, H, Sq)]   // head-shared against per-head
    [InlineData(1, 1, Sq, 2, 1, Sq)]   // head-shared against per-batch
    [InlineData(1, H, Sq, 2, H, Sq)]   // per-head against per-batch-and-head
    public void PlanKey_DistinguishesBiasDims(long aB, long aH, long aSq, long bB, long bH, long bSq)
    {
        Assert.NotEqual(Key(biasB: aB, biasH: aH, biasSq: aSq), Key(biasB: bB, biasH: bH, biasSq: bSq));
    }

    [Fact]
    public void PlanKey_HasBiasOnlyWhenBiasRowsAreGiven()
    {
        Assert.False(Key().HasBias);
        Assert.True(Key(biasB: 1, biasH: 1, biasSq: 1).HasBias);
        Assert.NotEqual(Key(), Key(biasB: 1, biasH: 1, biasSq: 1));
    }

    [Fact]
    public void PlanKey_Validate_RefusesWhatCudnnCouldNotBroadcast()
    {
        Key().Validate();
        Key(hKv: 2, biasB: 2, biasH: H, biasSq: Sq).Validate();
        Assert.Throws<ArgumentException>(() => Key(hKv: 3).Validate());            // 8 query heads over 3 K/V heads
        Assert.Throws<ArgumentException>(() => Key(biasB: 1, biasH: 3, biasSq: 1).Validate());
        Assert.Throws<ArgumentException>(() => Key(biasB: 3, biasH: 1, biasSq: 1).Validate());
        Assert.Throws<ArgumentException>(() => Key(biasB: 1, biasH: 1, biasSq: 5).Validate());
    }

    [Fact]
    public void PlanKey_CudnnDtype_FollowsTheIoDtype()
    {
        Assert.Equal(CudnnApi.CUDNN_DATA_HALF, Key().CudnnDtype);
        Assert.Equal(CudnnApi.CUDNN_DATA_BFLOAT16, Key(dtype: CudnnSdpa.SdpaDtype.BF16).CudnnDtype);
    }

    // ── bias dims ─────────────────────────────────────────────────────────────────────────────────────────────

    private static Tensor Mask(params long[] dims) => new(new TensorShape(dims), DType.F32);

    [Theory]
    [InlineData(new long[] { Sq, Sk }, 1, 1, Sq)]
    [InlineData(new long[] { 1, Sk }, 1, 1, 1)]
    [InlineData(new long[] { H, Sq, Sk }, 1, H, Sq)]
    [InlineData(new long[] { 1, 1, Sq, Sk }, 1, 1, Sq)]
    [InlineData(new long[] { 1, 1, 1, Sk }, 1, 1, 1)]
    [InlineData(new long[] { 1, H, Sq, Sk }, 1, H, Sq)]
    [InlineData(new long[] { 2, 1, Sq, Sk }, 2, 1, Sq)]
    [InlineData(new long[] { 2, H, 1, Sk }, 2, H, 1)]
    public void Bias_ReadsTheBroadcastDims(long[] shape, long b, long h, long sq)
    {
        using Tensor mask = Mask(shape);
        Assert.True(CudnnBias.TryFrom(mask, 2, H, Sq, Sk, out CudnnBias bias));
        Assert.Equal(new CudnnBias(b, h, sq), bias);
        Assert.True(bias.HasBias);
    }

    [Theory]
    [InlineData(new long[] { Sq, Sk + 1 })]
    [InlineData(new long[] { 3, Sq, Sk })]          // 3 heads against 8
    [InlineData(new long[] { 1, 3, Sq, Sk })]
    [InlineData(new long[] { 3, 1, Sq, Sk })]       // 3 batch items against 2
    [InlineData(new long[] { 1, 1, Sq - 1, Sk })]
    [InlineData(new long[] { Sk })]
    [InlineData(new long[] { 1, 1, 1, 1, Sk })]
    public void Bias_RefusesAMaskTheGraphCannotBroadcast(long[] shape)
    {
        using Tensor mask = Mask(shape);
        Assert.False(CudnnBias.TryFrom(mask, 2, H, Sq, Sk, out _));
    }

    /// <summary>A per-head bias needs 16-byte rows (the graph declares that alignment); head-shared biases keep their route.</summary>
    [Fact]
    public void Bias_PerHeadNeedsSixteenByteRows_HeadSharedDoesNot()
    {
        const long oddKeys = 77;
        using Tensor perHead = Mask([1, H, Sq, oddKeys]), shared = Mask([1, 1, Sq, oddKeys]), aligned = Mask([1, H, Sq, 80]);
        Assert.False(CudnnBias.TryFrom(perHead, 1, H, Sq, oddKeys, out _));
        Assert.True(CudnnBias.TryFrom(shared, 1, H, Sq, oddKeys, out _));
        Assert.True(CudnnBias.TryFrom(aligned, 1, H, Sq, 80, out _));
    }

    [Fact]
    public void Bias_NoMaskIsNoBias_AndANonF32MaskIsRefused()
    {
        Assert.True(CudnnBias.TryFrom(null, 2, H, Sq, Sk, out CudnnBias none));
        Assert.False(none.HasBias);
        using Tensor f16 = new(new TensorShape(1, 1, Sq, Sk), DType.F16);
        Assert.False(CudnnBias.TryFrom(f16, 2, H, Sq, Sk, out _));
    }

    // ── capability signature ──────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Signature_SeparatesDtypeGqaMaskBiasAxesAndHeadDim()
    {
        CudnnSdpaSignature plain = CudnnSdpaSignature.For(DType.F16, 16, 16, 128, CudnnBias.None);
        Assert.Equal(plain, CudnnSdpaSignature.For(DType.F16, 48, 48, 128, CudnnBias.None));      // head counts are not part of it
        Assert.NotEqual(plain, CudnnSdpaSignature.For(DType.BF16, 16, 16, 128, CudnnBias.None));
        Assert.NotEqual(plain, CudnnSdpaSignature.For(DType.F16, 16, 8, 128, CudnnBias.None));
        Assert.NotEqual(plain, CudnnSdpaSignature.For(DType.F16, 16, 16, 64, CudnnBias.None));
        Assert.NotEqual(plain, CudnnSdpaSignature.For(DType.F16, 16, 16, 128, new CudnnBias(1, 1, 1)));
        CudnnSdpaSignature shared = CudnnSdpaSignature.For(DType.F16, 16, 16, 128, new CudnnBias(1, 1, 64));
        Assert.NotEqual(shared, CudnnSdpaSignature.For(DType.F16, 16, 16, 128, new CudnnBias(1, 16, 64)));
        Assert.NotEqual(shared, CudnnSdpaSignature.For(DType.F16, 16, 16, 128, new CudnnBias(2, 1, 64)));
        Assert.NotEqual(CudnnSdpaSignature.For(DType.F16, 16, 16, 128, new CudnnBias(1, 1, 1)), shared);   // key-only against per-query
    }

    [Fact]
    public void Signature_RunsF32InF16_AndReadsBiasAxesFromExtents()
    {
        Assert.Equal(CudnnSdpaSignature.For(DType.F16, 4, 4, 64, CudnnBias.None), CudnnSdpaSignature.For(DType.F32, 4, 4, 64, CudnnBias.None));
        CudnnSdpaSignature s = CudnnSdpaSignature.For(DType.F16, 4, 4, 64, new CudnnBias(2, 4, 1));
        Assert.Equal(CudnnSdpaMask.Bias, s.Mask);
        Assert.Equal(CudnnBiasAxes.Batch | CudnnBiasAxes.Head, s.BiasAxes);
        Assert.Equal(CudnnBiasAxes.None, CudnnSdpaSignature.For(DType.F16, 4, 4, 64, new CudnnBias(1, 1, 1)).BiasAxes);
    }

    [Fact]
    public void Signature_OfACall_ReadsItsOperandsAndMask()
    {
        using Tensor mask = Mask(1, 48, 16, 16);
        CudnnSdpaSignature s = CudnnSdpaSignature.Of(Call(DType.BF16, 48, 12, mask: mask));
        Assert.Equal(new CudnnSdpaSignature(CudnnSdpa.SdpaDtype.BF16, true, CudnnSdpaMask.Bias, CudnnBiasAxes.Head | CudnnBiasAxes.Query, 128), s);
    }

    // ── failure state keyed by signature ──────────────────────────────────────────────────────────────────────

    private static readonly CudnnSdpaSignature Fp16 = CudnnSdpaSignature.For(DType.F16, 16, 16, 128, CudnnBias.None);
    private static readonly CudnnSdpaSignature Bf16 = CudnnSdpaSignature.For(DType.BF16, 16, 16, 128, CudnnBias.None);
    private static readonly CudnnSdpaSignature Gqa = CudnnSdpaSignature.For(DType.F16, 16, 8, 128, CudnnBias.None);
    private static readonly CudnnSdpaSignature Biased = CudnnSdpaSignature.For(DType.F16, 16, 16, 128, new CudnnBias(1, 16, 64));

    private static CudnnSdpaShape Shape(CudnnSdpaSignature s, long sq) =>
        new(s, DType.F16, new(CudnnSdpa.SdpaDtype.F16, 128, CudnnSdpa.View.HeadMajor(1, 16, sq, 128), CudnnSdpa.View.HeadMajor(1, 16, sq, 128),
            CudnnSdpa.View.HeadMajor(1, 16, sq, 128), CudnnSdpa.View.HeadMajor(1, 16, sq, 128)));

    [Fact]
    public void RefusedBf16Plans_NeverDisableTheFp16PathAtTheSameHeadDim()
    {
        CudnnSdpaHealth health = new();
        for (int i = 0; i < CudnnSdpaHealth.RefusalsToDisable + 2; i++)
            health.RecordFailure(Shape(Bf16, 64 + 8 * i), permanent: true, nowTicks: 0);
        Assert.False(health.Eligible(Bf16, 0));
        Assert.True(health.Eligible(Fp16, 0));
        Assert.True(health.Eligible(Gqa, 0));
        Assert.True(health.Eligible(Biased, 0));
    }

    [Fact]
    public void RefusedGqaAndBiasedPlans_AreIsolatedFromEachOtherAndFromPlain()
    {
        CudnnSdpaHealth health = new();
        for (int i = 0; i < CudnnSdpaHealth.RefusalsToDisable; i++)
            health.RecordFailure(Shape(Gqa, 64 + 8 * i), permanent: true, nowTicks: 0);
        Assert.False(health.Eligible(Gqa, 0));
        Assert.True(health.Eligible(Fp16, 0));
        Assert.True(health.Eligible(Biased, 0));
        for (int i = 0; i < CudnnSdpaHealth.RefusalsToDisable; i++)
            health.RecordFailure(Shape(Biased, 64 + 8 * i), permanent: true, nowTicks: 0);
        Assert.False(health.Eligible(Biased, 0));
        Assert.True(health.Eligible(Fp16, 0));
    }

    [Fact]
    public void AStructuralRefusal_HoldsForItsShapeOnly_AndTheSignatureGoesAfterThreeWithNoSuccess()
    {
        CudnnSdpaHealth health = new();
        CudnnSdpaHealth.FailureOutcome first = health.RecordFailure(Shape(Fp16, 64), permanent: true, nowTicks: 0);
        Assert.True(first.NewRefusal);
        Assert.True(first.Warn);
        Assert.False(first.SignatureDisabled);
        Assert.True(health.Refused(Shape(Fp16, 64)));
        Assert.False(health.Refused(Shape(Fp16, 72)));
        Assert.True(health.Eligible(Fp16, 0));
        Assert.False(health.RecordFailure(Shape(Fp16, 64), permanent: true, nowTicks: 0).NewRefusal);
        health.RecordFailure(Shape(Fp16, 72), permanent: true, nowTicks: 0);
        Assert.True(health.RecordFailure(Shape(Fp16, 80), permanent: true, nowTicks: 0).SignatureDisabled);
        Assert.False(health.Eligible(Fp16, long.MaxValue));
    }

    [Fact]
    public void ASignatureThatRan_IsNeverDisabledByRefusals()
    {
        CudnnSdpaHealth health = new();
        Assert.Equal(0, health.RecordSuccess(Fp16));
        for (int i = 0; i < 10; i++)
            Assert.False(health.RecordFailure(Shape(Fp16, 64 + 8 * i), permanent: true, nowTicks: 0).SignatureDisabled);
        Assert.True(health.Eligible(Fp16, 0));
        Assert.Equal(10, health.RefusedShapeCount);
    }

    [Fact]
    public void ATransientFailure_BacksOffPerSignature_ThenRecovers()
    {
        CudnnSdpaHealth health = new();
        CudnnSdpaHealth.FailureOutcome one = health.RecordFailure(Shape(Bf16, 64), permanent: false, nowTicks: 1_000);
        Assert.Equal(1, one.ConsecutiveFailures);
        Assert.Equal(2000, one.BackoffMs);
        Assert.False(health.Eligible(Bf16, 2_999));
        Assert.True(health.Eligible(Bf16, 3_000));
        Assert.True(health.Eligible(Fp16, 1_000));
        Assert.Equal(4000, health.RecordFailure(Shape(Bf16, 64), permanent: false, nowTicks: 3_000).BackoffMs);
        Assert.Equal(2, health.RecordSuccess(Bf16));
        Assert.True(health.Eligible(Bf16, 3_001));
        Assert.Empty(health.Diagnostics);
    }

    [Theory]
    [InlineData(1, 2_000L)]
    [InlineData(2, 4_000L)]
    [InlineData(9, 300_000L)]
    [InlineData(40, 300_000L)]
    public void Backoff_DoublesToACap(int failures, long expectedMs) => Assert.Equal(expectedMs, CudnnSdpaHealth.BackoffMs(failures));

    [Fact]
    public void Diagnostics_FoldPerHeadDim_WithoutLosingASignature()
    {
        CudnnSdpaHealth health = new();
        health.RecordFailure(Shape(Bf16, 64), permanent: false, nowTicks: Environment.TickCount64);
        health.RecordFailure(Shape(Bf16, 64), permanent: false, nowTicks: Environment.TickCount64);
        health.RecordFailure(Shape(Fp16, 64), permanent: false, nowTicks: Environment.TickCount64);
        Assert.Equal(2, health.Diagnostics.Count);
        Assert.Equal(2, health.Diagnostics[Bf16].ConsecutiveFailures);
        (int failures, bool permanent, _) = health.DimDiagnostics[128];
        Assert.Equal(2, failures);
        Assert.False(permanent);
    }

    // ── row declaration and call-only admission ───────────────────────────────────────────────────────────────

    private static AttentionCall Call(DType dtype, int hq, int hkv, int sq = 16, int skv = 16, int d = 128, Tensor? mask = null,
        AttentionNumerics numerics = default, AttentionRoute route = AttentionRoute.Dense, bool tokenMajor = false)
    {
        Tensor q = new(new TensorShape(1, hq, sq, d), dtype), k = new(new TensorShape(1, hkv, skv, d), dtype);
        Tensor v = new(new TensorShape(1, hkv, skv, d), dtype), o = new(new TensorShape(1, hq, sq, d), dtype);
        return new()
        {
            Query = tokenMajor ? AttentionOperand.TokenMajor(q, hq, d) : AttentionOperand.HeadMajor(q),
            Key = tokenMajor ? AttentionOperand.TokenMajor(k, hkv, d) : AttentionOperand.HeadMajor(k),
            Value = tokenMajor ? AttentionOperand.TokenMajor(v, hkv, d) : AttentionOperand.HeadMajor(v),
            Output = tokenMajor ? AttentionOperand.TokenMajor(o, hq, d) : AttentionOperand.HeadMajor(o),
            Scale = 0.125f, Mask = AttentionMask.Additive(mask), Numerics = numerics, Route = route,
        };
    }

    private static readonly AttentionNumerics GrantsBf16 = new() { QueryKey = AttentionFormats.Bf16, ProbValue = AttentionFormats.Bf16 };

    private static readonly AttentionKernelCapability HalfRow = CudnnSdpaRows.Capability(
        "test.half", AttentionRoute.Dense, AttentionDTypes.F16 | AttentionDTypes.BF16, AttentionLayouts.HeadMajor);

    /// <summary>A row's call-only verdict: its declared limits, then the shared rules.</summary>
    private static bool Reaches(AttentionKernelCapability row, in AttentionCall c) =>
        row.Excludes(in c) == AttentionExclusion.None && CudnnSdpaRows.Covers(in c);

    [Fact]
    public void Capability_DeclaresPerHeadBiasAndBf16_AndGqaOnlyWhenAsked()
    {
        Assert.False(HalfRow.Gqa);
        Assert.True(CudnnSdpaRows.Capability("test.g", AttentionRoute.TokenMajorGqa, AttentionDTypes.F16, AttentionLayouts.TokenMajor, gqa: true).Gqa);
        Assert.Equal(AttentionMaskSupport.HeadSharedBias | AttentionMaskSupport.PerHeadBias, HalfRow.Masks);
        Assert.Equal(AttentionFormats.F16 | AttentionFormats.Bf16, HalfRow.UsesQueryKey);
        Assert.Equal(AttentionFormats.F16 | AttentionFormats.Bf16, HalfRow.UsesProbValue);
        AttentionKernelCapability f32 = CudnnSdpaRows.Capability("test.f32", AttentionRoute.Dense, AttentionDTypes.F32, AttentionLayouts.HeadMajor);
        Assert.False(f32.Gqa);
        Assert.Equal(AttentionFormats.F16, f32.UsesQueryKey);
        Assert.Equal(32, HalfRow.MinHeadDim);
        Assert.Equal(256, HalfRow.MaxHeadDim);
    }

    [Fact]
    public void AGroupedCall_ReachesTheRowOnlyWhenItDeclaresGqa()
    {
        AttentionCall grouped = Call(DType.F16, 48, 12);
        Assert.Equal(AttentionExclusion.GroupedKv, HalfRow.Excludes(in grouped));
        Assert.False(Reaches(HalfRow, in grouped));
        AttentionKernelCapability gqaRow = HalfRow with { Gqa = true };
        Assert.True(Reaches(gqaRow, in grouped));
        AttentionCall mha = Call(DType.F16, 48, 48);
        Assert.True(Reaches(HalfRow, in mha));
    }

    [Fact]
    public void ABf16Call_ReachesTheRowOnlyWithTheGrant_AndOnlyAsAllBf16()
    {
        AttentionCall plain = Call(DType.BF16, 8, 8);
        Assert.Equal(AttentionExclusion.None, HalfRow.Excludes(in plain));
        Assert.False(CudnnSdpaRows.HalfOperands(in plain));
        Assert.False(Reaches(HalfRow, in plain));

        AttentionCall granted = plain with { Numerics = GrantsBf16 };
        Assert.True(Reaches(HalfRow, in granted));
        Assert.False(Reaches(HalfRow, plain with { Numerics = new() { QueryKey = AttentionFormats.Bf16 } }));      // QK alone is not PV
        Assert.False(Reaches(HalfRow, plain with { Numerics = AttentionNumerics.FromLegacy(true) }));               // the legacy flag grants F16 only

        AttentionCall mixed = granted with { Output = AttentionOperand.HeadMajor(new Tensor(new TensorShape(1, 8, 16, 128), DType.F16)) };
        Assert.False(CudnnSdpaRows.HalfOperands(in mixed));

        AttentionKernelCapability f16Only = CudnnSdpaRows.Capability("test.f16", AttentionRoute.Dense, AttentionDTypes.F16, AttentionLayouts.HeadMajor);
        Assert.Equal(AttentionExclusion.QueryType, f16Only.Excludes(in granted));
        Assert.Equal(AttentionFormats.F16, f16Only.UsesQueryKey);
    }

    [Fact]
    public void AnF16Call_NeedsNoGrant_AndF32IsNotHalf()
    {
        AttentionCall f16 = Call(DType.F16, 8, 8);
        Assert.True(Reaches(HalfRow, in f16));
        AttentionCall f32 = Call(DType.F32, 8, 8);
        Assert.False(CudnnSdpaRows.HalfOperands(in f32));
        Assert.Equal(AttentionExclusion.QueryType, HalfRow.Excludes(in f32));
    }

    [Fact]
    public void APerHeadBias_ReachesTheRowOnlyWhenItDeclaresIt_AndMatchingHeads()
    {
        using Tensor perHead = Mask(1, 8, 16, 16);
        AttentionCall call = Call(DType.F16, 8, 8, mask: perHead);
        Assert.True(Reaches(HalfRow, in call));
        AttentionKernelCapability sharedOnly = HalfRow with { Masks = AttentionMaskSupport.HeadSharedBias };
        Assert.Equal(AttentionExclusion.PerHeadBias, sharedOnly.Excludes(in call));
        Assert.False(Reaches(sharedOnly, in call));

        using Tensor wrongHeads = Mask(1, 4, 16, 16);
        AttentionCall mismatch = Call(DType.F16, 8, 8, mask: wrongHeads);
        Assert.False(CudnnSdpaRows.MaskFits(in mismatch));
        Assert.False(Reaches(HalfRow, in mismatch));
    }

    [Fact]
    public void SharedPerBatchAndKeyOnlyBiases_AreCarried()
    {
        foreach (long[] shape in new[] { new long[] { 1, 1, 16, 16 }, new long[] { 1, 1, 1, 16 }, new long[] { 16, 16 }, new long[] { 1, 16 } })
        {
            using Tensor mask = Mask(shape);
            AttentionCall call = Call(DType.F16, 8, 8, mask: mask);
            Assert.True(Reaches(HalfRow, in call), string.Join("x", shape));
        }
        using Tensor perBatchMask = Mask(2, 1, 16, 16);
        AttentionCall batched = Call(DType.F16, 8, 8) with
        {
            Query = AttentionOperand.HeadMajor(new Tensor(new TensorShape(2, 8, 16, 128), DType.F16)),
            Key = AttentionOperand.HeadMajor(new Tensor(new TensorShape(2, 8, 16, 128), DType.F16)),
            Value = AttentionOperand.HeadMajor(new Tensor(new TensorShape(2, 8, 16, 128), DType.F16)),
            Output = AttentionOperand.HeadMajor(new Tensor(new TensorShape(2, 8, 16, 128), DType.F16)),
            Mask = AttentionMask.Additive(perBatchMask),
        };
        Assert.True(Reaches(HalfRow, in batched));
        Assert.True(CudnnSdpaRows.MaskFits(in batched));
    }

    [Fact]
    public void ARowReachesOnlyItsOwnEntry()
    {
        AttentionKernelCapability gqaRow = CudnnSdpaRows.Capability("test.gqa", AttentionRoute.TokenMajorGqa, AttentionDTypes.F16 | AttentionDTypes.BF16,
            AttentionLayouts.TokenMajor, gqa: true);
        AttentionCall krea2 = Call(DType.F16, 48, 12, d: 128, route: AttentionRoute.TokenMajorGqa, tokenMajor: true);
        Assert.Equal(AttentionExclusion.None, gqaRow.Excludes(in krea2));
        Assert.Equal(AttentionExclusion.Entry, HalfRow.Excludes(in krea2));
        Assert.Equal(AttentionExclusion.Layout, HalfRow.Excludes(krea2 with { Route = AttentionRoute.Dense }));
    }

    [Theory]
    [InlineData(24, AttentionExclusion.HeadDim)]
    [InlineData(32, AttentionExclusion.None)]
    [InlineData(40, AttentionExclusion.None)]
    [InlineData(64, AttentionExclusion.None)]
    [InlineData(256, AttentionExclusion.None)]
    [InlineData(264, AttentionExclusion.HeadDim)]
    public void TheRowsKeepTheirDeclaredHeadDimEnvelope(int d, AttentionExclusion expected)
    {
        AttentionCall call = Call(DType.F16, 8, 8, d: d);
        Assert.Equal(expected, HalfRow.Excludes(in call));
    }
}
