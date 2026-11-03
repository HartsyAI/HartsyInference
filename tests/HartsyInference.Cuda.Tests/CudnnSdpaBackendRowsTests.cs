using System.Reflection;
using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>The real CUDA rows planned with no device, on a backend that never ran its constructor; calls stay small so no earlier row loads a kernel module.</summary>
public sealed class CudnnSdpaBackendRowsTests
{
    private const int S = 16, D = 128;

    private static readonly AttentionNumerics GrantsBf16 = new() { QueryKey = AttentionFormats.Bf16, ProbValue = AttentionFormats.Bf16 };

    internal static void Set(CudaBackend backend, string field, object value) =>
        (typeof(CudaBackend).GetField(field, BindingFlags.Instance | BindingFlags.NonPublic) ?? throw new MissingFieldException(field)).SetValue(backend, value);

    internal static CudaBackend Backend(bool cudnn = true, bool dead = false, CudnnSdpaHealth? health = null, ulong freeVram = 64UL << 30)
    {
        CudaBackend backend = (CudaBackend)RuntimeHelpers.GetUninitializedObject(typeof(CudaBackend));
        GC.SuppressFinalize(backend);   // the finalizer would tear down state the constructor never built
        Set(backend, "_sdpaCudnn", cudnn);
        Set(backend, "_cudnnSdpaDead", dead);
        Set(backend, "_cudnnSdpaHealth", health ?? new CudnnSdpaHealth());
        backend.TestFreeVramOverride = () => (nuint)freeVram;
        return backend;
    }

    internal static AttentionKernelRow[] Rows(CudaBackend backend, string builder) =>
        (AttentionKernelRow[])(typeof(CudaBackend).GetMethod(builder, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new MissingMethodException(builder)).Invoke(backend, null)!;

    private static AttentionCall TokenMajor(DType dt, int hq, int hkv, AttentionRoute? route = null, AttentionNumerics? numerics = null)
    {
        Tensor q = new(new TensorShape(1, S, hq, D), dt), k = new(new TensorShape(1, S, hkv, D), dt);
        Tensor v = new(new TensorShape(1, S, hkv, D), dt), o = new(new TensorShape(1, S, hq, D), dt);
        AttentionCall call = AttentionCall.FromTokenMajor(o, q, k, v, null, hq, hkv, D, 0.125f, true);
        return call with { Route = route ?? call.Route, Numerics = numerics ?? call.Numerics };
    }

    private static AttentionCall HeadMajor(DType dt, int hq, int hkv, long biasHeads = 0, AttentionNumerics? numerics = null)
    {
        Tensor q = new(new TensorShape(1, hq, S, D), dt), k = new(new TensorShape(1, hkv, S, D), dt);
        Tensor v = new(new TensorShape(1, hkv, S, D), dt), o = new(new TensorShape(1, hq, S, D), dt);
        Tensor? bias = biasHeads == 0 ? null : new Tensor(new TensorShape(1, biasHeads, S, S), DType.F32);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(o, q, k, v, bias, 0.125f, true);
        return numerics is { } n ? call with { Numerics = n } : call;
    }

    private static string Plan(CudaBackend backend, string rows, in AttentionCall call) => AttentionPlanner.Plan(Rows(backend, rows), in call).RowId;

    // ── declarations ──────────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void OnlyTheGroupedEntriesDeclareGqa_AndTheMaskedRowsDeclarePerHeadBias()
    {
        CudaBackend backend = Backend();
        IEnumerable<AttentionKernelRow> cudnn = new[] { "BuildDenseRows", "BuildTokenMajorRows", "BuildTokenMajorGqaRows", "BuildFlashRows" }
            .SelectMany(b => Rows(backend, b)).Where(r => r.Capability.Id.StartsWith("cudnn.", StringComparison.Ordinal)).ToArray();
        Assert.Equal(["cudnn.causal", "cudnn.f16", "cudnn.f32.bf16", "cudnn.f32.sat", "cudnn.tokenmajor", "cudnn.tokenmajor.gqa"],
            cudnn.Select(r => r.Capability.Id).Order().ToArray());
        foreach (AttentionKernelRow row in cudnn)
            Assert.Equal(row.Capability.Id is "cudnn.causal" or "cudnn.tokenmajor.gqa", row.Capability.Gqa);
        foreach (AttentionKernelRow row in cudnn.Where(r => r.Capability.Id != "cudnn.causal"))
            Assert.True((row.Capability.Masks & AttentionMaskSupport.PerHeadBias) != 0, row.Capability.Id);
    }

    [Fact]
    public void OnlyBf16CapableRowsDeclareBf16_AndNoCudnnRowWidensKv()
    {
        CudaBackend backend = Backend();
        AttentionKernelRow[] dense = Rows(backend, "BuildDenseRows"), tokenMajor = Rows(backend, "BuildTokenMajorRows");
        AttentionKernelRow[] grouped = Rows(backend, "BuildTokenMajorGqaRows"), flash = Rows(backend, "BuildFlashRows");
        AttentionKernelCapability Of(AttentionKernelRow[] rows, string id) => rows.Single(r => r.Capability.Id == id).Capability;
        foreach (AttentionKernelCapability c in new[] { Of(dense, "cudnn.f16"), Of(tokenMajor, "cudnn.tokenmajor"), Of(grouped, "cudnn.tokenmajor.gqa") })
        {
            Assert.True((c.QueryTypes & AttentionDTypes.BF16) != 0, c.Id);
            Assert.Equal(AttentionFormats.F16 | AttentionFormats.Bf16, c.UsesQueryKey);
        }
        Assert.Equal(0, (int)(Of(dense, "cudnn.f32.sat").QueryTypes & AttentionDTypes.BF16));
        foreach (AttentionKernelCapability c in new[] { Of(dense, "cudnn.f16"), Of(dense, "cudnn.f32.sat"), Of(dense, "cudnn.f32.bf16"),
            Of(tokenMajor, "cudnn.tokenmajor"), Of(grouped, "cudnn.tokenmajor.gqa"), Of(flash, "cudnn.causal") })
            Assert.Equal(AttentionLowering.None, c.Lowering & AttentionLowering.WidenKv);
        Assert.Equal(AttentionLowering.WidenKv, Of(grouped, "permute.widen.dense").Lowering & AttentionLowering.WidenKv);
    }

    [Fact]
    public void TheGroupedCudnnRowComesBeforeTheWideningRow_OnTheGroupedRouteOnly()
    {
        CudaBackend backend = Backend();
        Assert.Equal(["cudnn.tokenmajor.gqa", "permute.widen.dense"], Rows(backend, "BuildTokenMajorGqaRows").Select(r => r.Capability.Id).ToArray());
        Assert.DoesNotContain(Rows(backend, "BuildTokenMajorRows"), r => r.Capability.Id == "cudnn.tokenmajor.gqa");
        Assert.Equal("G.1", ((DelegateAttentionRow)Rows(backend, "BuildTokenMajorGqaRows")[0]).Branch);
    }

    /// <summary>The BF16 cast row: F32 only, narrowing to BF16 without saturating, between the saturating F16 row and the tiled row.</summary>
    [Fact]
    public void TheBf16CastRow_NarrowsF32ToBf16_BetweenTheSaturatingRowAndTheTiledRow()
    {
        AttentionKernelRow[] dense = Rows(Backend(), "BuildDenseRows");
        string[] ids = dense.Select(r => r.Capability.Id).ToArray();
        Assert.Equal(Array.IndexOf(ids, "cudnn.f32.sat") + 1, Array.IndexOf(ids, "cudnn.f32.bf16"));
        Assert.Equal(Array.IndexOf(ids, "cudnn.f32.bf16") + 1, Array.IndexOf(ids, "tiled.f32"));
        AttentionKernelRow row = dense.Single(r => r.Capability.Id == "cudnn.f32.bf16");
        AttentionKernelCapability c = row.Capability;
        Assert.Equal("S.16", ((DelegateAttentionRow)row).Branch);
        Assert.Equal(AttentionDTypes.F32, c.QueryTypes);
        Assert.Equal(AttentionFormats.Bf16, c.NarrowsF32To);
        Assert.Equal(AttentionFormats.Bf16, c.UsesQueryKey);
        Assert.Equal(AttentionFormats.Bf16, c.UsesProbValue);
        Assert.False(c.Saturates);
        Assert.False(c.StridedViews);
        Assert.Equal(AttentionLowering.Cast, c.Lowering);
        Assert.All(dense.Where(r => r != row), r => Assert.Equal(AttentionFormats.None, r.Capability.NarrowsF32To));
    }

    /// <summary>An F32 call reaches the BF16 row only with the grant; with the legacy flag as well the saturating row keeps it,
    /// and with too little free VRAM for the BF16 copies the call falls to the memory-bounded tiled row.</summary>
    [Fact]
    public void AnF32Call_PlansTheBf16RowWithTheGrantOnly()
    {
        CudaBackend backend = Backend();
        AttentionCall granted = HeadMajor(DType.F32, 8, 8, numerics: AttentionNumerics.Bf16Storage);
        Assert.Equal("cudnn.f32.bf16", Plan(backend, "BuildDenseRows", in granted));
        AttentionCall plain = HeadMajor(DType.F32, 8, 8, numerics: AttentionNumerics.Exact);
        Assert.Equal("materialized", Plan(backend, "BuildDenseRows", in plain));
        Assert.Contains("cudnn.f32.bf16 (Numerics)", AttentionPlanner.Plan(Rows(backend, "BuildDenseRows"), in plain).Declined);
        AttentionCall legacy = HeadMajor(DType.F32, 8, 8);
        Assert.Equal("cudnn.f32.sat", Plan(backend, "BuildDenseRows", in legacy));
        AttentionCall both = HeadMajor(DType.F32, 8, 8, numerics: AttentionNumerics.Bf16Storage with { LegacyAssertedBounded = true });
        Assert.Equal("cudnn.f32.sat", Plan(backend, "BuildDenseRows", in both));
        using Tensor perHead = new(new TensorShape(1, 8, S, S), DType.F32);
        AttentionCall masked = granted with { Mask = AttentionMask.Additive(perHead) };
        Assert.Equal("cudnn.f32.bf16", Plan(backend, "BuildDenseRows", in masked));

        Assert.Equal("tiled.f32", Plan(Backend(freeVram: 4096), "BuildDenseRows", in granted));
        Assert.Equal("materialized", Plan(Backend(cudnn: false), "BuildDenseRows", in granted));
    }

    /// <summary>Lens's joint attention (F32, head dim 64, no mask, no F16 assertion) can only reach an F32 or BF16 row, whatever the
    /// grant or free VRAM: no row that narrows to F16 (Sage, cuDNN F16, the saturating cast) admits it.</summary>
    [Theory]
    [InlineData(false, 64UL << 30)]
    [InlineData(false, 4096UL)]
    [InlineData(true, 64UL << 30)]
    [InlineData(true, 4096UL)]
    public void LensShapedF32Call_NeverReachesAnF16Row(bool bf16Grant, ulong freeVram)
    {
        const int heads = 24, d = 64;
        Tensor q = new(new TensorShape(1, heads, S, d), DType.F32), k = new(new TensorShape(1, heads, S, d), DType.F32);
        Tensor v = new(new TensorShape(1, heads, S, d), DType.F32), o = new(new TensorShape(1, heads, S, d), DType.F32);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(o, q, k, v, null, 0.125f, allowF16: false);
        if (bf16Grant) call = call with { Numerics = AttentionNumerics.Bf16Storage };
        string row = Plan(Backend(freeVram: freeVram), "BuildDenseRows", in call);
        Assert.Contains(row, (string[])["materialized", "tiled.f32", "cudnn.f32.bf16"]);
    }

    /// <summary>The demand refuses every row that declares F16, with numerics.sdpaF16 forced on as well, and an F32 row still serves the
    /// call: the F16-narrowing materialized variant admits only a call that asserts F16.</summary>
    [Theory]
    [InlineData(false, 64UL << 30)]
    [InlineData(false, 4096UL)]
    [InlineData(true, 64UL << 30)]
    [InlineData(true, 4096UL)]
    public void LensDemand_RefusesEveryF16Row_EvenWithTheKnobForced(bool bf16Grant, ulong freeVram)
    {
        const int heads = 24, d = 64;
        Tensor q = new(new TensorShape(1, heads, S, d), DType.F32), k = new(new TensorShape(1, heads, S, d), DType.F32);
        Tensor v = new(new TensorShape(1, heads, S, d), DType.F32), o = new(new TensorShape(1, heads, S, d), DType.F32);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(o, q, k, v, null, 0.125f, allowF16: false)
            with { Numerics = AttentionNumerics.WideRange(bf16Grant) };
        CudaBackend backend = Backend(freeVram: freeVram);
        Set(backend, "_sdpaF16ForceOn", true);

        Assert.Contains(Plan(backend, "BuildDenseRows", in call), (string[])["materialized", "tiled.f32", "cudnn.f32.bf16"]);
        Assert.All(Rows(backend, "BuildDenseRows").Where(r => ((r.Capability.UsesQueryKey | r.Capability.UsesProbValue) & AttentionFormats.F16) != 0),
            r => Assert.NotEqual(AttentionExclusion.None, r.Capability.Excludes(in call)));
    }

    /// <summary>Without the demand, numerics.sdpaF16 alone takes a Lens-shaped F32 call to the saturating cuDNN F16 row, which sits ahead
    /// of the BF16 row; the demand is what keeps that row off the call.</summary>
    [Fact]
    public void SdpaF16Forced_AloneRoutesALensShapedF32CallToTheF16Row()
    {
        const int heads = 24, d = 64;
        Tensor q = new(new TensorShape(1, heads, S, d), DType.F32), k = new(new TensorShape(1, heads, S, d), DType.F32);
        Tensor v = new(new TensorShape(1, heads, S, d), DType.F32), o = new(new TensorShape(1, heads, S, d), DType.F32);
        AttentionCall granted = AttentionCall.FromScaledDotProductAttention(o, q, k, v, null, 0.125f, allowF16: false)
            with { Numerics = AttentionNumerics.Bf16Storage };
        CudaBackend backend = Backend();
        Set(backend, "_sdpaF16ForceOn", true);

        Assert.Equal("cudnn.f32.sat", Plan(backend, "BuildDenseRows", in granted));
        AttentionCall demanded = granted with { Numerics = AttentionNumerics.WideRange(bf16Storage: true) };
        Assert.Equal("cudnn.f32.bf16", Plan(backend, "BuildDenseRows", in demanded));
    }

    /// <summary>Failure state is per signature: a BF16 cast signature given up leaves the F16 D=128 path and the saturating row
    /// eligible, and an F16 signature given up leaves the BF16 cast row eligible.</summary>
    [Fact]
    public void ARefusedBf16CastSignature_LeavesTheF16Path_AndTheReverse()
    {
        static CudnnSdpaHealth Refusing(DType compute)
        {
            CudnnSdpaHealth health = new();
            CudnnSdpaSignature signature = CudnnSdpaSignature.For(compute, 8, 8, D, CudnnBias.None);
            CudnnSdpa.SdpaDtype dtype = compute == DType.BF16 ? CudnnSdpa.SdpaDtype.BF16 : CudnnSdpa.SdpaDtype.F16;
            for (int i = 0; i < CudnnSdpaHealth.RefusalsToDisable; i++)
            {
                CudnnSdpa.View view = CudnnSdpa.View.HeadMajor(1, 8, S + 8 * (i + 1), D);
                health.RecordFailure(new CudnnSdpaShape(signature, DType.F32, new(dtype, D, view, view, view, view)), permanent: true, 0);
            }
            Assert.False(health.Eligible(signature, 0));
            return health;
        }

        AttentionCall granted = HeadMajor(DType.F32, 8, 8, numerics: AttentionNumerics.Bf16Storage);
        AttentionCall fp16 = HeadMajor(DType.F16, 8, 8), legacy = HeadMajor(DType.F32, 8, 8);
        CudaBackend bf16Refused = Backend(health: Refusing(DType.BF16));
        Assert.Equal("materialized", Plan(bf16Refused, "BuildDenseRows", in granted));
        Assert.Equal("cudnn.f16", Plan(bf16Refused, "BuildDenseRows", in fp16));
        Assert.Equal("cudnn.f32.sat", Plan(bf16Refused, "BuildDenseRows", in legacy));

        CudaBackend f16Refused = Backend(health: Refusing(DType.F16));
        Assert.Equal("cudnn.f32.bf16", Plan(f16Refused, "BuildDenseRows", in granted));
        Assert.Equal("materialized.f16", Plan(f16Refused, "BuildDenseRows", in legacy));
    }

    // ── grouped token-major ───────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void AGroupedF16Call_PlansOnTheGroupedCudnnRow_WithNoWidening()
    {
        CudaBackend backend = Backend();
        AttentionCall krea2 = TokenMajor(DType.F16, 48, 12);
        AttentionPlan plan = AttentionPlanner.Plan(Rows(backend, "BuildTokenMajorGqaRows"), in krea2);
        Assert.Equal("cudnn.tokenmajor.gqa", plan.RowId);
        Assert.DoesNotContain("WidenKv", plan.Describe());
        AttentionCall llm = TokenMajor(DType.F16, 16, 8);
        Assert.Equal("cudnn.tokenmajor.gqa", Plan(backend, "BuildTokenMajorGqaRows", in llm));
    }

    [Fact]
    public void EqualHeadsAndF32_KeepTheWideningRow()
    {
        CudaBackend backend = Backend();
        AttentionCall equal = TokenMajor(DType.F16, 24, 24, AttentionRoute.TokenMajorGqa);
        Assert.Equal("permute.widen.dense", Plan(backend, "BuildTokenMajorGqaRows", in equal));
        AttentionCall f32 = TokenMajor(DType.F32, 48, 12);
        Assert.Equal("permute.widen.dense", Plan(backend, "BuildTokenMajorGqaRows", in f32));
    }

    [Fact]
    public void ABf16GroupedCall_ReachesCudnnOnlyWithTheGrant()
    {
        CudaBackend backend = Backend();
        AttentionCall plain = TokenMajor(DType.BF16, 48, 12);
        Assert.Equal("permute.widen.dense", Plan(backend, "BuildTokenMajorGqaRows", in plain));
        AttentionCall granted = TokenMajor(DType.BF16, 48, 12, numerics: GrantsBf16);
        Assert.Equal("cudnn.tokenmajor.gqa", Plan(backend, "BuildTokenMajorGqaRows", in granted));
    }

    [Fact]
    public void TheKnobTheDeadFlagAndTheSignatureHealth_GateTheGroupedRow()
    {
        AttentionCall grouped = TokenMajor(DType.F16, 48, 12);
        Assert.Equal("permute.widen.dense", Plan(Backend(cudnn: false), "BuildTokenMajorGqaRows", in grouped));
        Assert.Equal("permute.widen.dense", Plan(Backend(dead: true), "BuildTokenMajorGqaRows", in grouped));

        CudnnSdpaHealth health = new();
        CudnnSdpaSignature gqa = CudnnSdpaSignature.For(DType.F16, 48, 12, D, CudnnBias.None);
        for (int i = 0; i < CudnnSdpaHealth.RefusalsToDisable; i++)
            health.RecordFailure(new CudnnSdpaShape(gqa, DType.F16, new(CudnnSdpa.SdpaDtype.F16, D, CudnnSdpa.View.TokenMajor(1, 48, S + i, D),
                CudnnSdpa.View.TokenMajor(1, 12, S + i, D), CudnnSdpa.View.TokenMajor(1, 12, S + i, D), CudnnSdpa.View.TokenMajor(1, 48, S + i, D))),
                permanent: true, 0);
        CudaBackend backend = Backend(health: health);
        Assert.Equal("permute.widen.dense", Plan(backend, "BuildTokenMajorGqaRows", in grouped));

        // The plain fp16 path at the same head dim is untouched: equal-head token-major and head-major MHA still plan on cuDNN.
        AttentionCall mhaTokenMajor = TokenMajor(DType.F16, 24, 24);
        Assert.Equal("cudnn.tokenmajor", Plan(backend, "BuildTokenMajorRows", in mhaTokenMajor));
        AttentionCall mhaHeadMajor = HeadMajor(DType.F16, 8, 8);
        Assert.Equal("cudnn.f16", Plan(backend, "BuildDenseRows", in mhaHeadMajor));
    }

    // ── head-major rows ───────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void APerHeadBiasF16Call_PlansOnCudnn_AndAMismatchedBiasDoesNot()
    {
        CudaBackend backend = Backend();
        AttentionCall perHead = HeadMajor(DType.F16, 8, 8, biasHeads: 8);
        Assert.Equal("cudnn.f16", Plan(backend, "BuildDenseRows", in perHead));
        AttentionCall mismatch = HeadMajor(DType.F16, 8, 8, biasHeads: 4);
        Assert.Equal("materialized.f16", Plan(backend, "BuildDenseRows", in mismatch));
        AttentionCall perHeadTokenMajor = TokenMajor(DType.F16, 4, 4);
        using Tensor mask = new(new TensorShape(1, 4, S, S), DType.F32);
        perHeadTokenMajor = perHeadTokenMajor with { Mask = AttentionMask.Additive(mask) };
        Assert.Equal("cudnn.tokenmajor", Plan(backend, "BuildTokenMajorRows", in perHeadTokenMajor));
    }

    [Fact]
    public void ABf16HeadMajorCall_ReachesCudnnOnlyWithTheGrant_AndALegacyFlagIsNotAGrant()
    {
        CudaBackend backend = Backend();
        AttentionCall plain = HeadMajor(DType.BF16, 8, 8);
        Assert.Equal("materialized.f16", Plan(backend, "BuildDenseRows", in plain));
        AttentionCall legacy = HeadMajor(DType.BF16, 8, 8, numerics: AttentionNumerics.FromLegacy(true));
        Assert.Equal("materialized.f16", Plan(backend, "BuildDenseRows", in legacy));
        AttentionCall granted = HeadMajor(DType.BF16, 8, 8, numerics: GrantsBf16);
        Assert.Equal("cudnn.f16", Plan(backend, "BuildDenseRows", in granted));
        AttentionCall tokenMajor = TokenMajor(DType.BF16, 8, 8, numerics: GrantsBf16);
        Assert.Equal("cudnn.tokenmajor", Plan(backend, "BuildTokenMajorRows", in tokenMajor));
    }

    /// <summary>The materialized row addresses 2^32 score elements: up to there it keeps every call it had (SD1.5's VAE mid-block at
    /// 2048² sits exactly on it), and past it an unmasked F16 or BF16 call runs query-tiled on F32 widenings, an F32 call on the tiled
    /// F32 rows, and a call no tiled row takes is refused naming the budget, as is one whose single query row across its heads is past
    /// 2^32, which no tile height brings back. Head dim 512 is past every fused row; the masked calls keep Sq·Skv within the mask
    /// contract's Int32 bound, so only heads take them past 2^32.</summary>
    [Theory]
    [InlineData("f16-at-bound", "f16", 1, 65536, 65536, "none", "materialized")]
    [InlineData("f16-under", "f16", 8, 4096, 4096, "none", "materialized")]
    [InlineData("f16-masked-under", "f16", 2, 4096, 4096, "full", "materialized")]
    [InlineData("bf16-at-bound", "bf16", 4, 32768, 32768, "none", "materialized")]
    [InlineData("f16-past", "f16", 1, 65537, 65536, "none", "tiled.cast")]
    [InlineData("bf16-past", "bf16", 2, 46341, 46341, "none", "tiled.cast")]
    [InlineData("f32-past", "f32", 1, 65537, 65536, "none", "tiled.f32")]
    [InlineData("f32-keymask-past", "f32", 4, 40000, 40000, "key", "tiled.f32.keybias")]
    [InlineData("f32-mask-past", "f32", 4, 40000, 40000, "full", "refuse")]
    [InlineData("f16-keymask-past", "f16", 4, 40000, 40000, "key", "refuse")]
    [InlineData("f16-one-row-past", "f16", 70000, 1, 65537, "none", "refuse")]
    public void TheMaterializedRowAddresses2Pow32ScoreElements(string name, string dtype, int heads, int sq, int skv, string mask, string row)
    {
        const int d = 512;
        DType dt = dtype switch { "f16" => DType.F16, "bf16" => DType.BF16, _ => DType.F32 };
        Tensor q = new(new TensorShape(1, heads, sq, d), dt), k = new(new TensorShape(1, heads, skv, d), dt);
        Tensor v = new(k.Shape, dt), o = new(q.Shape, dt);
        Tensor? bias = mask switch
        {
            "key" => new Tensor(new TensorShape(1, 1, 1, skv), DType.F32),
            "full" => new Tensor(new TensorShape(1, 1, sq, skv), DType.F32),
            _ => null,
        };
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(o, q, k, v, bias, 0.125f, false);
        CudaBackend backend = Backend();

        Assert.Equal(row, Plan(backend, "BuildDenseRows", in call));
        Assert.Equal(name.Contains("past", StringComparison.Ordinal), CudaBackend.ScoresPastMaterialized(in call));
        if (row != "refuse") return;
        AttentionUnsupportedException refused = Assert.Throws<AttentionUnsupportedException>(
            () => AttentionPlanner.Run(Rows(backend, "BuildDenseRows"), in call));
        Assert.Contains("2^32 (4294967296)", refused.Message);
        Assert.Contains(((long)heads * sq * skv).ToString(System.Globalization.CultureInfo.InvariantCulture), refused.Message);
    }

    /// <summary>The query tile: a quarter of free VRAM in F32 scores or the height <c>numerics.sdpaTile</c> forces, never past the 2^32
    /// score elements the softmax addresses (65075 rows over 66000 keys, however much memory is free or the knob asks), no division by an
    /// empty key set, and a call whose one query row is already past them is refused rather than tiled.</summary>
    [Theory]
    [InlineData(12L << 30, 24, 4480, 4480, 0, 4480)]
    [InlineData(80L << 30, 1, 66000, 66000, 0, 65075)]
    [InlineData(1L << 30, 1, 66000, 66000, 66000, 65075)]
    [InlineData(1L << 30, 1, 66000, 66000, 100, 100)]
    [InlineData(1L << 30, 1, 64, 0, 0, 64)]
    public void TheTileHeightNeverPassesTheBound(long freeBytes, int heads, int sq, int skv, int forced, long rows)
    {
        Assert.Equal(rows, CudaBackend.TileRows((ulong)freeBytes, heads, sq, skv, forced == 0 ? null : forced));
        Assert.True(rows * heads * skv <= CudaBackend.MaterializedMaxScores);
        AttentionUnsupportedException refused = Assert.Throws<AttentionUnsupportedException>(
            () => CudaBackend.TileRows((ulong)freeBytes, 70000, 1, 65537, forced == 0 ? null : forced));
        Assert.Contains("2^32 (4294967296)", refused.Message);
    }

    [Fact]
    public void AGroupedHeadMajorCall_PlansNoDenseRow()
    {
        CudaBackend backend = Backend();
        AttentionCall grouped = HeadMajor(DType.F16, 16, 8);
        Assert.Equal("none", Plan(backend, "BuildDenseRows", in grouped));
    }
}
