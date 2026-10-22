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

    internal static CudaBackend Backend(bool cudnn = true, bool dead = false, CudnnSdpaHealth? health = null)
    {
        CudaBackend backend = (CudaBackend)RuntimeHelpers.GetUninitializedObject(typeof(CudaBackend));
        GC.SuppressFinalize(backend);   // the finalizer would tear down state the constructor never built
        Set(backend, "_sdpaCudnn", cudnn);
        Set(backend, "_cudnnSdpaDead", dead);
        Set(backend, "_cudnnSdpaHealth", health ?? new CudnnSdpaHealth());
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
        Assert.Equal(["cudnn.causal", "cudnn.f16", "cudnn.f32.sat", "cudnn.tokenmajor", "cudnn.tokenmajor.gqa"],
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
        foreach (AttentionKernelCapability c in new[] { Of(dense, "cudnn.f16"), Of(dense, "cudnn.f32.sat"), Of(tokenMajor, "cudnn.tokenmajor"),
            Of(grouped, "cudnn.tokenmajor.gqa"), Of(flash, "cudnn.causal") })
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
        Assert.Equal("materialized", Plan(backend, "BuildDenseRows", in mismatch));
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
        Assert.Equal("materialized", Plan(backend, "BuildDenseRows", in plain));
        AttentionCall legacy = HeadMajor(DType.BF16, 8, 8, numerics: AttentionNumerics.FromLegacy(true));
        Assert.Equal("materialized", Plan(backend, "BuildDenseRows", in legacy));
        AttentionCall granted = HeadMajor(DType.BF16, 8, 8, numerics: GrantsBf16);
        Assert.Equal("cudnn.f16", Plan(backend, "BuildDenseRows", in granted));
        AttentionCall tokenMajor = TokenMajor(DType.BF16, 8, 8, numerics: GrantsBf16);
        Assert.Equal("cudnn.tokenmajor", Plan(backend, "BuildTokenMajorRows", in tokenMajor));
    }

    /// <summary>The materialized row addresses 2^32 score elements: up to there it keeps every call it had (SD1.5's VAE mid-block at
    /// 2048² sits exactly on it), and past it an unmasked F16 or BF16 call runs query-tiled on F32 widenings, an F32 call on the tiled
    /// F32 rows, and a call no tiled row takes is refused naming the budget. Head dim 512 is past every fused row.</summary>
    [Theory]
    [InlineData("f16-at-bound", "f16", 1, 65536, 65536, "none", "materialized")]
    [InlineData("f16-under", "f16", 8, 4096, 4096, "none", "materialized")]
    [InlineData("f16-masked-under", "f16", 2, 4096, 4096, "full", "materialized")]
    [InlineData("bf16-at-bound", "bf16", 4, 32768, 32768, "none", "materialized")]
    [InlineData("f16-past", "f16", 1, 65537, 65536, "none", "tiled.cast")]
    [InlineData("bf16-past", "bf16", 2, 46341, 46341, "none", "tiled.cast")]
    [InlineData("f32-past", "f32", 1, 65537, 65536, "none", "tiled.f32")]
    [InlineData("f32-keymask-past", "f32", 2, 46341, 46341, "key", "tiled.f32.keybias")]
    [InlineData("f32-mask-past", "f32", 2, 46341, 46341, "full", "refuse")]
    [InlineData("f16-keymask-past", "f16", 2, 46341, 46341, "key", "refuse")]
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
        Assert.Contains(CudaBackend.MaterializedMaxScores.ToString(System.Globalization.CultureInfo.InvariantCulture), refused.Message);
    }

    [Fact]
    public void AGroupedHeadMajorCall_PlansNoDenseRow()
    {
        CudaBackend backend = Backend();
        AttentionCall grouped = HeadMajor(DType.F16, 16, 8);
        Assert.Equal("none", Plan(backend, "BuildDenseRows", in grouped));
    }
}
